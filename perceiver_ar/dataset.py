# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python3
"""Dataset loaders."""

import abc
import enum
import functools
from typing import Mapping, Optional, Sequence, Text

from absl import logging
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

Batch = Mapping[Text, np.ndarray]
AUTOTUNE = tf.data.experimental.AUTOTUNE

PAD = '<pad>'
EOS = '<EOS>'
SOS = '<SOS>'
RESERVED_TOKENS = [PAD, EOS, SOS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1
SOS_ID = RESERVED_TOKENS.index(SOS)  # Normally 2

# Seed for seeded-shuffle evaluation.
SEEDED_SHUFFLE_SEEDS = (42, 17)


class Split(enum.Enum):
  r"""Events dataset split."""
  TRAIN = 1
  VALIDATION = 2
  TEST = 3


class Dataset(abc.ABC):
  """Interface for datasets."""

  @abc.abstractmethod
  def load(
      self, split: Split, is_training: bool, include_sos: bool,
  ) -> tf.data.Dataset:
    """Load the dataset for the given split."""

  @property
  @abc.abstractmethod
  def vocab_size(self) -> int:
    """Returns to vocabulary size for the dataset's tokenizer."""

  @property
  def event_idx_size(self) -> Sequence[int]:
    """Returns size of event indices."""
    # -1 means the size is not known, which is fine for rotary encoding.
    return [-1]


class DownsampledImagenetWithPositionsDataset(Dataset):
  """Downsampled ImageNet dataset with image position information."""

  def __init__(self, resolution=64, ordering='pixel_raster'):
    assert resolution in [32, 64]
    self._resolution = resolution
    self._ordering = ordering

  def load(self, split: Split, is_training: bool, include_sos: bool):
    if is_training:
      input_context = tf.distribute.InputContext(
          num_input_pipelines=jax.process_count(),
          input_pipeline_id=jax.process_index())

      ds = tfds.load(
          f'downsampled_imagenet/{self._resolution}x{self._resolution}',
          split=split.name.lower(),
          shuffle_files=True,
          read_config=tfds.ReadConfig(input_context=input_context))
    else:
      # Use sub-split API for validation/test because those have only 4 shards.
      splits = tfds.even_splits(split.name.lower(), n=jax.process_count())
      ds = tfds.load(
          f'downsampled_imagenet/{self._resolution}x{self._resolution}',
          split=splits[jax.process_index()],
          shuffle_files=False)

    def parse_example(ex):
      image = tf.cast(ex['image'], tf.int32)
      if self._ordering == 'r->g->b':
        events = tf.concat(
            (tf.reshape(image[..., 0], [-1]),
             tf.reshape(image[..., 1], [-1]),
             tf.reshape(image[..., 2], [-1])), axis=0)
      elif self._ordering == 'pixel_raster':
        events = tf.reshape(image, [-1])

      events += NUM_RESERVED_TOKENS

      # All event indices start at 1 to prevent confusion with padding.
      x_event_idxs = tf.reshape(
          tf.broadcast_to(
              tf.range(self._resolution)[None, :, None] + 1,
              [self._resolution, self._resolution, 3]),
          [-1])
      y_event_idxs = tf.reshape(
          tf.broadcast_to(
              tf.range(self._resolution)[:, None, None] + 1,
              [self._resolution, self._resolution, 3]),
          [-1])
      channel_event_idxs = tf.reshape(
          tf.broadcast_to(
              [1, 2, 3],
              [self._resolution, self._resolution, 3]),
          [-1])

      event_idxs = tf.stack(
          [x_event_idxs, y_event_idxs, channel_event_idxs], axis=1)

      if self._ordering == 'r->g->b':
        # Reshape to [num_pixels, num_pixel_channels(=3), num_position_dims(=3)]
        # Split on pixel channel dim to get [num_pixels, num_position_dims(=3)]
        # position arrays for each of the RGB channels.
        event_idxs = tf.reshape(event_idxs, [-1, 3, 3])
        # Concatenate to yield a sequence with position information for
        # R then G then B values
        event_idxs = tf.concat(
            [event_idxs[:, 0], event_idxs[:, 1], event_idxs[:, 2]], axis=0)

      if include_sos:
        events = tf.concat([[SOS_ID], events], axis=0)
        event_idxs = tf.concat([[[1, 1, 1]], event_idxs + 1], axis=0)

      return {'events': events, 'event_idxs': event_idxs}

    ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)

    return ds

  @property
  def vocab_size(self) -> int:
    # R/G/B byte + special characters.
    return 256 + NUM_RESERVED_TOKENS

  @property
  def event_idx_size(self) -> Sequence[int]:
    # [res, res, 3]
    # + 1 to avoid padding confusion
    # + 1 to reserve space for SOS in case it's used.
    return [self._resolution + 2, self._resolution + 2, 5]


class DummyDataset(Dataset):
  """A small dummy dataset for testing model handling."""

  def __init__(self, sequence_length: int = 10):
    self._sequence_length = sequence_length

  def load(self, split: Split, is_training: bool, include_sos: bool):
    del split, is_training

    num_sequences = 1

    base_val = []
    # Use valid text tokens for easier visualization.
    start_value = 'A'.encode('utf-8')[0]  # 65

    for i in range(num_sequences):
      offset = i * self._sequence_length
      base_val.append(
          np.arange(
              start=start_value + offset,
              stop=start_value + offset + self._sequence_length))
    base_val = np.vstack(base_val)
    assert base_val.max() < 256

    features = tf.constant(base_val.astype(np.int32)) + NUM_RESERVED_TOKENS

    def add_sos_eos(example):
      events = example
      if include_sos:
        events = tf.concat([[SOS_ID], events], axis=0)
      events = tf.concat([events, [EOS_ID]], axis=0)

      # Start at 1 to prevent confusion with padding.
      event_idxs = tf.range(tf.shape(events)[0]) + 1
      event_idxs = tf.expand_dims(event_idxs, axis=-1)

      return {'events': events, 'event_idxs': event_idxs}

    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(add_sos_eos, num_parallel_calls=AUTOTUNE)
    return ds

  @property
  def vocab_size(self) -> int:
    # UTF-8 bytes + special characters.
    return 256 + NUM_RESERVED_TOKENS

  @property
  def event_idx_size(self) -> Sequence[int]:
    # Sequence length
    # + 1 to avoid padding confusion
    # + 1 to reserve space for SOS in case it's used.
    return [self._sequence_length + 2]


class RandomMirroredDataset(Dataset):
  """A small dummy dataset for testing model handling."""

  def __init__(self, sequence_length):
    self._sequence_length = sequence_length

  def load(self, split: Split, is_training: bool, include_sos: bool):
    del is_training

    sequence_length = self._sequence_length
    # Account for EOS and optionally SOS tokens.
    sequence_length -= 2 if include_sos else 1
    assert sequence_length % 2 == 0

    # Use separate seeds for each controller.
    seed = (jax.process_index() * max([x.value for x in Split])) + split.value
    # TF requires a tuple of 2 random numbers for its seeds.
    ds = tf.data.Dataset.zip(
        (tf.data.Dataset.random(seed=seed),
         tf.data.Dataset.random(seed=seed + 1)))

    if split != Split.TRAIN:
      # Generate ~1M eval points, assuming only the second half of the sequence
      # will be evaluated.
      eval_sequences = int(1e6 // (self._sequence_length / 2))
      ds = ds.take(eval_sequences // jax.process_count())

    def gen_mirrored_sequence(seed1, seed2):
      seed = (seed1, seed2)
      seq = tf.random.stateless_uniform(
          shape=[sequence_length // 2], seed=seed,
          minval=NUM_RESERVED_TOKENS, maxval=256 + NUM_RESERVED_TOKENS,
          dtype=tf.int32)
      mirrored_seq = tf.concat([seq, tf.reverse(seq, axis=[0])], axis=0)
      return mirrored_seq

    ds = ds.map(gen_mirrored_sequence, num_parallel_calls=AUTOTUNE)

    def add_sos_eos(example):
      events = example
      if include_sos:
        events = tf.concat([[SOS_ID], events], axis=0)
      events = tf.concat([events, [EOS_ID]], axis=0)

      # Start at 1 to prevent confusion with padding.
      event_idxs = tf.range(tf.shape(events)[0]) + 1
      event_idxs = tf.expand_dims(event_idxs, axis=-1)

      return {'events': events, 'event_idxs': event_idxs}

    ds = ds.map(add_sos_eos, num_parallel_calls=AUTOTUNE)
    return ds

  @property
  def vocab_size(self) -> int:
    # UTF-8 bytes + special characters.
    return 256 + NUM_RESERVED_TOKENS

  @property
  def event_idx_size(self) -> Sequence[int]:
    # Sequence length
    # + 1 to avoid padding confusion
    return [self._sequence_length + 1]


DATASET_LOADERS = {
    'downsampled_imagenet_w_positions':
        DownsampledImagenetWithPositionsDataset(),
    'downsampled_imagenet_32_w_positions':
        DownsampledImagenetWithPositionsDataset(resolution=32),
    'downsampled_imagenet_w_positions_r->g->b':
        DownsampledImagenetWithPositionsDataset(ordering='r->g->b'),
    'dummy': DummyDataset(),
    'random_mirrored_32': RandomMirroredDataset(sequence_length=32),
    'random_mirrored_128': RandomMirroredDataset(sequence_length=128),
    'random_mirrored_256': RandomMirroredDataset(sequence_length=256),
    'random_mirrored_288': RandomMirroredDataset(sequence_length=288),
    'random_mirrored_304': RandomMirroredDataset(sequence_length=304),
    'random_mirrored_320': RandomMirroredDataset(sequence_length=320),
    'random_mirrored_384': RandomMirroredDataset(sequence_length=384),
    'random_mirrored_512': RandomMirroredDataset(sequence_length=512),
    'random_mirrored_1024': RandomMirroredDataset(sequence_length=1024),
    'random_mirrored_2048': RandomMirroredDataset(sequence_length=2048),
    'random_mirrored_4096': RandomMirroredDataset(sequence_length=4096),
    'random_mirrored_8192': RandomMirroredDataset(sequence_length=8192),
    'random_mirrored_16384': RandomMirroredDataset(sequence_length=16384),
    'random_mirrored_32768': RandomMirroredDataset(sequence_length=32768),
    'random_mirrored_65536': RandomMirroredDataset(sequence_length=65536),
    'random_mirrored_131072': RandomMirroredDataset(sequence_length=131072),
    'random_mirrored_24578': RandomMirroredDataset(sequence_length=24578),
}


def _pad_batch(r, batch_size):
  for k, v in r.items():
    paddings = [[0, batch_size - tf.shape(v)[0]]]
    paddings = tf.concat(
        [paddings, tf.zeros([tf.rank(v) - 1, 2], tf.int32)], axis=0)
    r[k] = tf.pad(v, paddings)
  return r


def load(
    dataset: Dataset,
    split: Split,
    *,
    is_training: bool,
    # batch_dims should be:
    # [device_count, per_device_batch_size] or [total_batch_size]
    batch_dims: Sequence[int],
    max_examples: Optional[int],
    max_context_length: int,
    filter_min_length: Optional[int],
    filter_max_length: Optional[int],
    is_local: bool,
    include_sos: bool,
    # The shortest sequence from the raw data that's used as input.
    # If minimum_crop_length is 3 and the input sequences is 'abcde',
    # the possible crops are 'abc', 'abcd', 'abcde'
    # ('a' and 'ab' are not produced)
    minimum_crop_length: int,
    filter_by_length_truncation: Optional[int] = None,
    include_event_idxs: bool,
) -> tf.data.Dataset:
  """Loads the given split of the dataset."""
  if filter_min_length is not None:
    assert filter_min_length <= max_context_length
  if filter_max_length is not None:
    assert max_context_length <= filter_max_length

  if is_training and max_examples:
    raise ValueError(
        'is_training=True is not compatible with max_examples > 0')

  logging.info(
      'Loading dataset for dataset %s, split %s, is_training=%s, batch_dims=%s, '
      'max_context_length=%d, filter_min_length=%s, filter_max_length=%s',
      dataset, split.name, is_training, batch_dims, max_context_length,
      filter_min_length, filter_max_length)

  ds = dataset.load(
      split=split, is_training=is_training, include_sos=include_sos)

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  options.experimental_optimization.map_parallelization = True
  if is_training:
    options.experimental_deterministic = False

  ds = ds.with_options(options)

  if filter_min_length is not None:
    ds = ds.filter(lambda r: tf.shape(r['events'])[0] >= filter_min_length)
  if filter_max_length is not None:
    ds = ds.filter(lambda r: tf.shape(r['events'])[0] <= filter_max_length)

  ds = ds.cache()

  if is_local:
    shuffle_buffer_size = 100
  else:
    shuffle_buffer_size = 100_000

  if is_training:
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  elif max_examples:
    # If evaluating with a shuffle, seed so numbers are comparable over time.
    ds = ds.shuffle(
        buffer_size=shuffle_buffer_size, seed=SEEDED_SHUFFLE_SEEDS[0])

  # Generate crops of each text. E.g. for max_context_length 3, the crops of
  # `abcdefgh` are `a00`, `ab0`, `abc`, `bcd`, ..., `efg`, `fgh`.
  # These may be subsampled.
  def generate_crops(r, minimum_crop_length=0):
    events = r['events']
    event_idxs = r['event_idxs']
    event_length = tf.range(tf.shape(events)[0]) + 1
    original_data_length = tf.size(events)

    # If the raw sequence is abcdefg and minimum crop is length 4,
    # we see abcd, abcde, ..., abcdefg, but not a, ab, or abc.
    min_crop_length = tf.maximum(
        minimum_crop_length,
        # We always need at least one token to predict.
        2 if include_sos else 1)
    min_crop_length = tf.reduce_min(
        [max_context_length, tf.shape(events)[0], min_crop_length])

    padding_length = max_context_length - min_crop_length
    events = tf.pad(events, [[padding_length, 0]])
    event_idxs = tf.pad(event_idxs, [[padding_length, 0], [0, 0]])
    event_length = tf.pad(event_length, [[padding_length, 0]])

    if is_training:
      # For training, select a single crop.
      if tf.shape(events)[0] > max_context_length:
        crop_start = tf.random.uniform(
            [],
            minval=0,
            maxval=tf.shape(events)[0] - max_context_length + 1,
            dtype=tf.int32)
      else:
        crop_start = 0
      events = events[crop_start:crop_start + max_context_length]
      event_idxs = event_idxs[crop_start:crop_start + max_context_length]
      event_length = event_length[crop_start + max_context_length - 1]
      ds = tf.data.Dataset.from_tensors({
          'events': events,
          'event_idxs': event_idxs,
          'input_length': event_length,
          'original_data_length': original_data_length,
      })
    else:
      # For eval, generate windows for every possible context length.
      # For the case of max_examples, we don't see every example, so
      # reverse the sequences so the windows are processed with the longest
      # context first so those are more likely to end up in the shuffle buffer.
      events = tf.reverse(events, axis=[0])
      event_idxs = tf.reverse(event_idxs, axis=[0])
      event_length = tf.reverse(event_length, axis=[0])
      ds = tf.data.Dataset.from_tensor_slices({
          'events': events,
          'event_idxs': event_idxs,
          'input_length': event_length
      })
      ds = ds.window(size=max_context_length, shift=1, drop_remainder=True)

      def flat_map_fn(x):
        ds = tf.data.Dataset.zip(
            (x['events'].batch(max_context_length),
             x['event_idxs'].batch(max_context_length),
             x['input_length'].batch(max_context_length)))

        def reconstruct_dict(events, event_idxs, input_length):
          return {
              'events': tf.reverse(events, axis=[0]),
              'event_idxs': tf.reverse(event_idxs, axis=[0]),
              'input_length': input_length[0]
          }

        ds = ds.map(reconstruct_dict, num_parallel_calls=AUTOTUNE)
        return ds
      ds = ds.interleave(
          flat_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    return ds

  generate_crops_ = functools.partial(
      generate_crops, minimum_crop_length=minimum_crop_length)
  ds = ds.interleave(
      generate_crops_, num_parallel_calls=AUTOTUNE, deterministic=True)
  if is_training:
    def _random_filter_by_length(r):
      filter_prob = (tf.cast(r['original_data_length'], tf.float32) /
                     filter_by_length_truncation)
      return tf.random.uniform(()) < filter_prob

    if filter_by_length_truncation:
      logging.info('Using random filtering by length')
      ds = ds.filter(_random_filter_by_length)
    else:
      logging.info('Not using random filtering by length')
  # original_data_length only used for random filtering
  ds = ds.map(
      lambda r: {k: v for k, v in r.items() if k != 'original_data_length'})

  def remove_leading_padding(r):
    if r['input_length'] < max_context_length:
      r['events'] = r['events'][-r['input_length']:]
      r['event_idxs'] = r['event_idxs'][-r['input_length']:]
    return r
  ds = ds.map(remove_leading_padding, num_parallel_calls=AUTOTUNE)

  if max_examples:
    assert max_examples % jax.process_count() == 0

    # If evaluating with a shuffle, seed so numbers are comparable over time.
    ds = ds.shuffle(buffer_size=shuffle_buffer_size,
                    seed=SEEDED_SHUFFLE_SEEDS[1])
    ds = ds.take(max_examples // jax.process_count())
    ds = ds.cache()

  for i, batch_size in enumerate(reversed(batch_dims)):
    if i == 0:
      # Terminally pad all cropped sequences with 0s (<pad>).
      ds = ds.padded_batch(
          batch_size,
          padded_shapes={
              'events': max_context_length,
              'event_idxs': [max_context_length, len(dataset.event_idx_size)],
              'input_length': []
          },
          padding_values=PAD_ID)
    else:
      ds = ds.batch(batch_size, num_parallel_calls=AUTOTUNE)
    # Pad incomplete batches with all-0 (<pad>) entries, which is useful for
    # ensuring we can use the entire eval set.
    ds = ds.map(functools.partial(_pad_batch, batch_size=batch_size),
                num_parallel_calls=AUTOTUNE)

  def ensure_shapes(r):
    tf.ensure_shape(r['events'], list(batch_dims) + [max_context_length])
    tf.ensure_shape(
        r['event_idxs'],
        list(batch_dims) + [max_context_length, len(dataset.event_idx_size)])
    tf.ensure_shape(r['input_length'], list(batch_dims))
    return r

  # Ensure TF knows the final shape after padding partial batches.
  ds = ds.map(ensure_shapes, num_parallel_calls=AUTOTUNE)

  if not include_event_idxs:
    # If the model isn't going to use them, save some TPU memory by deleting.
    def del_event_idxs(x):
      del x['event_idxs']
      return x
    ds = ds.map(del_event_idxs, num_parallel_calls=AUTOTUNE)

  ds = ds.prefetch(AUTOTUNE)

  return ds


def load_block_eval(
    dataset: tf.data.Dataset,
    split: Split,
    *,
    # batch_dims should be:
    # [device_count, per_device_batch_size] or [total_batch_size]
    batch_dims: Sequence[int],
    max_context_length: int,
    block_eval_window: int,
    filter_min_length: Optional[int],
    filter_max_length: Optional[int],
    include_sos: bool,
    include_event_idxs: bool,
) -> tf.data.Dataset:
  """Loads a split of the dataset in sequential blocks for cached evaluation."""
  if filter_min_length is not None:
    assert filter_min_length <= block_eval_window
  if filter_max_length is not None:
    assert block_eval_window <= filter_max_length

  if split == Split.TRAIN:
    raise ValueError('Use validation or test split for evaluation.')

  logging.info(
      'Loading dataset for dataset %s, split %s, batch_dims=%s, '
      'block_eval_window=%d, filter_min_length=%s, filter_max_length=%s',
      dataset, split.name, batch_dims, block_eval_window,
      filter_min_length, filter_max_length)

  ds = dataset.load(
      split=split, is_training=False, include_sos=include_sos)

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  options.experimental_optimization.map_parallelization = True
  ds = ds.with_options(options)

  if filter_min_length is not None:
    ds = ds.filter(lambda r: tf.shape(r['events'])[0] >= filter_min_length)
  if filter_max_length is not None:
    ds = ds.filter(lambda r: tf.shape(r['events'])[0] <= filter_max_length)

  ds = ds.cache()

  def _create_inputs_and_targets(data):
    sequence = data['events']
    return {
        'inputs': sequence[:-1],
        'input_idxs': data['event_idxs'][:-1],
        'targets': sequence[1:],
    }

  ds = ds.map(_create_inputs_and_targets, num_parallel_calls=AUTOTUNE)

  # Generate sequential blocks of text of size `block_eval_window` and include
  # preceeding tokens up to total size `max_context_length`.
  # For example for a document with tokens [1, 2, 3, 4, 5], `block_eval_window`
  # = 2 and `max_context_length=3`, we get the following chunks:  [1, 0, 0],
  # [1, 2, 3], [3, 4, 5]
  def generate_blocks(data):
    seq_len = tf.shape(data['inputs'])[0]

    remainder = seq_len % block_eval_window
    num_eval_blocks = seq_len // block_eval_window
    num_eval_blocks = tf.cond(
        tf.equal(remainder, 0),
        lambda: num_eval_blocks,
        lambda: num_eval_blocks + 1)

    end_indices = (tf.range(num_eval_blocks) + 1) * block_eval_window
    end_indices = tf.cond(
        tf.equal(remainder, 0),
        lambda: end_indices,
        lambda: end_indices - (block_eval_window - remainder))
    start_indices = tf.maximum(end_indices - max_context_length, 0)
    stacked_indices = tf.stack((start_indices, end_indices), axis=-1)

    def _extract_events(inputs, x):
      start, end = inputs[0], inputs[1]
      output = x[start:end]
      output = tf.pad(
          output, [[0, max_context_length - tf.shape(output)[0]]])

      return output

    def _extract_event_idxs(inputs, x):
      start, end = inputs[0], inputs[1]
      output = x[start:end]
      output = tf.pad(
          output, [[0, max_context_length - tf.shape(output)[0]], [0, 0]])

      return output

    def _generate_lengths(inputs):
      start, end = inputs[0], inputs[1]
      num_samples = end - start
      output_idxs = tf.range(start + 1, end + 1)
      output_idxs = tf.pad(output_idxs, [[0, max_context_length - num_samples]])
      return output_idxs

    all_inputs = tf.map_fn(
        lambda idxs: _extract_events(idxs, data['inputs']), stacked_indices)
    all_input_idxs = tf.map_fn(
        lambda idxs: _extract_event_idxs(idxs, data['input_idxs']),
        stacked_indices)
    all_targets = tf.map_fn(
        lambda idxs: _extract_events(idxs, data['targets']), stacked_indices)
    input_lengths = tf.map_fn(_generate_lengths, stacked_indices)

    return {
        'inputs': all_inputs,
        'input_idxs': all_input_idxs,
        'input_lengths': input_lengths,
        'targets': all_targets
    }

  ds = ds.map(generate_blocks, num_parallel_calls=AUTOTUNE)
  ds = ds.unbatch()

  for batch_size in reversed(batch_dims):
    ds = ds.batch(batch_size, num_parallel_calls=AUTOTUNE)
    # Pad incomplete batches with all-0 (<pad>) entries, which is useful for
    # ensuring we can use the entire eval set.
    ds = ds.map(functools.partial(_pad_batch, batch_size=batch_size))

  def ensure_shapes(data):
    data['inputs'] = tf.ensure_shape(
        data['inputs'], list(batch_dims) + [max_context_length])
    data['input_lengths'] = tf.ensure_shape(
        data['input_lengths'], list(batch_dims) + [max_context_length])
    data['input_idxs'] = tf.ensure_shape(
        data['input_idxs'],
        list(batch_dims) + [max_context_length, len(dataset.event_idx_size)])
    data['targets'] = tf.ensure_shape(
        data['targets'], list(batch_dims) + [max_context_length])
    return data

  # Ensure TF knows the final shape after padding partial batches.
  ds = ds.map(ensure_shapes)

  if not include_event_idxs:
    # If the model isn't going to use them, save some TPU memory by deleting.
    def del_event_idxs(x):
      del x['input_idxs']
      return x
    ds = ds.map(del_event_idxs, num_parallel_calls=AUTOTUNE)

  ds = ds.prefetch(AUTOTUNE)

  return ds
