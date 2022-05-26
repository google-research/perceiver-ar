"""Tests for dataset."""

from absl.testing import absltest
from absl.testing import parameterized
import dataset
import numpy as np
import tensorflow as tf


class DatasetTest(parameterized.TestCase):

  def test_dummy_task_eval(self):
    ds = dataset.load(
        dataset.DATASET_LOADERS['dummy'],
        split=dataset.Split.VALIDATION,
        is_training=False,
        batch_dims=[4],
        max_examples=None,
        max_context_length=15,
        filter_min_length=None,
        filter_max_length=None,
        is_local=True,
        include_sos=True,
        include_event_idxs=True,
        minimum_crop_length=0).as_numpy_iterator()
    batches = list(ds)
    self.assertLen(batches, 3)

    # Disable whitespace lint warning for better array formatting.
    # pylint: disable=bad-whitespace
    np.testing.assert_array_equal(
        batches[0]['events'], [
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,  1,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,  0,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75, 76,  0,  0,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75,  0,  0,  0,  0,  0,  0],
        ])
    np.testing.assert_array_equal(
        batches[0]['event_idxs'], np.expand_dims([
            [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  0,  0,  0],
            [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  0,  0,  0,  0],
            [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  0,  0,  0,  0,  0],
            [ 1,  2,  3,  4,  5,  6,  7,  8,  9,  0,  0,  0,  0,  0,  0]
        ], axis=-1))
    np.testing.assert_array_equal(batches[0]['input_length'], [12, 11, 10, 9])

    np.testing.assert_array_equal(
        batches[1]['events'], [
            [ 2, 68, 69, 70, 71, 72, 73, 74,  0,  0,  0,  0,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72, 73,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 2, 68, 69, 70, 71,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ])
    np.testing.assert_array_equal(
        batches[1]['event_idxs'], np.expand_dims([
            [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], axis=-1))
    np.testing.assert_array_equal(batches[1]['input_length'], [8, 7, 6, 5])

    np.testing.assert_array_equal(
        batches[2]['events'], [
            [ 2, 68, 69, 70,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 2, 68, 69,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 2, 68,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ])
    np.testing.assert_array_equal(
        batches[2]['event_idxs'], np.expand_dims([
            [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], axis=-1))
    np.testing.assert_array_equal(batches[2]['input_length'], [4, 3, 2, 0])
    # pylint: enable=bad-whitespace

  def test_minimum_crop_length(self):
    ds = dataset.load(
        dataset.DATASET_LOADERS['dummy'],
        split=dataset.Split.VALIDATION,
        is_training=False,
        batch_dims=[5],
        max_examples=None,
        max_context_length=15,
        filter_min_length=None,
        filter_max_length=None,
        is_local=True,
        include_sos=True,
        include_event_idxs=True,
        minimum_crop_length=5).as_numpy_iterator()
    batches = list(ds)
    self.assertLen(batches, 2)

    # Disable whitespace lint warning for better array formatting.
    # pylint: disable=bad-whitespace
    np.testing.assert_array_equal(
        batches[0]['events'], [
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,  1,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,  0,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75, 76,  0,  0,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75,  0,  0,  0,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72, 73, 74,  0,  0,  0,  0,  0,  0,  0],
        ])
    np.testing.assert_array_equal(
        batches[0]['input_length'], [12, 11, 10, 9, 8])

    np.testing.assert_array_equal(
        batches[1]['events'], [
            [ 2, 68, 69, 70, 71, 72, 73,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 2, 68, 69, 70, 71,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ])
    np.testing.assert_array_equal(batches[1]['input_length'], [7, 6, 5, 0, 0])
    # pylint: enable=bad-whitespace

  def test_sequence_shorter_than_minimum_crop(self):
    ds = dataset.load(
        dataset.DATASET_LOADERS['dummy'],
        split=dataset.Split.VALIDATION,
        is_training=False,
        batch_dims=[3],
        max_examples=None,
        max_context_length=15,
        filter_min_length=None,
        filter_max_length=None,
        is_local=True,
        include_sos=True,
        include_event_idxs=True,
        minimum_crop_length=20).as_numpy_iterator()
    batches = list(ds)
    self.assertLen(batches, 1)

    # Disable whitespace lint warning for better array formatting.
    # pylint: disable=bad-whitespace
    np.testing.assert_array_equal(
        batches[0]['events'], [
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,  1,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ])
    np.testing.assert_array_equal(batches[0]['input_length'], [12, 0, 0])
    # pylint: enable=bad-whitespace

  def test_sequence_longer_than_crop(self):
    ds = dataset.load(
        dataset.DATASET_LOADERS['dummy'],
        split=dataset.Split.VALIDATION,
        is_training=False,
        batch_dims=[8],
        max_examples=None,
        max_context_length=8,
        filter_min_length=None,
        filter_max_length=None,
        is_local=True,
        include_sos=True,
        include_event_idxs=True,
        minimum_crop_length=6).as_numpy_iterator()
    batches = list(ds)
    self.assertLen(batches, 1)

    # Disable whitespace lint warning for better array formatting.
    # pylint: disable=bad-whitespace
    np.testing.assert_array_equal(
        batches[0]['events'], [
            [71, 72, 73, 74, 75, 76, 77,  1],
            [70, 71, 72, 73, 74, 75, 76, 77],
            [69, 70, 71, 72, 73, 74, 75, 76],
            [68, 69, 70, 71, 72, 73, 74, 75],
            [ 2, 68, 69, 70, 71, 72, 73, 74],
            [ 2, 68, 69, 70, 71, 72, 73,  0],
            [ 2, 68, 69, 70, 71, 72,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0]
        ])
    np.testing.assert_array_equal(
        batches[0]['input_length'], [12, 11, 10, 9, 8, 7, 6, 0])
    # pylint: enable=bad-whitespace

  def test_training_example(self):
    ds = dataset.load(
        dataset.DATASET_LOADERS['dummy'],
        split=dataset.Split.TRAIN,
        is_training=True,
        batch_dims=[3],
        max_examples=None,
        max_context_length=10,
        filter_min_length=None,
        filter_max_length=None,
        is_local=True,
        include_sos=True,
        include_event_idxs=True,
        minimum_crop_length=0).as_numpy_iterator()
    batch = next(ds)

    # Train crops are random, so do some basic checks.
    # Sanity check shapes.
    self.assertEqual((3, 10), batch['events'].shape)
    self.assertEqual((3,), batch['input_length'].shape)
    # Ensure no all-padding examples.
    self.assertLess(np.max(np.sum(batch['events'] == 0, axis=1)), 10)
    self.assertGreater(np.min(batch['input_length']), 0)

  def test_training_short_example_long_context_and_min_crop(self):
    ds = dataset.load(
        dataset.DATASET_LOADERS['dummy'],
        split=dataset.Split.TRAIN,
        is_training=True,
        batch_dims=[3],
        max_examples=None,
        max_context_length=15,
        filter_min_length=None,
        filter_max_length=None,
        is_local=True,
        include_sos=True,
        include_event_idxs=True,
        minimum_crop_length=20).as_numpy_iterator()
    batch = next(ds)

    # Because of the long context length and minimum crop, the train examples
    # will all be the same.

    # pylint: disable=bad-whitespace
    np.testing.assert_array_equal(
        batch['events'], [
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,  1,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,  1,  0,  0,  0],
            [ 2, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,  1,  0,  0,  0],
        ])
    np.testing.assert_array_equal(batch['input_length'], [12, 12, 12])
    # pylint: enable=bad-whitespace

  def test_training_long_example_short_context_and_min_crop(self):
    ds = dataset.load(
        dataset.DATASET_LOADERS['dummy'],
        split=dataset.Split.VALIDATION,
        is_training=False,
        batch_dims=[10],
        max_examples=None,
        max_context_length=5,
        filter_min_length=None,
        filter_max_length=None,
        is_local=True,
        include_sos=True,
        include_event_idxs=True,
        minimum_crop_length=20).as_numpy_iterator()
    batches = list(ds)

    self.assertLen(batches, 1)
    batch = batches[0]

    # pylint: disable=bad-whitespace
    np.testing.assert_array_equal(
        batch['events'], [
            [74, 75, 76, 77,  1],
            [73, 74, 75, 76, 77],
            [72, 73, 74, 75, 76],
            [71, 72, 73, 74, 75],
            [70, 71, 72, 73, 74],
            [69, 70, 71, 72, 73],
            [68, 69, 70, 71, 72],
            [ 2, 68, 69, 70, 71],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
        ])
    np.testing.assert_array_equal(
        batch['input_length'], [12, 11, 10,  9,  8,  7,  6,  5,  0,  0])
    # pylint: enable=bad-whitespace

  def test_training_crops_include_final_token(self):
    ds = dataset.load(
        dataset.DATASET_LOADERS['dummy'],
        split=dataset.Split.TRAIN,
        is_training=True,
        batch_dims=[1024],
        max_examples=None,
        max_context_length=10,
        filter_min_length=None,
        filter_max_length=None,
        is_local=True,
        include_sos=True,
        include_event_idxs=True,
        minimum_crop_length=2).as_numpy_iterator()
    batch = next(ds)

    # We rely on the small training dataset size and the large batch size to
    # always result in at least one example crop including the full sequence
    # length. If this test becomes flaky, we can revisit making this more
    # deterministic somehow.
    self.assertEqual(np.max(batch['input_length']), 12)

  def test_random_mirrored_dataset(self):
    ds = dataset.RandomMirroredDataset(sequence_length=16).load(
        split=dataset.Split.TRAIN,
        is_training=True,
        include_sos=True)
    ex = next(ds.as_numpy_iterator())
    events = ex['events']
    self.assertLen(events, 16)
    self.assertEqual(dataset.SOS_ID, events[0])
    self.assertEqual(dataset.EOS_ID, events[-1])
    np.testing.assert_array_equal(events[1:8], np.flip(events[8:15]))

  def test_fixed_dataset_block_eval(self):

    class TestDataset:

      def load(self, split, is_training, include_sos):
        del split, is_training, include_sos
        ds = tf.data.Dataset.from_tensors(list(range(50, 61)))
        def create_example(events):
          event_idxs = tf.range(tf.shape(events)[0]) + 1
          event_idxs = tf.expand_dims(event_idxs, axis=-1)
          return {'events': events, 'event_idxs': event_idxs}
        return ds.map(create_example)

      @property
      def event_idx_size(self):
        return [-1]

    ds = dataset.load_block_eval(
        TestDataset(),
        split=dataset.Split.VALIDATION,
        batch_dims=[1],
        max_context_length=6,
        block_eval_window=3,
        filter_min_length=None,
        filter_max_length=None,
        include_sos=False,
        include_event_idxs=True,
    )

    inputs, input_idxs, targets = [], [], []
    for data in ds.as_numpy_iterator():
      inputs.append(list(data['inputs'][0]))
      input_idxs.append(list(data['input_idxs'][0]))
      targets.append(list(data['targets'][0]))

    self.assertSequenceEqual(
        inputs, [
            [50, 0, 0, 0, 0, 0],
            [50, 51, 52, 53, 0, 0],
            [51, 52, 53, 54, 55, 56],
            [54, 55, 56, 57, 58, 59],
        ])
    self.assertSequenceEqual(
        input_idxs, [
            [1, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 0, 0],
            [2, 3, 4, 5, 6, 7],
            [5, 6, 7, 8, 9, 10],
        ])
    self.assertSequenceEqual(
        targets, [
            [51, 0, 0, 0, 0, 0],
            [51, 52, 53, 54, 0, 0],
            [52, 53, 54, 55, 56, 57],
            [55, 56, 57, 58, 59, 60],
        ]
    )

  @parameterized.product(
      (dict(block_eval_window=2, max_context_length=2),
       dict(block_eval_window=2, max_context_length=4),
       dict(block_eval_window=2, max_context_length=8),
       dict(block_eval_window=3, max_context_length=3),
       dict(block_eval_window=3, max_context_length=6)),
      include_sos=(True,),
      seq_length=(10, 11, 13, 42),
      batch_size=(1, 2, 4),
  )
  def test_eval_loads_all_data(
      self, block_eval_window, max_context_length, include_sos, seq_length,
      batch_size):
    """Tests all targets tokens are seen with block_eval."""
    ds = dataset.DummyDataset(sequence_length=seq_length)

    # Get all possible targets
    raw_ds = ds.load(
        split=dataset.Split.VALIDATION,
        is_training=False,
        include_sos=include_sos)
    all_targets = next(raw_ds.as_numpy_iterator())['events'][1:]

    # Generate all overlapping blocks and extract the used targets
    overlapping_targets = []
    ds = dataset.load_block_eval(
        ds, split=dataset.Split.VALIDATION,
        batch_dims=[batch_size],
        max_context_length=max_context_length,
        block_eval_window=block_eval_window,
        include_sos=include_sos,
        include_event_idxs=True,
        filter_min_length=None,
        filter_max_length=None)
    for x in ds.as_numpy_iterator():
      for targets in x['targets']:
        # Remove padding
        targets = [t for t in targets if t != 0]
        # Take final block_eval_window elements only
        targets = targets[-block_eval_window:]
        overlapping_targets.append(targets)

    np.testing.assert_array_equal(all_targets,
                                  np.concatenate(overlapping_targets))

if __name__ == '__main__':
  absltest.main()
