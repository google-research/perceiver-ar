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
"""Jaxline Perceiver AR experiment."""

import datetime
import functools
import os
import signal
import threading
import time
from typing import Mapping, Optional, Text, Tuple

from absl import app
from absl import flags
from absl import logging
from perceiver_ar import dataset
import dill
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import platform
from jaxline import utils
from perceiver_ar import losses
from ml_collections import config_dict
import numpy as np
import optax
from perceiver_ar import perceiver_ar_model
import tensorflow as tf

FLAGS = flags.FLAGS

OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
Scalars = Mapping[Text, jnp.ndarray]


def events_to_shifted_inputs_and_targets(events):
  """Return model inputs and model targets from an event sequence."""
  # events = [a, b, c, d, 0]
  # input_events = [a, b, c, 0]
  # target_events = [b, c, d, 0]

  # First truncate input_events. Otherwise the sequence would be 1 too long.
  input_events = jax.vmap(perceiver_ar_model.truncate_sequence)(events)[:, :-1]
  target_events = events[:, 1:]
  return input_events, target_events


def get_config(arg_string):
  """Return config object for training."""
  sweep = arg_string
  is_local = True
  config = base_config.get_base_config()

  # Experiment config.
  config.train_batch_size = 2
  config.train_batch_size_per_device = 2
  config.dataset_loader = 'random_mirrored_32'
  config.max_context_length = 1024
  config.num_targets = 1024
  config.num_microbatches = 1
  config.checkpoint_dir = '/tmp/perceiver_ar'
  config.train_checkpoint_all_hosts = False
  config.restore_path = ''

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              optimizer=dict(
                  base_lr=3e-4,
                  training_steps=config.get_oneway_ref('training_steps'),
                  optimizer='adam',
                  max_norm=1.0,  # <= 0 to turn off.
                  schedule_type='constant',
                  # Learning rate warm-up can be used with any schedule.
                  # Set `warmup_steps` to 0 to jump right in with no warm-up.
                  warmup_steps=1e4,
                  warmup_initial_lr=0.0,
                  cosine_decay_kwargs=dict(
                      end_value=0.0,
                  ),
                  step_decay_kwargs=dict(
                      decay_boundaries=[0.5, 0.8, 0.95],
                      decay_rate=0.1,
                  ),
                  # Optimizer-specific kwargs:
                  sgd_kwargs=dict(
                      decay=0.9,
                      nesterov=True,
                  ),
                  adam_kwargs=dict(
                      b1=0.9,
                      b2=0.999,
                      eps=1e-8,
                  ),
                  lamb_kwargs=dict(
                      b1=0.9,
                      b2=0.999,
                      eps=1e-6,
                  ),
              ),
              model=dict(
                  perceiver_ar_kwargs=dict(
                      input_embed_dim=1024,
                      position_encoding='sinusoidal',
                      learnable_position_embeddings=False,
                      position_encoding_type='rotary',
                      fraction_to_rotate=0.5,
                      fraction_heads_to_rotate=1.0,
                      max_wavelength=config.get_oneway_ref(
                          'max_context_length'),
                      use_positions_from_data=True,
                      num_transformers_per_block=12,
                      z_index_dim=config.get_oneway_ref('num_targets'),
                      num_z_channels=1024,
                      num_cross_attend_heads=16,
                      cross_attention_head_group_size=0,
                      chunked_cross_attention=False,
                      chunked_cross_query_size=1024,
                      chunked_cross_key_size=4096,
                      num_transformer_heads=16,
                      cross_attend_widening_factor=4,
                      transformer_widening_factor=4,
                      initial_query_offsetting=False,
                      initial_query_embed_style='none',
                      share_learned_initial_query=False,
                      initial_query_embed_activation_name='sq_relu',
                      initial_query_embed_num_layers=2,
                      include_initial_cross_attention=True,
                      additional_cross_attend_after_layers=[],
                      dropout_prob=0.1,
                      z_pos_enc_init_scale=0.02,
                      concat_pos=True,
                      cross_attention_shape_for_attn='kv',
                      encoder_use_query_residual=True,
                      mask_style='final_block',
                      latent_dropout_prob=0.0,
                      encoder_dropout_prob=0.0,
                      num_latents_per_position=1,
                      activation_name='sq_relu',
                      train_input_positions=0,
                      use_negative_rotary_positions=False,
                  ),
              ),
              training=dict(
                  label_smoothing=0.0,
                  batch_size=config.get_oneway_ref('train_batch_size'),
                  batch_size_per_device=config.get_oneway_ref(
                      'train_batch_size_per_device'),
                  num_microbatches=config.get_oneway_ref('num_microbatches'),
              ),
              loss=dict(
                  z_loss=1e-4,
              ),
              data=dict(
                  dataset_loader=config.get_oneway_ref('dataset_loader'),
                  max_context_length=config.get_oneway_ref(
                      'max_context_length'),
                  filter_min_length=None,
                  filter_max_length=None,
                  filter_by_length_truncation=0,
                  is_local=is_local,
                  minimum_crop_length=0,
              ),
              evaluation=dict(
                  batch_size=2,
                  max_examples=1_000_000,
                  eval_block_size=0,
                  split='validation',
              ),
          )))

  config.update_from_flattened_dict(
      {k: v for k, v in
       get_sweep(sweep, train_device_count=1, eval_device_count=1)})

  # Training loop config.
  config.training_steps = 1000000
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60
  config.save_checkpoint_interval = 10  # just for in-memory checkpointing.
  config.checkpoint_interval_type = 'steps'
  config.eval_specific_checkpoint_dir = ''
  config.best_model_eval_metric = 'eval_accuracy'

  # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
  config.lock()

  return config


EXP_CONFIG = 'experiment_kwargs.config'
MODEL_KWARGS = f'{EXP_CONFIG}.model.perceiver_ar_kwargs'


def _get_batch_sweep(train_batch_sizes, train_batch_sizes_per_device,
                     train_device_count, eval_batch_size_per_device,
                     eval_device_count):
  """Utility to build a train/eval batch sweep."""

  def _batch(batch_size, batch_size_per_device):
    per_device_batch, remainder = divmod(batch_size, train_device_count)
    assert remainder == 0
    assert per_device_batch >= 1
    num_microbatches, remainder = divmod(per_device_batch,
                                         batch_size_per_device)
    if remainder != 0:
      raise ValueError(
          f'per_device_batch ({per_device_batch}) / batch_size_per_device '
          f'({batch_size_per_device}) does not divide cleanly.')
    assert num_microbatches >= 1

    batch_size_sweep = [
        ('train_batch_size', batch_size),
        ('train_batch_size_per_device', batch_size_per_device),
        ('num_microbatches', num_microbatches),
    ]
    return batch_size_sweep

  train_batch_sweep = []
  assert len(train_batch_sizes) == 1
  for train_bs in train_batch_sizes:
    for train_bs_per_device in train_batch_sizes_per_device:
      train_batch_sweep.extend(_batch(train_bs, train_bs_per_device))

  train_batch_sweep.extend([
      (f'{EXP_CONFIG}.evaluation.batch_size',
       eval_batch_size_per_device * eval_device_count),
  ])
  return train_batch_sweep


def get_sweep(sweep_name, train_device_count, eval_device_count):
  """Build the sweep."""
  if sweep_name == 'dummy_use_positions_from_data':
    sweep_parameters = [
        ('dataset_loader', 'dummy'),
        ('max_context_length', 16),
        ('experiment_kwargs.config.evaluation.max_examples', 0),
        ('experiment_kwargs.config.evaluation.eval_block_size', 1),
        (f'{MODEL_KWARGS}.max_wavelength', 16),
        (f'{MODEL_KWARGS}.z_index_dim', 16),
        (f'{MODEL_KWARGS}.use_positions_from_data', True),

        (f'{MODEL_KWARGS}.num_z_channels', 1024),
        (f'{MODEL_KWARGS}.input_embed_dim', 1024),

        (f'{MODEL_KWARGS}.num_cross_attend_heads', 16),
        (f'{MODEL_KWARGS}.position_encoding_type', 'absolute'),
        (f'{MODEL_KWARGS}.fraction_to_rotate', 0.5),

        (f'{MODEL_KWARGS}.cross_attend_widening_factor', 4),
        (f'{MODEL_KWARGS}.num_transformers_per_block', 2),

        (f'{MODEL_KWARGS}.num_transformer_heads', 16),
        (f'{MODEL_KWARGS}.transformer_widening_factor', 4),
        (f'{MODEL_KWARGS}.activation_name', 'sq_relu'),
        (f'{EXP_CONFIG}.optimizer.max_norm', 1.0),
        (f'{EXP_CONFIG}.optimizer.optimizer', 'adam'),
        (f'{EXP_CONFIG}.optimizer.schedule_type', 'constant'),
        (f'{EXP_CONFIG}.optimizer.base_lr', 3e-4),
        (f'{EXP_CONFIG}.optimizer.warmup_steps', 1e4),
        (f'{MODEL_KWARGS}.dropout_prob', 0.1),
        (f'{MODEL_KWARGS}.latent_dropout_prob', 0.0),
        (f'{MODEL_KWARGS}.position_encoding', 'sinusoidal'),
        (f'{MODEL_KWARGS}.learnable_position_embeddings', False),
        ('random_seed', 1),
    ]
    train_batch_sizes = [8]
    train_batch_sizes_per_device = [1]
    eval_batch_size_per_device = 1
  elif sweep_name == 'random_mirrored_32':
    sweep_parameters = [
        ('dataset_loader', 'random_mirrored_32'),
        ('max_context_length', 32),
        (f'{MODEL_KWARGS}.max_wavelength', 32),

        # ----- Start active sweep ------
        (f'{EXP_CONFIG}.data.minimum_crop_length', 16),
        (f'{MODEL_KWARGS}.z_index_dim', 16),
        (f'{MODEL_KWARGS}.position_encoding_type', 'absolute'),
        (f'{MODEL_KWARGS}.position_encoding', 'sinusoidal'),
        (f'{MODEL_KWARGS}.learnable_position_embeddings', False),
        (f'{MODEL_KWARGS}.use_positions_from_data', True),
        (f'{EXP_CONFIG}.optimizer.schedule_type', 'cosine'),
        (f'{EXP_CONFIG}.optimizer.base_lr', 3e-4),
        (f'{EXP_CONFIG}.optimizer.warmup_steps', 1e3),
        (f'{EXP_CONFIG}.evaluation.max_examples', 0),
        (f'{MODEL_KWARGS}.num_transformers_per_block', 1),
        # ----- End active sweep ------

        (f'{MODEL_KWARGS}.num_z_channels', 1024),
        (f'{MODEL_KWARGS}.input_embed_dim', 1024),
        (f'{MODEL_KWARGS}.num_cross_attend_heads', 16),
        (f'{MODEL_KWARGS}.cross_attend_widening_factor', 4),

        (f'{MODEL_KWARGS}.fraction_to_rotate', 0.5),
        (f'{MODEL_KWARGS}.num_transformer_heads', 16),
        (f'{MODEL_KWARGS}.transformer_widening_factor', 4),
        (f'{MODEL_KWARGS}.activation_name', 'sq_relu'),
        (f'{EXP_CONFIG}.optimizer.max_norm', 1.0),
        (f'{EXP_CONFIG}.optimizer.optimizer', 'adam'),
        (f'{MODEL_KWARGS}.dropout_prob', 0.1),
        (f'{MODEL_KWARGS}.latent_dropout_prob', 0.0),
        ('random_seed', 1),
    ]
    train_batch_sizes = [1024]
    train_batch_sizes_per_device = [256]
    eval_batch_size_per_device = 512
  elif sweep_name == 'random_mirrored_131072':
    sweep_parameters = [
        ('dataset_loader', 'random_mirrored_131072'),
        ('max_context_length', 131072),
        (f'{MODEL_KWARGS}.max_wavelength', 131072),

        # ----- Start active sweep ------
        (f'{EXP_CONFIG}.data.minimum_crop_length', 65536),
        (f'{MODEL_KWARGS}.cross_attention_head_group_size', 4),
        (f'{MODEL_KWARGS}.z_index_dim', 1024),
        (f'{MODEL_KWARGS}.position_encoding_type', 'absolute'),
        (f'{MODEL_KWARGS}.position_encoding', 'sinusoidal'),
        (f'{MODEL_KWARGS}.learnable_position_embeddings', False),
        (f'{MODEL_KWARGS}.use_positions_from_data', True),
        (f'{EXP_CONFIG}.optimizer.schedule_type', 'cosine'),
        (f'{EXP_CONFIG}.optimizer.base_lr', 3e-4),
        (f'{EXP_CONFIG}.optimizer.warmup_steps', 1e3),
        (f'{EXP_CONFIG}.evaluation.max_examples', 0),
        # ----- End active sweep ------

        (f'{MODEL_KWARGS}.num_z_channels', 1024),
        (f'{MODEL_KWARGS}.input_embed_dim', 1024),
        (f'{MODEL_KWARGS}.num_cross_attend_heads', 16),
        (f'{MODEL_KWARGS}.num_transformers_per_block', 6),
        (f'{MODEL_KWARGS}.cross_attend_widening_factor', 4),

        (f'{MODEL_KWARGS}.fraction_to_rotate', 0.5),
        (f'{MODEL_KWARGS}.num_transformer_heads', 16),
        (f'{MODEL_KWARGS}.transformer_widening_factor', 4),
        (f'{MODEL_KWARGS}.activation_name', 'sq_relu'),
        (f'{EXP_CONFIG}.optimizer.max_norm', 1.0),
        (f'{EXP_CONFIG}.optimizer.optimizer', 'adam'),
        (f'{MODEL_KWARGS}.dropout_prob', 0.1),
        (f'{MODEL_KWARGS}.latent_dropout_prob', 0.0),
        ('random_seed', 1),
    ]
    train_batch_sizes = [4096]
    train_batch_sizes_per_device = [1]
    eval_batch_size_per_device = 4
  elif sweep_name == 'imagenet_w_positions':
    sweep_parameters = [
        # ----- Start active sweep ------
        ('max_context_length', 12416),
        (f'{MODEL_KWARGS}.max_wavelength', 12416),
        (f'{MODEL_KWARGS}.input_embed_dim', 1024),
        (f'{MODEL_KWARGS}.num_cross_attend_heads', 16),
        (f'{MODEL_KWARGS}.fraction_to_rotate', 0.5),
        (f'{MODEL_KWARGS}.num_transformers_per_block', 60),

        (f'{MODEL_KWARGS}.z_index_dim', 1024),
        (f'{EXP_CONFIG}.evaluation.eval_block_size', 512),
        (f'{EXP_CONFIG}.evaluation.max_examples', 0),

        (f'{MODEL_KWARGS}.position_encoding_type', 'absolute'),
        (f'{MODEL_KWARGS}.position_encoding', 'random'),
        (f'{MODEL_KWARGS}.learnable_position_embeddings', True),
        (f'{MODEL_KWARGS}.use_positions_from_data', True),
        ('dataset_loader', 'downsampled_imagenet_w_positions'),

        # Switch to cosine for final 50k.
        # hyper.sweep(f'{EXP_CONFIG}.optimizer.schedule_type', ['cosine']),
        (f'{EXP_CONFIG}.optimizer.schedule_type', 'constant'),
        # ----- End active sweep ------
        (f'{MODEL_KWARGS}.num_transformer_heads', 16),
        (f'{MODEL_KWARGS}.cross_attend_widening_factor', 4),
        (f'{MODEL_KWARGS}.transformer_widening_factor', 4),
        (f'{MODEL_KWARGS}.activation_name', 'sq_relu'),
        (f'{EXP_CONFIG}.optimizer.max_norm', 1.0),
        (f'{EXP_CONFIG}.optimizer.optimizer', 'adam'),
        (f'{EXP_CONFIG}.optimizer.base_lr', 3e-4),
        (f'{MODEL_KWARGS}.num_z_channels', 1024),
        (f'{EXP_CONFIG}.optimizer.warmup_steps', 1e4),
        (f'{MODEL_KWARGS}.dropout_prob', 0.1),
        (f'{MODEL_KWARGS}.latent_dropout_prob', 0.0),
        ('random_seed', 1),
    ]
    train_batch_sizes = [2048]
    train_batch_sizes_per_device = [1]
    eval_batch_size_per_device = 4
  else:
    raise ValueError(f'Unknown sweep name: {sweep_name}')

  batch_sweep = _get_batch_sweep(
      train_batch_sizes=train_batch_sizes,
      train_batch_sizes_per_device=train_batch_sizes_per_device,
      train_device_count=train_device_count,
      eval_batch_size_per_device=eval_batch_size_per_device,
      eval_device_count=eval_device_count,
  )
  sweep = sweep_parameters + batch_sweep

  return sweep


class Experiment(experiment.AbstractExperiment):
  """Music sequence experiment."""

  # A map from object properties that will be checkpointed to their name
  # in a checkpoint. Currently we assume that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_state': 'state',
      '_opt_state': 'opt_state',
  }

  def __init__(self, mode, init_rng, config):
    """Initializes experiment."""

    super(Experiment, self).__init__(mode=mode, init_rng=init_rng)

    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    self._dataset = dataset.DATASET_LOADERS[self.config.data.dataset_loader]

    # Checkpointed experiment state.
    self._params = None
    self._state = None
    self._opt_state = None

    # Input pipelines.
    self._train_input = None
    self._eval_input = None

    self.forward = hk.transform_with_state(self._forward_fn)

    # NOTE: We donate all arguments other than rng and global_step, which
    # allows JAX (on some backends) to reuse the device memory associated with
    # these inputs to store the outputs of our function.
    self._update_func = jax.pmap(
        self._update_func, axis_name='i', donate_argnums=(0, 1, 2, 3))
    # Only donate `inputs, scalars` for eval because we reuse params and state
    # for all batches.
    self._eval_batch = jax.pmap(
        self._eval_batch, axis_name='i', donate_argnums=(2, 3))

    self._sum_eval_scalars = jax.pmap(self._sum_eval_scalars, axis_name='i')

  def _num_classes(self) -> int:
    return self._dataset.vocab_size

  def _input_idx_size(self) -> int:
    return self._dataset.event_idx_size

  def _get_model_kwargs_dict(self) -> config_dict.ConfigDict:
    return self.config.model.perceiver_ar_kwargs

  def _forward_fn(
      self,
      inputs: np.ndarray,
      input_idxs: np.ndarray,
      context: np.ndarray,
      is_training: bool,
      memory_type: str = 'none',
      memory: Optional[list] = None,  # pylint: disable=g-bare-generic
      z_index_dim: Optional[int] = None,
  ) -> jnp.ndarray:
    del context  # Unused
    model = perceiver_ar_model.PerceiverAR(
        num_classes=self._num_classes(),
        input_idx_size=self._input_idx_size(),
        max_context_length=self.config.data.max_context_length,
        **self._get_model_kwargs_dict())
    output = model(
        inputs=inputs, input_idxs=input_idxs, is_training=is_training,
        memory_type=memory_type, memory=memory, z_index_dim=z_index_dim)

    return output

  def _get_lr_schedule(self):
    kwargs = self.config.optimizer
    total_steps = kwargs.training_steps
    base_lr = kwargs.base_lr
    warmup_steps = kwargs.warmup_steps
    schedule_type = kwargs.schedule_type

    if warmup_steps < 0:
      raise ValueError(f'warmup_steps must be non-negative, got {warmup_steps}')

    if warmup_steps == 0:
      init_lr = base_lr
    else:
      # Linearly ramp up to base_lr from init_lr.
      init_lr = kwargs.warmup_initial_lr
    warmup_schedule = optax.linear_schedule(
        init_value=init_lr,
        end_value=base_lr,
        transition_steps=warmup_steps)

    if schedule_type == 'steps':
      boundaries = kwargs.step_decay_kwargs.decay_boundaries
      boundaries.sort()

      decay_rate = kwargs.step_decay_kwargs.decay_rate
      boundaries_and_scales = {
          int(boundary * total_steps): decay_rate for boundary in boundaries
      }
      schedule_fn = optax.piecewise_constant_schedule(
          init_value=base_lr, boundaries_and_scales=boundaries_and_scales)
    elif schedule_type == 'cosine':
      schedule_fn = optax.cosine_decay_schedule(
          init_value=base_lr,
          decay_steps=total_steps,
          alpha=kwargs.cosine_decay_kwargs.end_value/base_lr)
    elif schedule_type == 'constant':
      schedule_fn = optax.constant_schedule(base_lr)
    elif schedule_type == 't5':
      def schedule_fn(step_count):
        return base_lr * jnp.sqrt(warmup_steps / (step_count + warmup_steps))

    else:
      raise ValueError(f'Unknown learning rate schedule: {schedule_type}')

    # Warm up to the schedule.
    schedule_fn = optax.join_schedules(
        (warmup_schedule, schedule_fn),
        boundaries=[warmup_steps])

    return schedule_fn

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step: int, rng: jnp.ndarray, *unused_args,
           **unused_kwargs):
    """See base class."""

    if self._train_input is None:
      self._initialize_train()

    inputs = next(self._train_input)

    self._params, self._state, self._opt_state, scalars = (
        self._update_func(self._params, self._state, self._opt_state, inputs,
                          rng, global_step))

    scalars = utils.get_first(scalars)
    return scalars

  def _initialize_train(self):
    self._train_input = self._build_train_input().as_numpy_iterator()

    self._lr_schedule = self._get_lr_schedule()

    # TODO(fjord): Unlike the other optimizers, adafactor will decay
    # weights for biases. It may be worth adding that option and experimenting
    # to see if it makes a difference.
    if self.config.optimizer.optimizer == 'adafactor':
      # For adafactor, use the default, built-in weight decay.
      self._optimizer = optax.adafactor(learning_rate=self._lr_schedule)
    elif self.config.optimizer.optimizer == 'sm3':
      # SM3 optax implementation supports only a constant learning rate.
      assert self.config.optimizer.schedule_type == 'constant'
      self._optimizer = optax.sm3(learning_rate=self.config.optimizer.base_lr)
    else:
      # Build the optimizer by updating the optimizer chain.
      optax_chain = []

      if self.config.optimizer.optimizer == 'sgd':
        optax_chain.extend(
            [optax.trace(**self.config.optimizer.sgd_kwargs)])
      elif self.config.optimizer.optimizer == 'adam':
        # Adam / AdamW
        optax_chain.extend([
            optax.scale_by_adam(**self.config.optimizer.adam_kwargs),
        ])
      elif self.config.optimizer.optimizer == 'lamb':
        optax_chain.extend([
            optax.scale_by_adam(**self.config.optimizer.lamb_kwargs),
            optax.scale_by_trust_ratio()
        ])
      else:
        raise ValueError(
            f'Undefined optimizer {self.config.optimizer.optimizer}')

      # Scale by the (negative) learning rate.
      optax_chain.extend([
          optax.scale_by_schedule(self._lr_schedule),
          optax.scale(-1),
      ])

      self._optimizer = optax.chain(*optax_chain)

    # Gradient clipping?
    if self.config.optimizer.max_norm > 0:
      self._optimizer = optax.chain(
          optax.clip_by_global_norm(self.config.optimizer.max_norm),
          self._optimizer)

    # Check we haven't already restored params
    if self._params is None:
      logging.info(
          'Initializing parameters rather than restoring from checkpoint.')

      def init_net(init_rng, inputs):
        context_events, input_events, input_idxs, _ = self._parse_inputs(
            inputs, is_training=True)
        return self.forward.init(
            init_rng, input_events, input_idxs, context_events,
            is_training=True)

      init_net = jax.pmap(init_net, axis_name='i')
      init_opt = jax.pmap(self._optimizer.init, axis_name='i')

      # Init uses the same RNG key on all hosts+devices to ensure everyone
      # computes the same initial state.
      init_rng = utils.bcast_local_devices(self.init_rng)

      inputs = next(self._train_input)
      # Dims = [device, microbatch, batch, events]
      # Grab only a single microbatch.
      inputs = jax.tree_map(lambda x: x[:, 0], inputs)

      self._params, self._state = init_net(init_rng, inputs)
      self._opt_state = init_opt(self._params)

  def _load_data(self, split, is_training, batch_dims, max_examples):
    """Wrapper for dataset loading compatible with 1D and 2D data."""

    model_kwargs = self._get_model_kwargs_dict()

    include_event_idxs = model_kwargs.use_positions_from_data

    if not is_training and self.config.evaluation.eval_block_size:
      assert not max_examples, 'max_examples not supported with block eval'
      assert model_kwargs.z_index_dim >= self.config.evaluation.eval_block_size

      return dataset.load_block_eval(
          dataset=self._dataset,
          split=split,
          batch_dims=batch_dims,
          block_eval_window=self.config.evaluation.eval_block_size,
          max_context_length=self.config.data.max_context_length,
          filter_min_length=self.config.data.filter_min_length,
          filter_max_length=self.config.data.filter_max_length,
          include_sos=True,
          include_event_idxs=include_event_idxs)
    else:
      max_context_length = self.config.data.max_context_length
      # All configurations need at least a single token to predict.
      minimum_crop_length = 1

      # Since the model's inputs are preshifted, load an additional token.
      max_context_length += 1
      # Since the model's inputs are preshifted, we need to include an sos
      # token at the start of the sequence to prompt the initial prediciton.
      include_sos = True
      if is_training:
        # The model predicts the last z_index_dim tokens in a batch. To avoid
        # oversampling the very first tokens in a sequence, ensure that the
        # smallest crop contains the first z_index_dim tokens.
        # i.e. instead of seeing `a`, `ab`, `abc` (and training 3x to predict
        # `a`), just see `abc`.
        minimum_crop_length = model_kwargs.z_index_dim
        filter_by_length_truncation = (
            self.config.data.filter_by_length_truncation)
      else:
        filter_by_length_truncation = None

      # Optionally override minimum crop length for both train and eval.
      if self.config.data.minimum_crop_length:
        minimum_crop_length = self.config.data.minimum_crop_length

      # Since the model's inputs are preshifted, the minimum crop length should
      # be increased by 1 because the final token in the sequence is only used
      # for the target.
      minimum_crop_length += 1

      return dataset.load(
          dataset=self._dataset,
          split=split,
          is_training=is_training,
          batch_dims=batch_dims,
          max_examples=max_examples,
          max_context_length=max_context_length,
          filter_min_length=self.config.data.filter_min_length,
          filter_max_length=self.config.data.filter_max_length,
          filter_by_length_truncation=filter_by_length_truncation,
          is_local=self.config.data.is_local,
          include_sos=include_sos,
          minimum_crop_length=minimum_crop_length,
          include_event_idxs=include_event_idxs)

  def _build_train_input(self) -> tf.data.Dataset:
    """See base class."""
    split = dataset.Split.TRAIN

    return self._load_data(
        split=split,
        is_training=True,
        batch_dims=[
            jax.local_device_count(), self.config.training.num_microbatches,
            self.config.training.batch_size_per_device
        ],
        max_examples=None)

  def _one_hot(self, value):
    """One-hot encoding potentially over a sequence of labels."""
    return jax.nn.one_hot(value, self._num_classes())

  def _parse_inputs(self, inputs, is_training: bool):
    if not is_training and self.config.evaluation.eval_block_size:
      input_events = inputs['inputs']
      if 'input_idxs' in inputs:
        input_idxs = inputs['input_idxs']
      else:
        input_idxs = None
      target_events = inputs['targets']
    else:
      input_events, target_events = events_to_shifted_inputs_and_targets(
          inputs['events'])
      if 'event_idxs' in inputs:
        input_idxs, _ = events_to_shifted_inputs_and_targets(
            inputs['event_idxs'])
      else:
        input_idxs = None
    context_events = None

    return context_events, input_events, input_idxs, target_events

  def _loss_fn(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: dataset.Batch,
      rng: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Tuple[Scalars, hk.State]]:
    # Loss for Perceiver AR.

    context_events, input_events, input_idxs, target_events = self._parse_inputs(
        inputs, is_training=True)
    output, state = self.forward.apply(
        params, state, rng, input_events, input_idxs, context_events,
        is_training=True)

    label_smoothing = self.config.training.label_smoothing
    if not (label_smoothing >= 0. and label_smoothing < 1.):
      raise ValueError(
          f'label_smoothing is {label_smoothing} and should be in [0, 1)')

    logits = output.input_events_logits
    labels = target_events

    # Subsample labels to match the prediction positions.
    @jax.vmap
    def index_positions(arr, indices):
      return jnp.take(arr, indices, axis=0)
    labels = index_positions(labels, output.latent_last_steps)

    # Note: total_z_loss is already incorporated into total_loss:
    # no need to add it in.
    loss, z_loss, weight_sum = losses.compute_weighted_cross_entropy(
        logits,
        labels,
        z_loss=self.config.loss.z_loss,
        label_smoothing=label_smoothing)
    del weight_sum

    # Negative latent indices indicate that predictions should be ignored.
    loss_mask = output.latent_last_steps >= 0
    # Ensure no invalid targets (i.e. <PAD>, which is 0) contribute to the loss.
    loss_mask = (labels > 0) * loss_mask
    loss *= loss_mask
    z_loss *= loss_mask

    loss_scalars = dict()

    def add_aux_loss_metrics(start_idx, end_idx, name):
      events_loss = jnp.sum(loss[:, start_idx:end_idx])
      events_z_loss = jnp.sum(z_loss[:, start_idx:end_idx])
      num_events = jnp.sum(loss_mask[:, start_idx:end_idx])

      loss_scalars.update({
          f'{name}_events_loss':
              events_loss,
          f'num_{name}_events':
              num_events,
          f'{name}_events_loss_per_event':
              jnp.where(num_events > 0, events_loss / num_events, 0),
          f'{name}_events_z_loss':
              events_z_loss,
          f'{name}_events_z_loss_per_event':
              jnp.where(num_events > 0, events_z_loss / num_events, 0),
      })

    add_aux_loss_metrics(0, loss.shape[1], 'next')

    # And add the total loss as well.
    loss_scalars['total_loss'] = loss_scalars['next_events_loss']
    loss_scalars['total_z_loss'] = loss_scalars['next_events_z_loss']

    return loss_scalars['total_loss'], (loss_scalars, state)

  def _update_func(
      self,
      params: hk.Params,
      state: hk.State,
      opt_state: OptState,
      inputs: dataset.Batch,
      rng: jnp.ndarray,
      global_step: int,
  ) -> Tuple[hk.Params, hk.State, OptState, Scalars]:
    """Applies an update to parameters and returns new state."""
    num_microbatches = inputs['events'].shape[0]
    num_examples = np.prod(inputs['events'].shape[:2])
    p_batches = jax.lax.psum(1, axis_name='i')
    num_examples *= p_batches

    max_context = inputs['events'].shape[-1]

    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)

    grad_rng = jax.random.split(rng, num_microbatches)

    def _microbatch_step(loop_cnt, loop_state):
      grads_accum, loss_scalars_accum, state = loop_state
      microbatch_input = jax.tree_map(lambda x: x[loop_cnt], inputs)
      grads, (loss_scalars, state) = grad_loss_fn(params, state,
                                                  microbatch_input,
                                                  grad_rng[loop_cnt])
      grads_accum = jax.tree_multimap(jnp.add, grads_accum, grads)
      loss_scalars_accum = jax.tree_multimap(jnp.add, loss_scalars_accum,
                                             loss_scalars)
      return grads_accum, loss_scalars_accum, state

    grads_accum_init = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype),
                                    params)
    loss_scalars_accum_init = {
        'total_loss': 0.0,
        'total_z_loss': 0.0,
        'next_events_loss': 0.0,
        'next_events_z_loss': 0.0,
        'num_next_events': 0.0,
        'next_events_loss_per_event': 0.0,
        'next_events_z_loss_per_event': 0.0,
    }

    loop_state_init = (grads_accum_init, loss_scalars_accum_init, state)

    if num_microbatches == 1:
      # No loop is needed, so call the function directly, which is easier for
      # XLA to optimze.
      grads_accum, loss_scalars_accum, state = _microbatch_step(
          loop_cnt=0, loop_state=loop_state_init)
    else:
      grads_accum, loss_scalars_accum, state = jax.lax.fori_loop(
          0, num_microbatches, _microbatch_step, loop_state_init)

    grads_accum = jax.lax.psum(grads_accum, axis_name='i')
    loss_scalars_accum = jax.lax.psum(loss_scalars_accum, axis_name='i')

    # Grab the learning rate to log before performing the step.
    learning_rate = self._lr_schedule(global_step)

    # Compute and apply updates via our optimizer.
    updates, opt_state = self._optimizer.update(grads_accum, opt_state, params)
    params = optax.apply_updates(params, updates)

    n_params = 0
    for k in params.keys():
      for l in params[k]:
        n_params = n_params + np.prod(params[k][l].shape)

    # Scalars to log (note: we log the results from the first device).
    scalars = {
        'learning_rate':
            learning_rate,
        'train_max_context':
            max_context,
        'n_params (M)':
            float(n_params / 1e6),
        'train_num_examples':
            num_examples,
        'global_gradient_norm':
            optax.global_norm(grads_accum),
        'global_gradient_norm_per_example':
            optax.global_norm(grads_accum) / num_examples,
    }
    for k, v in loss_scalars_accum.items():
      scalars[f'train_{k}'] = v
      scalars[f'train_{k}_per_example'] = v / num_examples

    return params, state, opt_state, scalars

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, **unused_args):
    """See base class."""
    global_step = np.array(utils.get_first(global_step))
    scalars = jax.device_get(self._eval_epoch(rng, global_step))

    logging.info('[Step %d] Eval scalars: %s', global_step, scalars)
    return scalars

  # Buckets 0 and powers of 2 from 512 to 131072.
  EVAL_INDEX_BUCKETS = np.concatenate([[0], 2**np.arange(9, 18)])
  # Pair the buckets to make ranges.
  EVAL_INDEX_BUCKETS = np.stack(
      [EVAL_INDEX_BUCKETS[:-1], EVAL_INDEX_BUCKETS[1:]]).T
  # Convert to python list to allow mixing float and int.
  EVAL_INDEX_BUCKETS = EVAL_INDEX_BUCKETS.tolist()
  # Add a final range that extends to inf.
  EVAL_INDEX_BUCKETS.append([EVAL_INDEX_BUCKETS[-1][1], np.inf])

  def _eval_batch(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: dataset.Batch,
      scalars: Scalars,
      rng: jnp.ndarray,
  ) -> Tuple[Scalars, Optional[Scalars], Optional[Scalars], Optional[Scalars]]:
    """Evaluates a batch."""
    # The evaluation loss is computed using only the immediate next-step
    # prediction, to reflect the autoregressively decomposed loglikelihood
    # estimated by the model.
    context_events, input_events, input_idxs, target_events = self._parse_inputs(
        inputs, is_training=False)
    output, _ = self.forward.apply(
        params, state, rng, input_events, input_idxs, context_events,
        is_training=False)

    eval_block_size = self.config.evaluation.eval_block_size
    if eval_block_size:
      @jax.vmap
      def index_arr_at_locs(arr, locs):
        return arr[locs]

      target_positions_for_loss = output.latent_last_steps[:, -eval_block_size:]
      targets = index_arr_at_locs(target_events, target_positions_for_loss)
      logits = output.input_events_logits[:, -eval_block_size:]
      input_lengths = index_arr_at_locs(
          inputs['input_lengths'], target_positions_for_loss)
    else:
      @jax.vmap
      def get_targets_and_logits_incremental(events, raw_logits):
        length = perceiver_ar_model.get_sequence_length(events)
        last_token_idx = jnp.maximum(0, length - 1)

        # Decoding happens only at latent locations: take the final latent,
        # which corresponds to the next event.
        logits = raw_logits[-1]

        return events[last_token_idx], logits

      targets, logits = get_targets_and_logits_incremental(
          target_events, output.input_events_logits)
      input_lengths = inputs['input_length']

    accuracy = jnp.argmax(logits, axis=-1) == targets

    loss, z_loss, weight_sum = losses.compute_weighted_cross_entropy(
        logits,
        targets)
    del z_loss, weight_sum
    num_examples = jnp.ones_like(loss, dtype=jnp.int32)

    batch_scalars = {
        'eval_loss': loss,
        'eval_accuracy': accuracy,
        'eval_num_examples': num_examples,
    }

    # Evaluate additional details, related to placement of latents in seq.
    extra_scalars = {}
    for bucket in Experiment.EVAL_INDEX_BUCKETS:
      target_token_idx = jnp.maximum(0, input_lengths - 1)
      in_seq_bucket = jnp.logical_and(target_token_idx >= bucket[0],
                                      target_token_idx < bucket[1])

      extra_scalars[
          f'eval_sequence_{bucket[0]}_{bucket[1]}_num_examples'] = in_seq_bucket
      extra_scalars[
          f'eval_sequence_{bucket[0]}_{bucket[1]}_loss'] = in_seq_bucket * loss
    batch_scalars.update(extra_scalars)

    # Don't include scores for padding events.
    batch_scalars = jax.tree_map(lambda x: x * (targets != dataset.PAD_ID),
                                 batch_scalars)

    # Sum across batch.
    batch_scalars = jax.tree_map(jnp.sum, batch_scalars)

    # Add to previous results.
    return jax.tree_multimap(jnp.add, batch_scalars, scalars)

  def _sum_eval_scalars(self, scalars: Scalars):
    # Sum scalars across hosts.
    return jax.lax.psum(scalars, axis_name='i')

  def _build_eval_input(self) -> Tuple[tf.data.Dataset, Optional[int]]:
    if self.config.evaluation.split == 'validation':
      split = dataset.Split.VALIDATION
    elif self.config.evaluation.split == 'test':
      split = dataset.Split.TEST
    else:
      raise ValueError(
          f'Unknown evaluation split: {self.config.evaluation.split}')

    if not self._eval_input:
      num_devices = jax.device_count()
      global_batch_size = self.config.evaluation.batch_size
      per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

      if ragged:
        raise ValueError(
            f'Global batch size {global_batch_size} must be divisible by '
            f'num devices {num_devices}')

      self._eval_input = self._load_data(
          split=split,
          is_training=False,
          batch_dims=[jax.local_device_count(), per_device_batch_size],
          max_examples=self.config.evaluation.max_examples)
    return self._eval_input

  def _eval_epoch(self, rng, global_step):
    """Evaluates an epoch."""
    params = self._params
    state = self._state

    tick = time.time()

    scalars = {
        'eval_loss': 0.0,
        'eval_accuracy': 0.0,
        'eval_num_examples': 0,
    }
    extra_scalars = {}
    for bucket in Experiment.EVAL_INDEX_BUCKETS:
      extra_scalars[f'eval_sequence_{bucket[0]}_{bucket[1]}_num_examples'] = (
          0.0)
      extra_scalars[f'eval_sequence_{bucket[0]}_{bucket[1]}_loss'] = 0.0
    scalars.update(extra_scalars)
    scalars = utils.bcast_local_devices(scalars)

    def calculate_metrics_from_scalars(scalars, n_batches):
      # Sum scalars across hosts.
      scalars = self._sum_eval_scalars(scalars)

      metrics = utils.get_first(scalars)
      # Ensure calculations have finished for accurate timekeeping.
      metrics = jax.tree_map(lambda x: x.block_until_ready(), metrics)

      num_examples = metrics['eval_num_examples']
      metrics['eval_loss'] /= num_examples
      metrics['eval_loss_bits'] = metrics['eval_loss'] / jnp.log(2)
      metrics['eval_accuracy'] /= num_examples
      # Metric used for best_model_eval_metric must be a float, not an ndarray.
      metrics['eval_accuracy'] = float(metrics['eval_accuracy'])

      for bucket in Experiment.EVAL_INDEX_BUCKETS:
        bucket_loss_k = (
            f'eval_sequence_{bucket[0]}_{bucket[1]}_loss')
        bucket_num_examples_k = (
            f'eval_sequence_{bucket[0]}_{bucket[1]}_num_examples')
        metrics[bucket_loss_k] = jnp.where(
            metrics[bucket_num_examples_k] > 0,
            metrics[bucket_loss_k] / metrics[bucket_num_examples_k], 0.0)

      eval_time = time.time() - tick
      metrics['eval_seconds'] = eval_time
      metrics['eval_samples_per_second'] = num_examples / eval_time
      metrics['eval_n_batches'] = n_batches
      metrics['eval_seconds_per_batch'] = eval_time / n_batches
      return metrics

    ds_iter = self._build_eval_input()
    n_batches = 0
    for i, inputs in enumerate(ds_iter.as_numpy_iterator()):
      logging.info('Eval batch %d starting.', i)
      scalars = self._eval_batch(params, state, inputs, scalars, rng)

      logging.info('Eval batch %d complete.', i)
      n_batches = i + 1
    logging.info('Completed all eval batches.')

    return calculate_metrics_from_scalars(scalars, n_batches=n_batches)


# In memory to serialized checkpoint management based on
# https://github.com/deepmind/dd_two_player_games


def restore_state_to_in_memory_checkpointer(restore_path, config):
  """Initializes experiment state from a checkpoint."""

  # Load pretrained experiment state.
  python_state_path = os.path.join(restore_path, 'checkpoint.dill')
  with open(python_state_path, 'rb') as f:
    pretrained_state = dill.load(f)
  logging.info('Restored checkpoint from %s', python_state_path)

  # Assign state to a dummy experiment instance for the in-memory checkpointer,
  # broadcasting to devices.
  dummy_experiment = Experiment(
      mode='train', init_rng=0, config=config.experiment_kwargs.config)
  for attribute, key in Experiment.CHECKPOINT_ATTRS.items():
    if key not in pretrained_state:
      continue
    setattr(dummy_experiment, attribute,
            utils.bcast_local_devices(pretrained_state[key]))

  jaxline_state = dict(
      global_step=pretrained_state['global_step'],
      experiment_module=dummy_experiment)
  snapshot = utils.SnapshotNT(0, jaxline_state)

  # Finally, seed the jaxline `utils.InMemoryCheckpointer` global dict.
  utils.GLOBAL_CHECKPOINT_DICT['latest'] = utils.CheckpointNT(
      threading.local(), [snapshot])


def _get_step_date_label(global_step):
  # Date removing microseconds.
  date_str = datetime.datetime.now().isoformat().split('.')[0]
  return f'step_{global_step}_{date_str}'


def _save_state_from_in_memory_checkpointer(
    save_path, experiment_class: experiment.AbstractExperiment):
  """Saves experiment state to a checkpoint."""
  logging.info('Saving model.')
  for (checkpoint_name, checkpoint) in utils.GLOBAL_CHECKPOINT_DICT.items():
    if not checkpoint.history:
      logging.info('Nothing to save in "%s"', checkpoint_name)
      continue

    pickle_nest = checkpoint.history[-1].pickle_nest
    global_step = pickle_nest['global_step']

    state_dict = {'global_step': global_step}
    for attribute, key in experiment_class.CHECKPOINT_ATTRS.items():
      state_dict[key] = utils.get_first(
          getattr(pickle_nest['experiment_module'], attribute))
    save_dir = os.path.join(
        save_path, checkpoint_name, _get_step_date_label(global_step))
    python_state_path = os.path.join(save_dir, 'checkpoint.dill')
    os.makedirs(save_dir, exist_ok=True)
    with open(python_state_path, 'wb') as f:
      dill.dump(state_dict, f)
    logging.info(
        'Saved "%s" checkpoint to %s', checkpoint_name, python_state_path)


def _setup_signals(save_model_fn):
  """Sets up a signal for model saving."""
  # Save a model on Ctrl+C.
  def sigint_handler(unused_sig, unused_frame):
    # Ideally, rather than saving immediately, we would then "wait" for a good
    # time to save. In practice this reads from an in-memory checkpoint that
    # only saves every 30 seconds or so, so chances of race conditions are very
    # small.
    save_model_fn()
    logging.info(r'Use `Ctrl+\` to save and exit.')

  # Exit on `Ctrl+\`, saving a model.
  prev_sigquit_handler = signal.getsignal(signal.SIGQUIT)
  def sigquit_handler(unused_sig, unused_frame):
    # Restore previous handler early, just in case something goes wrong in the
    # next lines, so it is possible to press again and exit.
    signal.signal(signal.SIGQUIT, prev_sigquit_handler)
    save_model_fn()
    logging.info(r'Exiting on `Ctrl+\`')

    # Re-raise for clean exit.
    os.kill(os.getpid(), signal.SIGQUIT)

  signal.signal(signal.SIGINT, sigint_handler)
  signal.signal(signal.SIGQUIT, sigquit_handler)


def main(argv, experiment_class: experiment.AbstractExperiment):

  # Maybe restore a model.
  restore_path = FLAGS.config.get('restore_path', None)
  if restore_path:
    restore_state_to_in_memory_checkpointer(restore_path, config=FLAGS.config)

  # Maybe save a model.
  save_dir = os.path.join(FLAGS.config.checkpoint_dir, 'models')
  if FLAGS.config.one_off_evaluate:
    save_model_fn = lambda: None  # No need to save checkpoint in this case.
  else:
    save_model_fn = functools.partial(
        _save_state_from_in_memory_checkpointer, save_dir, experiment_class)
  _setup_signals(save_model_fn)  # Save on Ctrl+C (continue) or Ctrl+\ (exit).

  try:
    platform.main(experiment_class, argv)
  finally:
    save_model_fn()  # Save at the end of training or in case of exception.


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(lambda argv: main(argv, Experiment))
