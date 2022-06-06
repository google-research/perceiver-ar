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
"""Perceiver AR architecture and components."""

import functools
import math
from typing import Any, List, Optional, Sequence, Tuple

import chex
import flax
import flax.linen
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@flax.struct.dataclass
class Masks:
  encoder: chex.Array
  processor: chex.Array


@flax.struct.dataclass
class Output:
  input_events_logits: chex.Array
  encoder_mask: chex.Array
  processor_mask: chex.Array
  latent_last_steps: chex.Array
  perceiver_state: Sequence[chex.Array]


def get_sequence_length(sequence):
  """Return the length of non-zero entries in the sequence."""
  # Return the first index where a 0 occurs.
  length = jnp.argmax(sequence == 0)

  # If argmax returns 0, that means that either
  # 1) No 0s were found, and the sequence length is the full length of the array
  # 2) There's padding immediately at the beginning, indicating that the array
  #    is all padding and the sequence length is 0.
  length = jnp.where(jnp.logical_and(length == 0, sequence[0] != 0),
                     sequence.shape[0], length)
  return length


def truncate_sequence(sequence):
  """Replace final token in sequence with padding."""
  length = get_sequence_length(sequence)
  sequence = sequence.at[jnp.maximum(0, length - 1)].set(0)
  return sequence


def fill_diagonal(a, val):
  """Fill the diagonal of the last two dimensions of an array with a value."""
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)


@jax.vmap
def make_positions_terminal_relative(pos_seq, input_seq):
  """Convert positions or position encodings to terminal-relative coords."""
  # From [0, 1, 2, ..., T, x, x, ..., x] to:
  #                                     [T, ..., 2, 1, 0, x, x, ..., x]
  #                                            last input ^
  seq_len = get_sequence_length(input_seq)
  pos_seq = jnp.flip(pos_seq)
  return jnp.roll(pos_seq, seq_len, axis=0)


#  -----------------------------------------------------------
#  ----------------------  Primitives  -----------------------
#  -----------------------------------------------------------


def conv_1d(
    output_channels,
    init_scale=1.0,
    with_bias=True,
    name=None):
  """A 1D convolution."""
  return hk.Linear(
      output_size=output_channels,
      with_bias=with_bias,
      w_init=hk.initializers.VarianceScaling(init_scale),
      name=name)


def mlp(
    num_hiddens,
    num_layers=2,
    init_scale=1.0,
    with_bias=True,
    activation_fn=jax.nn.relu,
    name=None):
  """A simple MLP with non-linearities."""
  layers = []
  for _ in range(num_layers):
    layers += [conv_1d(num_hiddens, init_scale, with_bias), activation_fn]
  return hk.Sequential(layers=layers, name=name)


def layer_norm(x):
  return hk.LayerNorm(
      axis=-1, create_scale=True, create_offset=True, use_fast_variance=True)(x)


def generate_sinusoidal_features(size,
                                 max_len=2048,
                                 min_scale=1.0,
                                 max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
    size: embedding size.
    max_len: maximum possible length for the input.
    min_scale: float: minimum frequency-scale in sine grating.
    max_scale: float: maximum frequency-scale in sine grating.

  Returns:
    output: init function returning `(max_len, size)`
  """

  pe = np.zeros((max_len, size), dtype=np.float32)
  position = np.arange(0, max_len)[:, np.newaxis]
  scale_factor = -np.log(max_scale / min_scale) / (size // 2 - 1)
  div_term = min_scale * np.exp(np.arange(0, size // 2) * scale_factor)
  pe[:, :size // 2] = np.sin(position * div_term)
  pe[:, size // 2: 2 * (size // 2)] = np.cos(position * div_term)
  return jnp.array(pe)


def generate_linear_features(size,
                             max_len=2048):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
    size: embedding size.
    max_len: maximum possible length for the input.

  Returns:
    output: init function returning `(max_len, size)`
  """
  position = np.arange(0, max_len)[:, np.newaxis]
  return jnp.broadcast_to(position, (max_len, size)).astype(jnp.float32)


def generate_fourier_features(pos, n_bands, max_res=224, concat_pos=True):
  """Generate a Fourier frequency position encoding with linear spacing.

  Args:
    pos: The position of n points in d dimensional space.
      A jnp array of shape [n, d].
    n_bands: The number of bands (K) to use.
    max_res: The maximum resolution (i.e. the number of pixels per dim).
    concat_pos: Concatenate the input position encoding to the Fourier features?
  Returns:
    embedding: A 1D jnp array of shape [n, d * (1 + 2 * n_bands)].
      Output dimensions are ordered as
      [dim_1, dim_2, ..., dim_d,
       sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ...,
       sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d),
       cos(pi*f_1*dim_1), ..., cos(pi*f_K*dim_1), ...,
       cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)],
     where dim_i is pos[:, i] and f_k is the kth frequency band.
  """
  min_freq = 1.0
  # Nyquist frequency at the target resolution:
  max_freq = max_res / 2
  freq_bands = jnp.linspace(min_freq, max_freq, num=n_bands, endpoint=True)

  # Get frequency bands for each spatial dimension.
  pos_freq_bands = jnp.einsum('nd, k->ndk', pos, freq_bands)
  pos_freq_bands = jnp.reshape(pos_freq_bands,
                               [-1, np.prod(pos_freq_bands.shape[1:])])

  # Output is size [n, 2 * d * n_bands]
  encoding = jnp.concatenate(
      [jnp.sin(jnp.pi * pos_freq_bands),
       jnp.cos(jnp.pi * pos_freq_bands)], axis=-1)
  # Concatenate the raw input positions.
  if concat_pos:
    encoding = jnp.concatenate([pos, encoding], axis=-1)
  return encoding


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
  """Generate an array of position indices for an N-D input array.

  Args:
    index_dims: The shape of the index dimensions of the input array.
    output_range: The min and max values taken by each input index dimension.
  Returns:
    A jnp array of shape [index_dims[0], index_dims[1], .., index_dims[-1], N].
  """
  def _linspace(n_xels_per_dim):
    return jnp.linspace(
        output_range[0], output_range[1],
        num=n_xels_per_dim,
        endpoint=True, dtype=jnp.float32)

  dim_ranges = [
      _linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
  array_index_grid = jnp.meshgrid(*dim_ranges, indexing='ij')

  return jnp.stack(array_index_grid, axis=-1)


def make_block_causal_masks(
    inputs,
    latent_index_dim: int,
    latents_per_position: int,
    batch_size: int,
    mask_style: str,
    rng_key,
    latent_dropout_prob: float,
    is_training: bool):
  """Constructs block-causal attention masks."""
  # Latents divide the sequence as evenly as possible.
  input_index_dim = inputs.shape[1]

  def batch_broadcast(arr):
    return jnp.broadcast_to(arr, (batch_size,) + arr.shape)

  input_id = jnp.arange(start=0, stop=input_index_dim, dtype=jnp.float32)
  input_id = batch_broadcast(input_id)

  if mask_style == 'final_block':
    # Place all latents at the end with a stride of 1. Only one latent is
    # placed at each trailing position past the first, invalid ones are
    # discarded.
    assert latent_index_dim % latents_per_position == 0
    @jax.vmap
    def get_steps(events, rng_key):
      del rng_key
      sequence_length = get_sequence_length(events)
      num_unique_positions = latent_index_dim // latents_per_position
      last_steps = sequence_length * jnp.ones(
          [num_unique_positions], dtype=jnp.int32)
      offsets = jnp.arange(start=-num_unique_positions, stop=0, step=1)
      last_steps += offsets
      last_steps = jnp.maximum(last_steps, -1)  # -1 means invalid
      return last_steps
  else:
    raise ValueError(f'Unknown mask_style: {mask_style}.')

  # latent_last_steps is B x latent_index_dim // latents_per_position x C
  # It's used for position attention to avoid duplicated computation.
  # all_latent_positions is B x latent_index_dim x C
  # It's used to construct masks (except the loss mask), because it shows
  # how latents are related to each other.
  rng_key, steps_key = jax.random.split(rng_key)
  steps_keys = jax.random.split(steps_key, num=inputs.shape[0])
  latent_last_steps = get_steps(inputs, steps_keys)
  all_latent_positions = jnp.repeat(
      latent_last_steps, repeats=latents_per_position, axis=1)

  encoder_mask_raw = flax.linen.make_attention_mask(
      query_input=all_latent_positions, key_input=input_id,
      pairwise_fn=jnp.greater_equal, dtype=inputs.dtype)

  # Mask invalid inputs as well:
  input_mask = inputs > 0
  input_mask_array = flax.linen.make_attention_mask(
      query_input=jnp.ones(
          [batch_size, latent_index_dim], dtype=inputs.dtype),
      key_input=input_mask)
  encoder_mask_final = encoder_mask_raw * input_mask_array

  # Latents can pool from latents with the same or earlier final inputs.
  processor_mask = flax.linen.make_attention_mask(
      query_input=all_latent_positions, key_input=all_latent_positions,
      pairwise_fn=jnp.greater_equal, dtype=inputs.dtype)

  # Mask any invalid latents (i.e. with negative index):
  key_is_valid = (all_latent_positions >= 0).astype(
      all_latent_positions.dtype)
  valid_latent_mask = flax.linen.make_attention_mask(
      query_input=jnp.ones([batch_size, latent_index_dim]),
      key_input=key_is_valid)
  processor_mask = flax.linen.combine_masks(
      processor_mask, valid_latent_mask)

  if is_training:
    # Drop out latents: they can't influence any other latents.
    rng_key, latent_dropout_key = jax.random.split(rng_key)
    keep_rate = 1.0 - latent_dropout_prob
    latent_dropout_keys = jax.random.bernoulli(
        latent_dropout_key, keep_rate, shape=[batch_size, latent_index_dim])
    latent_dropout_mask = flax.linen.make_attention_mask(
        query_input=jnp.ones([batch_size, latent_index_dim]),
        key_input=latent_dropout_keys)
    processor_mask = flax.linen.combine_masks(
        processor_mask, latent_dropout_mask)

  # Force diagonal to be unmasked.
  processor_mask = fill_diagonal(processor_mask, 1.0)

  masks = Masks(encoder=encoder_mask_final, processor=processor_mask)

  return masks, latent_last_steps


def _make_rotation_matrices(
    x: jnp.ndarray,
    max_wavelength: int,
    positions: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Builds the cosine and sine matrices used to compute rotary embeddings.

  Args:
    x: The array the rotary embeddings will be applied to.
    max_wavelength: Maximum wavelength that will appear in sin/cosine waveforms.
      This specifies the maximum sequence length for identifying unique
      positions.
    positions: A [B, T] tensor of positions.
  Returns:
    cos_matrix: [B, 1, T, head_dim] cosine component of the embedding rotation.
    sin_matrix: [B, 1, T, head_dim] sine component of the embedding rotation.
  """
  batch_size, seq_len, _, head_dim = x.shape

  # head_dim is assumed to be constructed/padded so it's even
  assert head_dim % 2 == 0

  # Generated log-spaced wavelengths between 1 and the max_wavelength.
  num_bands = head_dim // 2
  freq = max_wavelength**((2./head_dim)*jnp.linspace(0, num_bands, num_bands))
  inv_freq = 1./freq
  inv_freq = jnp.repeat(inv_freq, 2, axis=0)  # 2x for sin / cos

  radians = jnp.einsum('bi,j -> bij', positions, inv_freq)  # [T, head_dim]
  radians = jnp.reshape(radians, (batch_size, 1, seq_len, head_dim))
  return jnp.cos(radians), jnp.sin(radians)


def _splice_array(x: jnp.ndarray) -> jnp.ndarray:
  """Reorders the embedding dimension of an array, to make rotation easier."""
  # head_dim is assumed to be constructed/padded so it's even
  assert x.shape[-1] % 2 == 0

  even_dims = x[..., ::2]
  odd_dims = x[..., 1::2]
  return jnp.stack((-odd_dims, even_dims), axis=-1).reshape(x.shape)


def _apply_rotary_encoding(
    x: jnp.ndarray,
    max_wavelength: int,
    positions: jnp.ndarray,
) -> jnp.ndarray:
  """Applies the rotary embedding matrix to an input array.

  Computes R*x, the multiplication between the rotation matrix R, and input x.

  Args:
    x: Array of shape [B, T, num_heads, head_dim]
    max_wavelength: Maximum wavelength that will appear in sin/cosine waveforms.
      This specifies the maximum sequence length for identifying unique
      positions.
    positions: A [B, T] tensor of positions.
  Returns:
    Array of rotary encoded input, of shape [B, T, num_heads, head_dim].
  """
  # {cos, sin}_matrix are [B, 1, T, head_dim]
  cos_matrix, sin_matrix = _make_rotation_matrices(
      x, max_wavelength, positions)
  # [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
  x = jnp.moveaxis(x, -2, -3)
  # Apply the rotation.
  rotary_embeddings = x * cos_matrix + _splice_array(x) * sin_matrix
  # [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim]
  return jnp.moveaxis(rotary_embeddings, -3, -2)


def _apply_rotary_encoding_to_subset(
    x: jnp.ndarray,
    fraction_to_rotate: float,
    fraction_heads_to_rotate: float,
    max_wavelength: int,
    positions: jnp.ndarray,
) -> jnp.ndarray:
  """Applies a rotary positional encoding to a subset of dimensions."""
  if fraction_to_rotate > 1.0 or fraction_to_rotate <= 0.0:
    raise ValueError(
        f'fraction_to_rotate must be in (0, 1], got {fraction_to_rotate}.')
  _, _, num_heads, dim_per_head = x.shape

  def _to_even(x):
    return math.floor(x / 2.) * 2
  num_rotated_channels = _to_even(dim_per_head * fraction_to_rotate)

  num_rotated_heads = math.floor(fraction_heads_to_rotate * num_heads)

  if num_rotated_heads != num_heads:
    x_unrotated = x[..., num_rotated_heads:, :]
    x = x[..., :num_rotated_heads, :]

  if num_rotated_channels == dim_per_head:
    x = _apply_rotary_encoding(x, max_wavelength, positions)
  else:
    x_r = x[..., :num_rotated_channels]
    x_p = x[..., num_rotated_channels:]
    x_r = _apply_rotary_encoding(x_r, max_wavelength, positions)
    x = jnp.concatenate((x_r, x_p), axis=-1)

  if num_rotated_heads != num_heads:
    x = jnp.concatenate((x, x_unrotated), axis=-2)
  return x


#  -----------------------------------------------------------
#  -----------------------  Modules  -------------------------
#  -----------------------------------------------------------


class TrainablePositionEncoding(hk.Module):
  """Trainable position encoding."""

  def __init__(self, index_dim, num_channels, init_scale=0.02, name=None):
    super(TrainablePositionEncoding, self).__init__(name=name)
    self._index_dim = index_dim
    self._num_channels = num_channels
    self._init_scale = init_scale

  def __call__(self, batch_size):
    pos_embs = hk.get_parameter(
        'pos_embs', [self._index_dim, self._num_channels],
        init=hk.initializers.TruncatedNormal(stddev=self._init_scale))

    # If inputs shape > 2, broadcast to batch.
    if batch_size is not None:
      pos_embs = jnp.broadcast_to(pos_embs, (batch_size,) + pos_embs.shape)
    return pos_embs


@chex.dataclass
class AttentionState:
  """State of the Attention module."""
  k: jnp.DeviceArray  # [B, T, H, D_k]
  v: jnp.DeviceArray  # [B, T, H, D_v]
  kv_positions: Optional[jnp.DeviceArray]  # [B, T]
  memory_mask: Optional[jnp.DeviceArray]  # [B, T]


class Attention(hk.Module):
  """Multi-headed {cross, self}-attention."""

  def __init__(self,
               dropout_prob,
               position_encoding_type,
               fraction_to_rotate,
               max_wavelength,
               num_heads=8,
               init_scale=1.0,
               with_final_bias=True,
               final_init_scale_multiplier=1.,
               channels_per_head=None,
               qkv_multi_head=False,
               qk_channels=None,
               v_channels=None,
               output_channels=None,
               fraction_heads_to_rotate=1.0,
               name=None):
    super(Attention, self).__init__(name=name)
    self._num_heads = num_heads
    self._init_scale = init_scale
    self._with_final_bias = with_final_bias
    self._final_init_scale = final_init_scale_multiplier * init_scale
    self._dropout_prob = dropout_prob
    self._qkv_multi_head = qkv_multi_head

    # If none of these are passed, the Q input determines the output shape:
    self._qk_channels = qk_channels
    self._v_channels = v_channels
    self._output_channels = output_channels

    self._position_encoding_type = position_encoding_type
    self._fraction_to_rotate = fraction_to_rotate
    self._fraction_heads_to_rotate = fraction_heads_to_rotate
    self._max_wavelength = max_wavelength

  @hk.transparent
  def _multihead_linear(self, inputs: jnp.DeviceArray, hidden_size: int,
                        name: str):
    linear = hk.Linear(
        self._num_heads * hidden_size,
        with_bias=True,
        w_init=hk.initializers.VarianceScaling(scale=self._init_scale),
        name=name)
    out = linear(inputs)
    return jnp.reshape(out, inputs.shape[:-1] + (self._num_heads, hidden_size))

  @hk.transparent
  def _multihead_bias(self, hidden_size: int, name=Optional[str]):
    # Store bias as flat parameter for ZeRO.
    bias = hk.get_parameter(
        name, [self._num_heads * hidden_size],
        init=hk.initializers.RandomNormal(stddev=0.02))
    bias = jnp.reshape(bias, [self._num_heads, hidden_size])
    return bias

  @hk.transparent
  def _rotary_position_embeddings(
      self,
      q: jnp.DeviceArray,
      k: jnp.DeviceArray,
      q_positions: jnp.DeviceArray,
      kv_positions: jnp.DeviceArray,
      use_bias: bool = False,
  ) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
    """Compute attention using rotary encodings."""
    head_dim = q.shape[-1]
    rotary_queries = _apply_rotary_encoding_to_subset(
        q, self._fraction_to_rotate, self._fraction_heads_to_rotate,
        self._max_wavelength, q_positions)
    rotary_keys = _apply_rotary_encoding_to_subset(
        k, self._fraction_to_rotate, self._fraction_heads_to_rotate,
        self._max_wavelength, kv_positions)

    if use_bias:
      rotary_bias = self._multihead_bias(head_dim, 'rotary_bias')
      rotary_queries += rotary_bias
    return rotary_queries, rotary_keys

  @hk.transparent
  def _attend(self, q, k, v, mask: Optional[jnp.ndarray],
              q_positions: Optional[jnp.ndarray],
              kv_positions: Optional[jnp.ndarray],
              is_cross_attend: bool,
              is_training: bool):
    """Computes multi-head attention using a query, key and value.

    Args:
      q: Query with shape [batch, q_indices, num_heads, head_dim].
      k: Key with shape [batch, kv_indices, num_heads, head_dim].
      v: Value with shape [batch, kv_indices, num_heads, head_dim].
      mask: optional attention mask.
      q_positions: A [batch, q_indices] tensor of query positions for
        rotary attention.
      kv_positions: A [batch, kv_indices] tensor of key/value positions
        for rotary attention.
      is_cross_attend: whether the queries and keys come from the same array.
      is_training: whether this is used in a training context. If not, attention
        masks will be saved as state.
    Returns:
      Output of the attention with shape [batch, q_indices, hiddens]
    """
    batch, q_indices, num_heads, head_dim = q.shape
    hiddens = num_heads * head_dim

    if self._position_encoding_type == 'absolute':
      attention = jnp.einsum('bthd,bThd->bhtT', q, k)
    elif self._position_encoding_type == 'rotary':
      if q_positions is None or kv_positions is None:
        raise ValueError(
            'Both q_positions and kv_positions must be specified '
            'for rotary attention.')
      rotary_queries, rotary_keys = self._rotary_position_embeddings(
          q, k, q_positions, kv_positions)
      attention = jnp.einsum(
          'bthd,bThd->bhtT', rotary_queries, rotary_keys)
    else:
      raise ValueError('Invalid position encoding type.')

    scale = 1. / math.sqrt(head_dim)
    attention *= scale
    if mask is not None:
      mask = jnp.broadcast_to(
          mask, (mask.shape[0],) + (num_heads,) + mask.shape[2:])
      assert mask.shape == attention.shape
      # Mask values of 0.0 indicate that an entry will be masked.
      attention = jnp.where(mask, attention, -1e30)

    # Uncomment these for attention analysis.
    # They use extra memory, so leaving off for now.
    # if not is_training:
    #   hk.set_state('attention', attention)
    normalized = jax.nn.softmax(attention)
    # if not is_training:
    #   hk.set_state('attention_normalized', normalized)
    if is_training:
      normalized = hk.dropout(hk.next_rng_key(), self._dropout_prob, normalized)
    summed = jnp.einsum('bhtT,bThd->bthd', normalized, v)
    return jnp.reshape(summed, [batch, q_indices, hiddens])

  @hk.transparent
  def _query_chunk_attention(self,
                             query,
                             key,
                             value,
                             mask,
                             precision,
                             is_training: bool,
                             key_chunk_size: int = 4096):
    """Multi-head dot product attention with a limited number of queries."""
    num_kv, k_features = key.shape
    v_features = value.shape[-1]

    qk_channels_per_head = k_features // self._num_heads
    v_channels_per_head = v_features // self._num_heads

    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(qk_channels_per_head)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value, mask, dropout_rng):
      query = query.reshape(
          query.shape[0], self._num_heads, qk_channels_per_head)
      key = key.reshape(
          key.shape[0], self._num_heads, qk_channels_per_head)
      value = value.reshape(
          value.shape[0], self._num_heads, v_channels_per_head)

      attn_weights = jnp.einsum('qhd,khd->qhk', query, key, precision=precision)
      mask = jnp.broadcast_to(jnp.moveaxis(mask, 0, 1), attn_weights.shape)
      attn_weights = jnp.where(mask, attn_weights, -1e30)
      max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
      max_score = jax.lax.stop_gradient(max_score)
      exp_weights = jnp.exp(attn_weights - max_score)
      if is_training:
        exp_weights = hk.dropout(dropout_rng, self._dropout_prob, exp_weights)
      exp_values = jnp.einsum(
          'vhf,qhv->qhf', value, exp_weights, precision=precision)
      return (exp_values, exp_weights.sum(axis=-1),
              max_score.reshape((query.shape[0], self._num_heads)))

    def chunk_scanner(chunk_idx):
      key_chunk = jax.lax.dynamic_slice(
          key, (chunk_idx, 0),
          slice_sizes=(key_chunk_size, k_features))
      value_chunk = jax.lax.dynamic_slice(
          value, (chunk_idx, 0),
          slice_sizes=(key_chunk_size, v_features))
      mask_chunk = jax.lax.dynamic_slice(
          mask, (0, 0, chunk_idx),
          slice_sizes=(1, query.shape[0], key_chunk_size))
      return summarize_chunk(
          query, key_chunk, value_chunk, mask_chunk, hk.next_rng_key())

    _, (chunk_values, chunk_weights, chunk_max) = hk.scan(
        lambda _, x: ((), chunk_scanner(x)), (),
        xs=jnp.arange(0, num_kv, key_chunk_size))

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    all_values /= all_weights
    return all_values.reshape(
        all_values.shape[0], all_values.shape[1] * all_values.shape[2])

  @hk.transparent
  def _chunked_attend(self,
                      q,
                      k,
                      v,
                      mask: Optional[jnp.ndarray],
                      q_positions: Optional[jnp.ndarray],
                      kv_positions: Optional[jnp.ndarray],
                      is_training: bool,
                      precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
                      query_chunk_size: int = 1024,
                      key_chunk_size: int = 4096):
    """Memory-efficient multi-head dot product attention."""
    # Implementation adapted from Self-Attention Does Not Need O(n^2) Memory
    # Rabe and Staats, https://arxiv.org/pdf/2112.05682.pdf
    if self._position_encoding_type == 'absolute':
      pass
    elif self._position_encoding_type == 'rotary':
      if q_positions is None or kv_positions is None:
        raise ValueError(
            'Both q_positions and kv_positions must be specified '
            'for rotary attention.')
      q, k = self._rotary_position_embeddings(q, k, q_positions, kv_positions)
    else:
      raise ValueError('Invalid position encoding type.')

    if mask is None:
      mask = jnp.ones([q.shape[0], 1, q.shape[1], k.shape[1]])

    # Split heads were required for rotary position encoding above.
    # Now combine the heads so that the shapes are less likely to require TPU
    # padding as we iterate through query and key chunks.
    q = jnp.reshape(q, [q.shape[0], q.shape[1], q.shape[2] * q.shape[3]])
    k = jnp.reshape(k, [k.shape[0], k.shape[1], k.shape[2] * k.shape[3]])
    v = jnp.reshape(v, [v.shape[0], v.shape[1], v.shape[2] * v.shape[3]])

    def attn(q, k, v, mask):
      num_q, q_features = q.shape

      def chunk_scanner(chunk_idx, _):
        query_chunk = jax.lax.dynamic_slice(
            q, (chunk_idx, 0),
            slice_sizes=(min(query_chunk_size, num_q), q_features))
        mask_chunk = jax.lax.dynamic_slice(
            mask, (0, chunk_idx, 0),
            slice_sizes=(mask.shape[0],
                         min(query_chunk_size, num_q),
                         k.shape[0]))
        return (chunk_idx + query_chunk_size,
                self._query_chunk_attention(
                    query_chunk, k, v, mask_chunk,
                    key_chunk_size=key_chunk_size, precision=precision,
                    is_training=is_training))

      _, res = hk.scan(
          chunk_scanner,
          init=0,
          xs=None,
          length=math.ceil(num_q / query_chunk_size))
      return res.reshape(num_q, v.shape[-1])
    return hk.vmap(attn, split_rng=(not hk.running_init()))(q, k, v, mask)

  def __call__(self,
               inputs_q,
               inputs_kv,
               is_cross_attend: bool,
               is_training: bool,
               memory_type: str,
               mask: Optional[jnp.ndarray] = None,
               memory: Optional[AttentionState] = None,
               q_positions: Optional[jnp.ndarray] = None,
               kv_positions: Optional[jnp.ndarray] = None,
               head_group_size: int = 0,
               use_chunked_attention: bool = False,
               query_chunk_size: int = 1024,
               key_chunk_size: int = 4096):
    # Q and K must have the same number of channels.
    # Default to preserving Q's input's shape.
    if self._qk_channels is None:
      self._qk_channels = inputs_q.shape[-1]
    # V's num_channels determines the shape of the output of QKV-attention.
    # Default to the same number of channels used in the key-query operation.
    if self._v_channels is None:
      self._v_channels = self._qk_channels
    # Project the output of QKV attention to a desired number of channels.
    # Default to the same number as the output of the QKV attention operation.
    if self._output_channels is None:
      self._output_channels = self._v_channels

    assert self._qk_channels % self._num_heads == 0
    assert self._v_channels % self._num_heads == 0
    qk_channels_per_head = self._qk_channels // self._num_heads
    v_channels_per_head = self._v_channels // self._num_heads

    # -----------------------------
    # ------ Compute Q, K, V ------
    # -----------------------------
    # Project QKV to a common feature dimension.
    q = conv_1d(self._qk_channels, init_scale=self._init_scale)(inputs_q)
    k = conv_1d(self._qk_channels, init_scale=self._init_scale)(inputs_kv)
    v = conv_1d(self._v_channels, init_scale=self._init_scale)(inputs_kv)

    # Reshape channels for multi-head attention.
    batch, q_time, _ = q.shape
    _, kv_time, _ = k.shape

    q = jnp.reshape(q, [batch, q_time, self._num_heads, qk_channels_per_head])
    k = jnp.reshape(k, [batch, kv_time, self._num_heads, qk_channels_per_head])
    v = jnp.reshape(v, [batch, kv_time, self._num_heads, v_channels_per_head])

    # --------------------
    # ------ Memory ------
    # --------------------
    def _update_memory(mem_arr, new_vals):
      # Add new memories to the right, possibly overwriting the oldest memories.
      # e.g.
      # [0, 0, 0] <- [0, 0 mem_0] or
      # [mem_0, mem_1, mem_2] <- [mem_1, mem_2, mem_3] or
      # [mem_0, mem_1, mem_2] <- [mem_2, mem_3, mem_4]
      num_new_vals = new_vals.shape[1]
      assert num_new_vals <= mem_arr.shape[1]
      mem_arr = jnp.roll(mem_arr, axis=1, shift=-num_new_vals)
      return jax.lax.dynamic_update_slice_in_dim(
          mem_arr, new_vals, start_index=-num_new_vals, axis=1)

    # Grab additional attention targets from memory.
    if memory is None:
      memory_mask = None
    else:
      # We assume that masking is not required when caching, i.e. that the
      # current prediction is subsequent to all inputs.
      if memory_type == 'none':
        raise ValueError(
            'Memory input not expected when memory_type is `none`.')
      elif memory_type == 'kv':
        assert mask is None
        k = jnp.concatenate([memory.k, k], axis=1)
        v = jnp.concatenate([memory.v, v], axis=1)
        if kv_positions is not None:
          kv_positions = jnp.concatenate(
              [memory.kv_positions, kv_positions], axis=1)
        memory_mask = memory.memory_mask
        if memory_mask is not None:
          memory_mask = jnp.concatenate(
              [memory_mask,
               jnp.ones([batch, memory.k.shape[1]])], axis=-1)
      elif memory_type == 'fixed_size_kv':
        assert memory.memory_mask is not None
        k = _update_memory(memory.k, k)
        v = _update_memory(memory.v, v)
        if kv_positions is not None:
          kv_positions = _update_memory(memory.kv_positions, kv_positions)
        memory_mask = _update_memory(
            memory.memory_mask,
            jnp.ones([batch, kv_time], dtype=jnp.float32))
        attn_mask = flax.linen.make_attention_mask(
            query_input=jnp.ones([batch, q_time], dtype=jnp.float32),
            key_input=memory_mask)
        if mask is None:
          mask = attn_mask
        else:
          mask = flax.linen.combine_masks(attn_mask, mask)
      else:
        raise ValueError(f'Unknown memory_type: {memory_type}')

    # ------------------------------
    # ------ Attention -> MLP ------
    # ------------------------------
    if head_group_size:
      assert not use_chunked_attention
      # Attention maps are [n_heads, q_indices, kv_indices], so memory usage
      # grows linearly with the number of heads regardless of embedding sizes.
      # However, this is only a temporary allocation for the softmax.
      # This option computes heads in smaller groups, trading compute for
      # memory.
      # Note that finding the right option here usually requires some trial and
      # error because the combination of XLA optimizations and TPU padding means
      # it's not always clear what configuration will actually use less memory
      # in practice. In general, the largest group that works is best.
      per_head_results = []
      assert self._num_heads % head_group_size == 0
      for i in range(0, self._num_heads, head_group_size):
        per_head_result = self._attend(
            q[:, :, i:i + head_group_size],
            k[:, :, i:i + head_group_size],
            v[:, :, i:i + head_group_size],
            mask=mask,
            q_positions=q_positions,
            kv_positions=kv_positions,
            is_cross_attend=is_cross_attend,
            is_training=is_training)
        per_head_results.append(per_head_result)
      result = jnp.concatenate(per_head_results, axis=-1)
    else:
      if use_chunked_attention:
        result = self._chunked_attend(q, k, v,
                                      mask=mask,
                                      q_positions=q_positions,
                                      kv_positions=kv_positions,
                                      is_training=is_training,
                                      query_chunk_size=query_chunk_size,
                                      key_chunk_size=key_chunk_size)
      else:
        result = self._attend(q, k, v,
                              mask=mask,
                              q_positions=q_positions,
                              kv_positions=kv_positions,
                              is_cross_attend=is_cross_attend,
                              is_training=is_training)

    outputs = conv_1d(
        self._output_channels,
        with_bias=self._with_final_bias,
        init_scale=self._final_init_scale)(result)

    if memory_type == 'none':
      memory = None
    else:
      memory = AttentionState(
          k=k,
          v=v,
          kv_positions=kv_positions,
          memory_mask=memory_mask)

    return outputs, memory


def get_activation(activation_name):
  # TODO(drewjaegle): explore GeGLU and SwiGLU activations.
  # Narang et al 2021 (https://arxiv.org/abs/2102.11972)
  if activation_name == 'sq_relu':
    return lambda x: jax.nn.relu(x)**2
  else:
    return getattr(jax.nn, activation_name)


class Dense(hk.Module):
  """A Transformer-style dense module to follow attention."""

  def __init__(self,
               dropout_prob,
               activation_name,
               widening_factor=4,
               init_scale=1.,
               name=None):
    super(Dense, self).__init__(name=name)
    self._widening_factor = widening_factor
    self._dropout_prob = dropout_prob
    self._init_scale = init_scale
    self._activation_name = activation_name

  def __call__(self, x, is_training=True):
    dropout_prob = self._dropout_prob if is_training else 0.0
    output_channels = x.shape[-1]
    x = conv_1d(
        output_channels=self._widening_factor * output_channels,
        init_scale=self._init_scale)(x)
    x = get_activation(self._activation_name)(x)
    x = conv_1d(
        output_channels=output_channels,
        init_scale=self._init_scale)(x)
    return hk.dropout(hk.next_rng_key(), dropout_prob, x)


class SelfAttention(hk.Module):
  """A self-attention module, including a dense block."""

  def __init__(self,
               dropout_prob,
               position_encoding_type,
               fraction_to_rotate,
               max_wavelength,
               widening_factor=4,
               dropout_attn_prob=0.0,
               num_heads=8,
               att_init_scale=1.0,
               dense_init_scale=1.0,
               activation_name='gelu',
               fraction_heads_to_rotate=1.0,
               name=None):
    super(SelfAttention, self).__init__(name=name)
    self._widening_factor = widening_factor
    self._dropout_prob = dropout_prob
    self._dropout_attn_prob = dropout_attn_prob
    self._num_heads = num_heads
    self._att_init_scale = att_init_scale
    self._dense_init_scale = dense_init_scale
    self._activation_name = activation_name
    self._position_encoding_type = position_encoding_type
    self._fraction_to_rotate = fraction_to_rotate
    self._fraction_heads_to_rotate = fraction_heads_to_rotate
    self._max_wavelength = max_wavelength

  def __call__(self,
               inputs,
               memory_type: str,
               mask: Optional[jnp.ndarray] = None,
               memory: Optional[AttentionState] = None,
               is_training=True,
               q_positions=None,
               kv_positions=None):
    dropout_prob = self._dropout_prob if is_training else 0.0
    dropout_attn_prob = self._dropout_attn_prob if is_training else 0.0

    x = inputs
    qkv_inputs = layer_norm(inputs)
    attention, attention_state = Attention(
        num_heads=self._num_heads,
        init_scale=self._att_init_scale,
        dropout_prob=dropout_attn_prob,
        position_encoding_type=self._position_encoding_type,
        fraction_to_rotate=self._fraction_to_rotate,
        fraction_heads_to_rotate=self._fraction_heads_to_rotate,
        max_wavelength=self._max_wavelength)(
            qkv_inputs,
            qkv_inputs,
            is_cross_attend=False,
            is_training=is_training,
            mask=mask,
            memory_type=memory_type,
            memory=memory,
            q_positions=q_positions,
            kv_positions=kv_positions)
    attention = hk.dropout(hk.next_rng_key(), dropout_prob, attention)
    x += attention

    x += Dense(
        widening_factor=self._widening_factor,
        dropout_prob=dropout_prob,
        init_scale=self._dense_init_scale,
        activation_name=self._activation_name)(
            layer_norm(x), is_training=is_training)
    return x, attention_state


class CrossAttention(hk.Module):
  """A cross-attention module, including a dense block."""

  def __init__(self,
               dropout_prob,
               position_encoding_type,
               fraction_to_rotate,
               max_wavelength,
               head_group_size: Optional[int],
               use_chunked_attention: bool,
               query_chunk_size: int = 1024,
               key_chunk_size: int = 4096,
               widening_factor=1,
               dropout_attn_prob=0.0,
               num_heads=8,
               att_init_scale=1.0,
               dense_init_scale=1.0,
               shape_for_attn='kv',
               use_query_residual=False,
               activation_name='gelu',
               fraction_heads_to_rotate=1.0,
               name=None):
    super(CrossAttention, self).__init__(name=name)
    self._widening_factor = widening_factor
    self._dropout_prob = dropout_prob
    self._dropout_attn_prob = dropout_attn_prob
    self._num_heads = num_heads
    self._att_init_scale = att_init_scale
    self._dense_init_scale = dense_init_scale
    self._shape_for_attn = shape_for_attn
    self._use_query_residual = use_query_residual
    self._activation_name = activation_name
    self._position_encoding_type = position_encoding_type
    self._fraction_to_rotate = fraction_to_rotate
    self._fraction_heads_to_rotate = fraction_heads_to_rotate
    self._max_wavelength = max_wavelength
    self._head_group_size = head_group_size
    self._use_chunked_attention = use_chunked_attention
    self._query_chunk_size = query_chunk_size
    self._key_chunk_size = key_chunk_size

  def __call__(self,
               inputs_q,
               inputs_kv,
               mask,
               memory_type: str,
               memory=None,
               is_training=True,
               q_positions=None,
               kv_positions=None):
    dropout_prob = self._dropout_prob if is_training else 0.0
    dropout_attn_prob = self._dropout_attn_prob if is_training else 0.0

    output_channels = inputs_q.shape[-1]
    if self._shape_for_attn == 'q':
      qk_channels = inputs_q.shape[-1]
    elif self._shape_for_attn == 'kv':
      qk_channels = inputs_kv.shape[-1]
    else:
      raise ValueError(f'Unknown value {self._shape_for_attn} for '
                       'shape_for_attention.')

    attention, attention_state = Attention(
        num_heads=self._num_heads,
        init_scale=self._att_init_scale,
        dropout_prob=dropout_attn_prob,
        qk_channels=qk_channels,
        output_channels=output_channels,
        position_encoding_type=self._position_encoding_type,
        fraction_to_rotate=self._fraction_to_rotate,
        fraction_heads_to_rotate=self._fraction_heads_to_rotate,
        max_wavelength=self._max_wavelength)(
            layer_norm(inputs_q),
            layer_norm(inputs_kv),
            is_cross_attend=True,
            is_training=is_training,
            mask=mask,
            memory_type=memory_type,
            memory=memory,
            q_positions=q_positions,
            kv_positions=kv_positions,
            head_group_size=self._head_group_size,
            use_chunked_attention=self._use_chunked_attention,
            query_chunk_size=self._query_chunk_size,
            key_chunk_size=self._key_chunk_size)
    attention = hk.dropout(hk.next_rng_key(), dropout_prob, attention)

    # Optionally include a residual to the query.
    # Consider omitting the residual if the semantics of query and output
    # are different, e.g. if queries are positions and outputs are pixels.
    if self._use_query_residual:
      x = inputs_q + attention
    else:
      x = attention

    x += Dense(
        widening_factor=self._widening_factor,
        dropout_prob=dropout_prob,
        init_scale=self._dense_init_scale,
        activation_name=self._activation_name)(
            layer_norm(x), is_training=is_training)
    return x, attention_state


#  -----------------------------------------------------------
#  -----------------------  Perceiver  -----------------------
#  -----------------------------------------------------------


class PerceiverAR(hk.Module):
  """Perceiver AR."""

  def __init__(
      self,
      num_classes,
      input_idx_size,
      max_context_length,
      position_encoding_type,
      input_embed_dim=1024,
      position_encoding='sinusoidal',
      learnable_position_embeddings=False,
      num_transformers_per_block=12,
      z_index_dim=1024,
      num_z_channels=1024,
      num_cross_attend_heads=1,
      num_transformer_heads=8,
      cross_attend_widening_factor=1,
      transformer_widening_factor=1,
      initial_query_offsetting=False,
      initial_query_embed_style='none',
      share_learned_initial_query=False,
      initial_query_embed_activation_name='sq_relu',
      initial_query_embed_num_layers=2,
      include_initial_cross_attention=True,
      additional_cross_attend_after_layers=None,
      dropout_prob=0.0,
      z_pos_enc_init_scale=0.02,
      concat_pos=True,
      cross_attention_shape_for_attn='kv',
      encoder_use_query_residual=True,
      mask_style='final_block',
      latent_dropout_prob=0.0,
      encoder_dropout_prob=0.0,
      num_latents_per_position=1,
      activation_name='sq_relu',
      fraction_to_rotate=0.25,
      fraction_heads_to_rotate=1.0,
      max_wavelength=8192,
      cross_attention_head_group_size=0,
      chunked_cross_attention=False,
      chunked_cross_query_size=1024,
      chunked_cross_key_size=4096,
      use_positions_from_data=False,
      train_input_positions=0,
      final_layer_init_zeros=True,
      use_negative_rotary_positions=False,
      name='perceiver_ar'):
    super().__init__(name=name)
    # Feature and task parameters:
    self._learnable_position_embeddings = learnable_position_embeddings
    self._position_encoding_type = position_encoding_type

    # For backward compatibility with previous position generating method.
    self._use_negative_rotary_positions = use_negative_rotary_positions

    self._use_positions_from_data = use_positions_from_data
    self._train_input_positions = train_input_positions
    if self._train_input_positions:
      assert self._train_input_positions < max_context_length

    self._initial_query_offsetting = initial_query_offsetting
    if initial_query_offsetting:
      # The initial cross-attention query positions get increased by 1, so we
      # also increase max_wavelength, such that the indices do not clash in the
      # rotary encoding step. See __call__ for more details.
      max_wavelength += 1

    self._initial_query_embed_style = initial_query_embed_style
    if initial_query_embed_style == 'none':
      pass
    elif initial_query_embed_style == 'mlp':
      self.q_embed_z_linear = mlp(
          num_hiddens=num_z_channels,
          num_layers=initial_query_embed_num_layers,
          activation_fn=get_activation(initial_query_embed_activation_name))
    elif initial_query_embed_style == 'learned':
      self._share_learned_initial_query = share_learned_initial_query
      first_dim = 1 if share_learned_initial_query else z_index_dim
      # If the query vector is shared across query all positions, repeat it in
      # the __call__ method.
      self.q_embed_learned = hk.get_parameter(
          name='q_embed_learned',
          shape=[first_dim, num_z_channels],
          init=hk.initializers.TruncatedNormal(stddev=0.02))
    else:
      raise ValueError(
          f'Unknown initial_query_embed_style: {initial_query_embed_style}')

    if (self._position_encoding_type == 'rotary' and
        max_wavelength < max_context_length):
      raise ValueError(
          'max_wavelength should not be smaller than max_context_length!'
          f'max_wavelength set to {max_wavelength} for a '
          f'max_context_length of {max_context_length}.')

    self._num_latents_per_position = num_latents_per_position

    self._num_classes = num_classes
    self._concat_pos = concat_pos

    # Architecture parameters:
    self._input_embed_dim = input_embed_dim
    self._z_index_dim = z_index_dim

    self._mask_style = mask_style

    self._latent_dropout_prob = latent_dropout_prob
    self._encoder_dropout_prob = encoder_dropout_prob

    # Check that we can use multihead-attention with these shapes.
    assert num_z_channels % num_transformer_heads == 0
    assert num_z_channels % num_cross_attend_heads == 0

    # Set up architecture construction:
    if additional_cross_attend_after_layers is None:
      additional_cross_attend_after_layers = []

    if additional_cross_attend_after_layers:
      assert len(additional_cross_attend_after_layers) == len(
          set(additional_cross_attend_after_layers))
      assert min(additional_cross_attend_after_layers) >= 0
      assert max(
          additional_cross_attend_after_layers) < num_transformers_per_block
    self._additional_cross_attend_after_layers = sorted(
        additional_cross_attend_after_layers)

    self.input_embed = hk.Embed(
        vocab_size=num_classes, embed_dim=self._input_embed_dim)
    if self._position_encoding_type == 'absolute':
      self.position_embeddings = []
      for idx_size in input_idx_size:
        if idx_size == -1:
          if self._use_positions_from_data:
            raise ValueError(
                f'Cannot use unknown input_idx_size ({input_idx_size}) with '
                f'use_positions_from_data because the required embedding size '
                f'cannot be determined.')
          idx_size = max_context_length

        # Initialize the sequence's position encoding:
        if position_encoding == 'sinusoidal':
          position_encodings = generate_sinusoidal_features(
              size=self._input_embed_dim, max_len=idx_size)
        elif position_encoding == 'linear':
          # For debugging only.
          position_encodings = generate_linear_features(
              size=self._input_embed_dim, max_len=idx_size)
        elif position_encoding == 'random':
          position_encodings = None  # Use hk.Embed initialization.
        elif position_encoding == 'fourier':
          position_encodings = generate_fourier_features(
              pos=build_linear_positions([idx_size]),
              n_bands=self._input_embed_dim // 2,
              max_res=idx_size,
              concat_pos=False)
        else:
          raise ValueError(f'Unknown position_encoding: {position_encoding}')

        self.position_embeddings.append(hk.Embed(
            vocab_size=idx_size,
            embed_dim=self._input_embed_dim,
            embedding_matrix=position_encodings))
    elif self._position_encoding_type == 'rotary':
      # No need to pre-generate position embeddings for rotary encoding.
      pass
    else:
      raise ValueError(
          f'Unknown position_encoding_type: {self._position_encoding_type}')

    # Build the encoder:
    self._include_initial_cross_attention = include_initial_cross_attention
    if self._include_initial_cross_attention:
      self.initial_cross_attn = CrossAttention(
          dropout_prob=dropout_prob,
          position_encoding_type=self._position_encoding_type,
          fraction_to_rotate=fraction_to_rotate,
          fraction_heads_to_rotate=fraction_heads_to_rotate,
          max_wavelength=max_wavelength,
          widening_factor=cross_attend_widening_factor,
          num_heads=num_cross_attend_heads,
          shape_for_attn=cross_attention_shape_for_attn,
          use_query_residual=encoder_use_query_residual,
          activation_name=activation_name,
          head_group_size=cross_attention_head_group_size,
          use_chunked_attention=chunked_cross_attention,
          query_chunk_size=chunked_cross_query_size,
          key_chunk_size=chunked_cross_key_size)
    else:
      if self._mask_style != 'final_block':
        raise ValueError(
            'If omitting the initial cross-attention, mask_style must be '
            f'final_block. Instead, got {self._mask_style}.')

    # Initialize the latent array.
    self.z_linear = conv_1d(num_z_channels)

    if self._num_latents_per_position > 1:
      self.z_pos_enc = TrainablePositionEncoding(
          index_dim=self._num_latents_per_position,
          num_channels=num_z_channels,
          init_scale=z_pos_enc_init_scale, name='z_pos_enc')

    # Build the processor:
    self.latent_transformer = []  # type: List[Any]
    for i in range(num_transformers_per_block):
      self.latent_transformer.append(SelfAttention(
          dropout_prob=dropout_prob,
          position_encoding_type=self._position_encoding_type,
          fraction_to_rotate=fraction_to_rotate,
          fraction_heads_to_rotate=fraction_heads_to_rotate,
          max_wavelength=max_wavelength,
          widening_factor=transformer_widening_factor,
          num_heads=num_transformer_heads,
          activation_name=activation_name))
      if i in self._additional_cross_attend_after_layers:
        self.latent_transformer.append(CrossAttention(
            dropout_prob=dropout_prob,
            position_encoding_type=self._position_encoding_type,
            fraction_to_rotate=fraction_to_rotate,
            fraction_heads_to_rotate=fraction_heads_to_rotate,
            max_wavelength=max_wavelength,
            widening_factor=cross_attend_widening_factor,
            num_heads=num_cross_attend_heads,
            shape_for_attn=cross_attention_shape_for_attn,
            use_query_residual=encoder_use_query_residual,
            activation_name=activation_name,
            head_group_size=cross_attention_head_group_size,
            use_chunked_attention=chunked_cross_attention,
            query_chunk_size=chunked_cross_query_size,
            key_chunk_size=chunked_cross_key_size))

    if final_layer_init_zeros:
      w_init = jnp.zeros
    else:
      w_init = hk.initializers.VarianceScaling(1.0)
    self.final_layer = hk.Linear(
        self._num_classes, w_init=w_init, name='logits')

  def _build_position_encodings(self, inputs, input_idxs):
    """Construct position encodings."""
    if input_idxs is None:
      input_idxs = jnp.expand_dims(self._get_positions(inputs), axis=-1)

    pe = sum(self.position_embeddings[i](input_idxs[..., i])
             for i in range(input_idxs.shape[-1]))
    if not self._learnable_position_embeddings:
      pe = jax.lax.stop_gradient(pe)

    return pe

  def _get_positions(self, inputs):
    """Construct position encodings."""
    batch_size = inputs.shape[0]
    length = inputs.shape[1]

    pos = jnp.arange(length)
    pos = jnp.broadcast_to(pos, (batch_size,) + pos.shape)

    pos = make_positions_terminal_relative(pos, inputs)

    return pos

  def _build_network_inputs(self, inputs, input_idxs):
    """Construct the final input, including position encoding."""
    assert inputs.ndim == 2  # (batch, length)

    # Build position encodings. Do this before right shifting inputs
    # because that adds a padding token to the beginning of the sequence, which
    # would confuse sequence length calculations.
    if self._position_encoding_type == 'rotary':
      if input_idxs is not None:
        # TODO(fjord): support multi-dimensional positions for rotary encoding.
        if input_idxs.shape[2] != 1:
          raise ValueError(
              f'input_idxs must have one-dimensional event indices for rotary '
              f'encoding, but got {input_idxs.shape}')
        positions = input_idxs[:, :, 0]
      else:
        positions = self._get_positions(inputs)
        # For backward compatibility with previous position generating method.
        if self._use_negative_rotary_positions:
          positions = -positions
      position_encodings = None
    else:
      position_encodings = self._build_position_encodings(inputs, input_idxs)
      positions = None

    embedded_inputs = self.input_embed(inputs)

    if self._position_encoding_type == 'absolute':
      embedded_inputs += position_encodings

    return embedded_inputs, positions

  def _select_random_input_positions(
      self, inputs, embedded_inputs, encoder_mask, positions):
    def sel(inputs, embedded_inputs, encoder_mask, positions):
      pos = jnp.arange(inputs.shape[0])
      # TODO(drewjaegle): Switch to faster one-hot indexing.
      pos = jax.random.permutation(hk.next_rng_key(), pos)
      pos = pos[:self._train_input_positions]

      # Ensure that positions resulting in padding events are sorted at the end.
      pos = pos[jnp.argsort(inputs[pos] == 0, kind='stable')]

      inputs = inputs[pos]
      embedded_inputs = embedded_inputs[pos]
      encoder_mask = encoder_mask[:, :, pos]
      if positions is not None:
        positions = positions[pos]

      return inputs, embedded_inputs, encoder_mask, positions
    return hk.vmap(sel, split_rng=(not hk.running_init()))(
        inputs, embedded_inputs, encoder_mask, positions)

  def __call__(self, inputs, input_idxs, is_training,
               memory_type='none', memory=None, z_index_dim=None,
               use_remat=False):
    batch_size = inputs.shape[0]

    # Optionally override model's (non-parametric) z_index_dim
    if z_index_dim is None:
      z_index_dim = self._z_index_dim

    if self._use_positions_from_data:
      assert input_idxs is not None
    else:
      if memory is not None:
        raise ValueError('use_positions_from_data must be used for caching.')
      input_idxs = None

    if z_index_dim == 1:
      # No masking needed: all memories influence the current output.
      masks = Masks(encoder=None, processor=None)

      latent_last_steps = jnp.full([batch_size, 1], inputs.shape[1] - 1)
    else:
      masks, latent_last_steps = make_block_causal_masks(
          inputs,
          latent_index_dim=z_index_dim,
          latents_per_position=self._num_latents_per_position,
          batch_size=batch_size,
          mask_style=self._mask_style,
          rng_key=hk.next_rng_key(),
          latent_dropout_prob=self._latent_dropout_prob,
          is_training=is_training)
      assert inputs.shape[1] >= z_index_dim

    embedded_inputs, positions = self._build_network_inputs(inputs, input_idxs)

    @jax.vmap
    def index_per_latent(arr, indices):
      return jnp.take(arr, indices, axis=0)

    # Grab the initial state and position encoding used for the latents.
    latent_positions = None
    if self._position_encoding_type == 'absolute':
      # Position encodings not factored separately from the content/state.
      pass
    # Index the positions or position encodings as well as the inputs.
    # Note that ignored latents have a last step of -1. This will lead to them
    # taking the final (-1th) position encoding, but these latents are masked
    # in the loss.
    elif self._position_encoding_type == 'rotary':
      latent_positions = index_per_latent(positions, latent_last_steps)
    else:
      raise ValueError(
          f'Unknown position encoding: {self._position_encoding_type}')

    last_step_embeddings = index_per_latent(embedded_inputs, latent_last_steps)
    if self._initial_query_embed_style == 'learned':
      # Use the learned representation as query vector for all inputs.
      if not self._share_learned_initial_query and memory is not None:
        raise ValueError(
            'Learned, unshared query embeddings not yet supported for caching.')

      repeat_2nd_dim = (z_index_dim if self._share_learned_initial_query else 1)
      initial_q_input = jnp.tile(self.q_embed_learned,
                                 reps=(inputs.shape[0], repeat_2nd_dim, 1))
    else:
      # Use the encodings from the last inputs as query vectors.
      initial_q_input = self.z_linear(last_step_embeddings)

    if self._num_latents_per_position > 1:
      if memory is not None:
        raise ValueError(
            'Caching not yet supported for num_latents_per_position > 1.')
      # Along axis 1, latents are arranged as:
      # [pos_0_lat_0, pos_0_lat_1, ..., pos_0_lat_K, pos_1_lat_0, ...]
      # Position encodings are kept unreplicated to avoid unneeded computation
      # in relative position attention.

      # Replicate content encodings:
      initial_q_input = jnp.repeat(
          initial_q_input, self._num_latents_per_position, axis=1)

      # Build the latent index encodings.
      z_pos_enc = self.z_pos_enc(batch_size=batch_size)
      num_unique_positions = z_index_dim // self._num_latents_per_position
      z_pos_enc = jnp.tile(z_pos_enc, [1, num_unique_positions, 1])

      initial_q_input += z_pos_enc

    # Additional processing of the initial cross-attention query.
    if self._initial_query_embed_style == 'mlp':
      # Embed the cross-attention query.
      initial_q_input = self.q_embed_z_linear(initial_q_input)
    if self._initial_query_offsetting:
      # Modify the query positions, such that each element is increased by 1.
      # The resulting positions represent the true indices of the sequence
      # elements that we wish to predict. In __init__, we also set
      # max_wavelength = max_context_length + 1, to avoid indices clashing in
      # the rotary encoding step.
      latent_positions = jnp.add(latent_positions,
                                 jnp.ones_like(latent_positions))

    perceiver_state = []

    if is_training:
      if memory is not None:
        raise ValueError('Caching not supported at train.')
      if self._train_input_positions:
        inputs, embedded_inputs, encoder_mask, positions = (
            self._select_random_input_positions(
                inputs, embedded_inputs, masks.encoder, positions))
        masks = Masks(encoder=encoder_mask, processor=masks.processor)

      # Encoder dropout in cross-attends
      keep_rate = 1.0 - self._encoder_dropout_prob
      encoder_dropout_keys = jax.random.bernoulli(
          hk.next_rng_key(),
          keep_rate,
          shape=[batch_size, embedded_inputs.shape[1]])
      encoder_dropout_mask = flax.linen.make_attention_mask(
          query_input=jnp.ones([batch_size, z_index_dim],
                               dtype=embedded_inputs.dtype),
          key_input=encoder_dropout_keys)
      encoder_mask = flax.linen.combine_masks(
          masks.encoder, encoder_dropout_mask)
      masks = Masks(encoder=encoder_mask, processor=masks.processor)

    memory_idx = 0

    # This is the main Perceiver cross attend with the raw inputs.
    if self._include_initial_cross_attention:
      if memory is None:
        attn_memory = None
      else:
        attn_memory = memory[memory_idx]
        memory_idx += 1
      z, initial_cross_attend_state = self.initial_cross_attn(
          initial_q_input, embedded_inputs,
          mask=masks.encoder,
          memory_type=memory_type,
          memory=attn_memory,
          is_training=is_training,
          # Input position encodings belong to the keys/values, so here we use
          # the inputs' position encodings rather than the latents'.
          q_positions=latent_positions,
          kv_positions=positions)
      perceiver_state.append(initial_cross_attend_state)
    else:
      z = initial_q_input

    # This is the main stack of self-attention blocks, including optional
    # additional cross attends to the raw inputs.
    for attn in self.latent_transformer:
      if memory is None:
        attn_memory = None
      else:
        attn_memory = memory[memory_idx]
        memory_idx += 1

      if attn.name.startswith('cross_attention'):
        def call_attn():
          return attn(  # pylint: disable=cell-var-from-loop
              z,
              embedded_inputs,
              memory_type=memory_type,
              memory=attn_memory,
              is_training=is_training,
              mask=masks.encoder,
              q_positions=latent_positions,
              kv_positions=positions)
      elif attn.name.startswith('self_attention'):
        def call_attn():
          return attn(  # pylint: disable=cell-var-from-loop
              z,
              memory_type=memory_type,
              memory=attn_memory,
              is_training=is_training,
              mask=masks.processor,
              q_positions=latent_positions,
              kv_positions=latent_positions)
      else:
        raise ValueError(f'Unknown attention type for layer: {attn}')

      if use_remat:
        z, attention_state = hk.remat(call_attn)()
      else:
        z, attention_state = call_attn()
      perceiver_state.append(attention_state)

    # Project to logits shape.
    z = layer_norm(z)
    if self._num_latents_per_position > 1:
      # Average over latents with the same position.
      z = hk.AvgPool(
          window_shape=(self._num_latents_per_position, 1),
          strides=self._num_latents_per_position,
          padding='VALID')(z)
    input_events_logits = self.final_layer(z)

    output = Output(
        input_events_logits=input_events_logits,
        encoder_mask=masks.encoder,
        processor_mask=masks.processor,
        latent_last_steps=latent_last_steps,
        perceiver_state=perceiver_state,
    )

    return output
