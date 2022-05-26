"""Tests for perceiver_ar."""

from absl.testing import absltest
from absl.testing import parameterized
import dataset
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import perceiver_ar
import sample_utils


def generate_inputs(input_index_dim, num_invalid_inputs, batch_size=1):
  invalid_inputs = jnp.zeros(
      [batch_size, num_invalid_inputs], dtype=jnp.float32)
  valid_inputs = jnp.ones(
      [batch_size, input_index_dim - num_invalid_inputs], dtype=jnp.float32)
  inputs = jnp.concatenate((valid_inputs, invalid_inputs), axis=-1)
  return inputs


class PerceiverARTest(parameterized.TestCase):

  def test_get_sequence_length(self):
    self.assertEqual(
        3, perceiver_ar.get_sequence_length(jnp.array([1, 2, 3, 0])))
    self.assertEqual(
        4, perceiver_ar.get_sequence_length(jnp.array([1, 2, 3, 4])))
    self.assertEqual(
        0, perceiver_ar.get_sequence_length(jnp.array([0, 0, 0, 0])))

  def test_truncate_sequence(self):
    np.testing.assert_array_equal(
        perceiver_ar.truncate_sequence(jnp.array([1, 2, 3, 4])),
        [1, 2, 3, 0])
    np.testing.assert_array_equal(
        perceiver_ar.truncate_sequence(jnp.array([1, 2, 3, 0])),
        [1, 2, 0, 0])
    np.testing.assert_array_equal(
        perceiver_ar.truncate_sequence(jnp.array([1, 0, 0, 0])),
        [0, 0, 0, 0])
    np.testing.assert_array_equal(
        perceiver_ar.truncate_sequence(jnp.array([0, 0, 0, 0])),
        [0, 0, 0, 0])

  def test_make_positions_terminal_relative(self):
    """Tests realigning positions relative to the last input."""
    input_seq = [
        [1., 1., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 0., 0.],
        [1., 1., 1., 1.]
    ]

    pos_seq = [
        [0., 1., 2., 3.],
        [0., 1., 2., 3.],
        [0., 1., 2., 3.],
        [0., 1., 2., 3.],
    ]

    expected_output = [
        # Outputs to the right of 0 will generally be ignored.
        [1., 0., 3., 2.],
        [0., 3., 2., 1.],
        [3., 2., 1., 0.],
        [3., 2., 1., 0.],
    ]
    input_seq = np.array(input_seq)
    pos_seq = np.array(pos_seq)
    expected_output = np.array(expected_output)

    output = perceiver_ar.make_positions_terminal_relative(pos_seq, input_seq)
    np.testing.assert_array_equal(output, expected_output)

  # ----------------------------------
  # ----        Test masks        ----
  # ----------------------------------

  COMMON_BLOCK_CAUSAL_MASK_KWARGS = {
      'is_training': True,
      'latents_per_position': 1,
      'latent_dropout_prob': 0.0,
  }

  def test_full_sequence(self):
    batch_size = 1
    latent_index_dim = 6
    rng_key = jax.random.PRNGKey(seed=42)

    inputs = generate_inputs(input_index_dim=10, num_invalid_inputs=0)

    masks, latent_last_steps = perceiver_ar.make_block_causal_masks(
        inputs=inputs,
        latent_index_dim=latent_index_dim,
        batch_size=batch_size,
        mask_style='final_block',
        rng_key=rng_key,
        **PerceiverARTest.COMMON_BLOCK_CAUSAL_MASK_KWARGS)

    # The final 6 positions are assigned to the latents.
    expected_encoder = [
        [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
    expected_processor = [
        [1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]]
    expected_last_steps = [4, 5, 6, 7, 8, 9]

    np.testing.assert_array_equal(masks.encoder[0, 0], expected_encoder)
    np.testing.assert_array_equal(masks.processor[0, 0], expected_processor)
    np.testing.assert_array_equal(latent_last_steps[0], expected_last_steps)

  def test_initial_sequence(self):
    """Tests the scenario where there's only an <SOS> token at the beginning."""
    batch_size = 1
    latent_index_dim = 6
    rng_key = jax.random.PRNGKey(seed=42)

    inputs = generate_inputs(input_index_dim=10, num_invalid_inputs=9)

    masks, latent_last_steps = perceiver_ar.make_block_causal_masks(
        inputs=inputs,
        latent_index_dim=latent_index_dim,
        batch_size=batch_size,
        mask_style='final_block',
        rng_key=rng_key,
        **PerceiverARTest.COMMON_BLOCK_CAUSAL_MASK_KWARGS)

    expected_encoder = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
    expected_processor = [
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.]]
    expected_last_steps = [-1, -1, -1, -1, -1, 0]

    np.testing.assert_array_equal(masks.encoder[0, 0], expected_encoder)
    np.testing.assert_array_equal(masks.processor[0, 0], expected_processor)
    np.testing.assert_array_equal(latent_last_steps[0], expected_last_steps)

  def test_partial_sequence_partial_latents(self):
    """Tests a partial sequence without enough positions to fill the latents."""
    batch_size = 1
    latent_index_dim = 6
    rng_key = jax.random.PRNGKey(seed=42)

    inputs = generate_inputs(input_index_dim=10, num_invalid_inputs=6)

    masks, latent_last_steps = perceiver_ar.make_block_causal_masks(
        inputs=inputs,
        latent_index_dim=latent_index_dim,
        batch_size=batch_size,
        mask_style='final_block',
        rng_key=rng_key,
        **PerceiverARTest.COMMON_BLOCK_CAUSAL_MASK_KWARGS)

    expected_encoder = [
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]]
    expected_processor = [
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 1., 1., 0., 0.],
        [0., 0., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 1.]]
    expected_last_steps = [-1, -1, 0, 1, 2, 3]

    np.testing.assert_array_equal(masks.encoder[0, 0], expected_encoder)
    np.testing.assert_array_equal(masks.processor[0, 0], expected_processor)
    np.testing.assert_array_equal(latent_last_steps[0], expected_last_steps)

  def test_partial_sequence_latents_full(self):
    """Tests a partial sequence with enough positions to fill the latents."""
    batch_size = 1
    latent_index_dim = 6
    rng_key = jax.random.PRNGKey(seed=42)

    inputs = generate_inputs(input_index_dim=10, num_invalid_inputs=3)

    masks, latent_last_steps = perceiver_ar.make_block_causal_masks(
        inputs=inputs,
        latent_index_dim=latent_index_dim,
        batch_size=batch_size,
        mask_style='final_block',
        rng_key=rng_key,
        **PerceiverARTest.COMMON_BLOCK_CAUSAL_MASK_KWARGS)

    expected_encoder = [
        [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0., 0., 0.]]
    expected_last_steps = [1, 2, 3, 4, 5, 6]

    expected_processor = [
        [1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]]

    np.testing.assert_array_equal(masks.encoder[0, 0], expected_encoder)
    np.testing.assert_array_equal(masks.processor[0, 0], expected_processor)
    np.testing.assert_array_equal(latent_last_steps[0], expected_last_steps)

  @parameterized.product(
      input_index_dim=(6, 10),
      num_invalid_inputs=(0, 3, 5),
  )
  def test_last_latent_predicts_last_valid_step(
      self, input_index_dim, num_invalid_inputs):
    """Eval requires the last latent predict the last valid input. Test this."""
    batch_size = 1
    latent_index_dim = 6
    rng_key = jax.random.PRNGKey(seed=42)

    last_valid_step = input_index_dim - num_invalid_inputs - 1
    inputs = generate_inputs(
        input_index_dim=input_index_dim,
        num_invalid_inputs=num_invalid_inputs)

    _, latent_last_steps = perceiver_ar.make_block_causal_masks(
        inputs=inputs,
        latent_index_dim=latent_index_dim,
        batch_size=batch_size,
        mask_style='final_block',
        rng_key=rng_key,
        **PerceiverARTest.COMMON_BLOCK_CAUSAL_MASK_KWARGS)
    last_latent_predicts = np.asarray(latent_last_steps[0][-1])

    np.testing.assert_equal(last_latent_predicts, last_valid_step)

  @parameterized.product(
      position_encoding_type=('rotary', 'absolute'),
      query_chunk_size=(32, 16, 8),
      key_chunk_size=(64, 32, 16),
      use_mask=(True, False),
  )
  def test_chunked_cross_attend(self,
                                position_encoding_type,
                                query_chunk_size,
                                key_chunk_size,
                                use_mask):
    """Verify chunked attention returns results close to regular attention."""

    def forward_fn(inputs_q, inputs_kv, mask, q_positions, kv_positions,
                   use_chunked, is_training):
      attn = perceiver_ar.Attention(
          dropout_prob=0.1,
          position_encoding_type=position_encoding_type,
          fraction_to_rotate=0.5,
          max_wavelength=1024,
          num_heads=8)
      return attn(
          inputs_q,
          inputs_kv,
          is_cross_attend=False,
          is_training=is_training,
          memory_type='none',
          memory=None,
          mask=mask,
          q_positions=q_positions,
          kv_positions=kv_positions,
          use_chunked_attention=use_chunked,
          query_chunk_size=query_chunk_size,
          key_chunk_size=key_chunk_size)

    forward = hk.transform(forward_fn)

    key = hk.PRNGSequence(0)

    batch = 2
    q_length = 32
    kv_length = 64
    emb_dim = 128
    inputs_q = jax.random.uniform(next(key), [batch, q_length, emb_dim])
    inputs_kv = jax.random.uniform(next(key), [batch, kv_length, emb_dim])

    if use_mask:
      mask = np.ones([batch, 1, q_length, kv_length])
      # Mask out the last 32 kv indices to verify that masking is working.
      mask[:, :, :, -32:] = 0.0
    else:
      mask = None

    q_positions = jnp.arange(q_length)
    q_positions = jnp.broadcast_to(q_positions, [batch, q_length])
    kv_positions = jnp.arange(kv_length)
    kv_positions = jnp.broadcast_to(kv_positions, [batch, kv_length])

    params = forward.init(
        next(key), inputs_q, inputs_kv, mask, q_positions, kv_positions,
        use_chunked=False, is_training=True)

    outputs, _ = forward.apply(
        params, next(key), inputs_q, inputs_kv, mask, q_positions, kv_positions,
        use_chunked=False, is_training=False)

    outputs_chunked, _ = forward.apply(
        params, next(key), inputs_q, inputs_kv, mask, q_positions, kv_positions,
        use_chunked=True, is_training=False)

    np.testing.assert_allclose(outputs, outputs_chunked, atol=1e-6)

    # Verify that outputs with dropout aren't the same as the ones without
    # dropout. Because the dropout with chunks happens with a different PRNG
    # chain, we wouldn't necessarily expect chunked and regular attention with
    # dropout to be the same.
    outputs_w_dropout, _ = forward.apply(
        params, next(key), inputs_q, inputs_kv, mask, q_positions, kv_positions,
        use_chunked=False, is_training=True)

    outputs_chunked_w_dropout, _ = forward.apply(
        params, next(key), inputs_q, inputs_kv, mask, q_positions, kv_positions,
        use_chunked=True, is_training=True)

    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(outputs, outputs_w_dropout,
                                 atol=1e-6)

    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(outputs_chunked, outputs_chunked_w_dropout,
                                 atol=1e-6)

  @parameterized.product(
      max_context_length=(2, 3, 5),
      num_self_attend_layers=(1, 2, 3),
      position_encoding_type=('rotary', 'absolute'),
      memory_type=('kv', 'fixed_size_kv')
  )
  def test_caching_context_matches_latents(
      self,
      max_context_length,
      num_self_attend_layers,
      position_encoding_type,
      memory_type):
    # All match so no longer-term dependencies are introduced by caching:
    z_index_dim = num_steps = max_context_length

    num_classes = 23
    batch_size = 2

    rng_key = hk.PRNGSequence(42)

    # Construct the input data
    inputs_all = jax.random.randint(
        next(rng_key),
        [batch_size, num_steps],
        minval=dataset.NUM_RESERVED_TOKENS,
        maxval=num_classes, dtype=jnp.int32)

    # Locations are sequential and increasing, but may start at random points.
    input_idxs_all = jnp.arange(num_steps)[None, :, None] + 1  # Add 1 for PAD
    max_start = 123
    input_idxs_all += jax.random.randint(
        next(rng_key),
        [batch_size],
        minval=0, maxval=max_start)[:, None, None]
    input_idx_size = [max_start + max_context_length + 2]

    # Initialize a model.
    num_cross_attend_heads = 2
    num_transformer_heads = 2
    num_z_channels = 32
    block_causal_perceiver_kwargs = dict(
        position_encoding_type=position_encoding_type,
        input_embed_dim=32,
        max_wavelength=64,
        use_positions_from_data=True,
        num_transformers_per_block=num_self_attend_layers,
        z_index_dim=z_index_dim,
        num_z_channels=32,
        num_cross_attend_heads=2,
        num_transformer_heads=2,
        include_initial_cross_attention=True,
        dropout_prob=0.0,
        latent_dropout_prob=0.0,
        encoder_dropout_prob=0.0,
        # Needed to compare outputs.
        final_layer_init_zeros=False)

    def forward_fn(inputs, input_idxs, memory=None, z_index_dim=None):
      model = perceiver_ar.PerceiverAR(
          num_classes,
          input_idx_size,
          max_context_length,
          **block_causal_perceiver_kwargs)
      return model(
          inputs,
          input_idxs,
          is_training=False,
          memory_type=memory_type,
          memory=memory,
          z_index_dim=z_index_dim)

    forward = hk.transform(forward_fn)

    # Initialize memory:
    if memory_type == 'kv':
      # one state for each attention layer, including the cross-attend.
      memory = [None for _ in range(num_self_attend_layers + 1)]
    elif memory_type == 'fixed_size_kv':
      memory = sample_utils.initialize_memory(
          batch_size,
          num_transformers_per_block=num_self_attend_layers,
          num_cross_attend_heads=num_cross_attend_heads,
          num_transformer_heads=num_transformer_heads,
          num_z_channels=num_z_channels,
          max_context_length_memory=max_context_length,
          z_index_dim_memory=z_index_dim,
          position_encoding_type=position_encoding_type,
          memory_type=memory_type)

    logits_cached_all = []
    logits_uncached_all = []

    # For inputs [3, 4, 5, 6],
    #
    # Uncached fed in as:
    #   [3, PAD, PAD, PAD],
    #   [3, 4, PAD, PAD],
    #   [3, 4, 5, PAD],
    #   [3, 4, 5, 6]
    #
    # Cached fed in as:
    #   [3],
    #   [4],
    #   [5],
    #   [6]
    def _crop_and_pad_inputs(inputs, input_idxs, step):
      end_idx = min(step + 1, num_steps)
      fill_size = max(0, num_steps - step - 1)
      inputs_padding = jnp.full(
          [batch_size, fill_size],
          fill_value=dataset.PAD_ID,
          dtype=inputs_all.dtype)
      inputs_padded = jnp.concatenate(
          [inputs[:, :end_idx], inputs_padding], axis=-1)

      input_idxs_padding = jnp.zeros(
          [batch_size, fill_size, 1], dtype=input_idxs_all.dtype)
      input_idxs_padded = jnp.concatenate(
          [input_idxs[:, :end_idx, :], input_idxs_padding],
          axis=1)

      return inputs_padded, input_idxs_padded

    for i in range(num_steps):
      inputs_uncached, input_idxs_uncached, = _crop_and_pad_inputs(
          inputs_all, input_idxs_all, i)
      inputs_cached = inputs_all[:, i][:, None]
      input_idxs_cached = input_idxs_all[:, i, :][:, None, :]

      if i == 0:
        params = forward.init(
            next(rng_key), inputs_uncached, input_idxs_uncached)

      outputs_uncached = forward.apply(
          params, next(rng_key), inputs_uncached, input_idxs_uncached)
      logits_uncached_all.append(outputs_uncached.input_events_logits[:, -1, :])

      outputs_cached = forward.apply(
          params, next(rng_key),
          inputs_cached, input_idxs_cached, memory=memory, z_index_dim=1)
      logits_cached_all.append(outputs_cached.input_events_logits[:, -1, :])
      memory = outputs_cached.perceiver_state

    np.testing.assert_allclose(
        logits_cached_all, logits_uncached_all, atol=1e-5)

  @parameterized.product(
      max_context_length=(7,),
      z_index_dim=(1, 2, 3, 7),
      num_self_attend_layers=(2, 3),
      position_encoding_type=('absolute', 'rotary'),
      memory_type=('kv', 'fixed_size_kv',)
  )
  def test_caching_context_exceeds_latents(
      self,
      max_context_length,
      z_index_dim,
      num_self_attend_layers,
      position_encoding_type,
      memory_type):
    num_classes = 23
    batch_size = 2

    rng_key = hk.PRNGSequence(42)

    # Construct the input data
    inputs_all = jax.random.randint(
        next(rng_key),
        [batch_size, max_context_length],
        minval=dataset.NUM_RESERVED_TOKENS,
        maxval=num_classes, dtype=jnp.int32)

    # Locations are sequential and increasing, but may start at random points.
    # Add 1 for PAD
    input_idxs_all = jnp.arange(max_context_length)[None, :, None] + 1
    max_start = 123
    input_idxs_all += jax.random.randint(
        next(rng_key),
        [batch_size],
        minval=0, maxval=max_start)[:, None, None]
    input_idx_size = [max_start + max_context_length + 2]

    # Initialize a model.
    num_cross_attend_heads = 2
    num_transformer_heads = 2
    num_z_channels = 32
    block_causal_perceiver_kwargs = dict(
        position_encoding_type=position_encoding_type,
        input_embed_dim=32,
        max_wavelength=64,
        use_positions_from_data=True,
        num_transformers_per_block=num_self_attend_layers,
        z_index_dim=z_index_dim,
        num_z_channels=32,
        num_cross_attend_heads=num_cross_attend_heads,
        num_transformer_heads=num_transformer_heads,
        include_initial_cross_attention=True,
        dropout_prob=0.0,
        latent_dropout_prob=0.0,
        encoder_dropout_prob=0.0,
        # Needed to compare outputs.
        final_layer_init_zeros=False)

    def forward_fn(inputs, input_idxs, memory=None, z_index_dim=None):
      model = perceiver_ar.PerceiverAR(
          num_classes,
          input_idx_size,
          max_context_length,
          **block_causal_perceiver_kwargs)
      return model(
          inputs, input_idxs, is_training=False,
          memory_type=memory_type, memory=memory, z_index_dim=z_index_dim)

    forward = hk.transform(forward_fn)

    params = forward.init(next(rng_key), inputs_all, input_idxs_all)

    # Step through a set of max_context_length inputs. Exceeding this with
    # caching will introduce additional long-term dependencies.
    #
    # For inputs [3, 4, 5, 6, 7] (max_context_length = 5) & z_index_dim = 3:
    #
    # Uncached fed in all at once:
    #   [3, 4, 5, 6, 7],
    #
    # Cached fed in in steps, once per z_index_dim:
    #   [3, 4, 5],
    #   [6],
    #   [7],
    outputs_uncached = forward.apply(
        params, next(rng_key), inputs_all, input_idxs_all)
    logits_uncached = outputs_uncached.input_events_logits

    # Initialize memory:
    if memory_type == 'kv':
      # one state for each attention layer, including the cross-attend.
      memory = [None for _ in range(num_self_attend_layers + 1)]
    elif memory_type == 'fixed_size_kv':
      memory = sample_utils.initialize_memory(
          batch_size,
          num_transformers_per_block=num_self_attend_layers,
          num_cross_attend_heads=num_cross_attend_heads,
          num_transformer_heads=num_transformer_heads,
          num_z_channels=num_z_channels,
          max_context_length_memory=max_context_length,
          z_index_dim_memory=z_index_dim,
          position_encoding_type=position_encoding_type,
          memory_type=memory_type)

    logits_cached = []
    start_idx = 0
    for i in range(z_index_dim):
      end_idx = max_context_length - z_index_dim + i + 1
      outputs_cached = forward.apply(
          params, next(rng_key),
          inputs_all[:, start_idx:end_idx],
          input_idxs_all[:, start_idx:end_idx, :],
          memory=memory,
          z_index_dim=1)
      memory = outputs_cached.perceiver_state
      logits_cached.append(outputs_cached.input_events_logits)
      start_idx = end_idx
    logits_cached = jnp.concatenate(logits_cached, axis=1)
    np.testing.assert_allclose(logits_cached, logits_uncached, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
