"""Utils for sampling and caching."""

import jax
import jax.numpy as jnp

import perceiver_ar


# -----------------------------
# ----- Caching utilities -----
# -----------------------------
def print_memory_shape(perceiver_state):
  for idx, memory in enumerate(perceiver_state):
    print(f'Memory shape, layer {idx}: {memory.k.shape[1]}')


def pad_axis(x, padding, axis):
  full_padding = [(0, 0)] * x.ndim
  full_padding[axis] = padding
  return jnp.pad(x, full_padding)


def initialize_memory(
    batch_size,
    num_transformers_per_block,
    num_cross_attend_heads,
    num_transformer_heads,
    num_z_channels,
    # These two can use different sizes than the model was trained with:
    max_context_length_memory,
    z_index_dim_memory,
    position_encoding_type,
    memory_type='fixed_size_kv'):
  """Initialize a memory for caching."""
  memory = []
  for i in range(num_transformers_per_block + 1):
    if i == 0:
      num_memories = max_context_length_memory
      num_heads = num_cross_attend_heads
      num_channels = num_z_channels // num_cross_attend_heads
    else:
      num_memories = z_index_dim_memory
      num_heads = num_transformer_heads
      num_channels = num_z_channels // num_transformer_heads
    k = jnp.zeros(
        [batch_size, num_memories, num_heads, num_channels], dtype=jnp.float32)
    v = jnp.zeros_like(k)

    if position_encoding_type == 'absolute':
      # Positions are factored in at the input, not at each layer.
      kv_positions = None
    elif position_encoding_type == 'rotary':
      kv_positions = jnp.zeros([batch_size, num_memories], dtype=jnp.int32)
    else:
      raise ValueError('Memory not supported for position encoding type'
                       f' {position_encoding_type}.')

    if memory_type == 'fixed_size_kv':
      memory_mask = jnp.zeros([batch_size, num_memories], dtype=jnp.float32)
    elif memory_type == 'kv':
      memory_mask = None
    else:
      raise ValueError('Unknown memory_type: {memory_type}')

    memory.append(perceiver_ar.AttentionState(
        k=k, v=v, kv_positions=kv_positions, memory_mask=memory_mask))

  return memory


# ----------------------------
# ---- Sampling utilities ----
# ----------------------------
def process_logits(i, logits, top_p=1., temperature=1., modality=None):
  """Process logits for nucleus and/or temperature sampling.

  Args:
    i: The index being sampled.
    logits: The input logits.
    top_p: The top_p parameter for nucleus sampling. Nucleus sampling is used
      if top_p < 1.
    temperature: The sampling temperature.
    modality: The data modality.
  Returns:
    The processed logits.
  """
  if modality == 'image':
    print('Adding channel masking for image inference.')
    channel = (i - 1) % 3
    logits = jax.lax.cond(
        channel == 0,
        lambda _: jnp.where(  # pylint: disable=g-long-lambda
            jnp.concatenate([jnp.ones([1, 3]),
                             jnp.ones([1, 256]),
                             jnp.zeros([1, 512])], axis=1),
            logits,
            -1e30),
        lambda _: logits,
        operand=None)
    logits = jax.lax.cond(
        channel == 1,
        lambda _: jnp.where(  # pylint: disable=g-long-lambda
            jnp.concatenate([
                jnp.ones([1, 3]),
                jnp.zeros([1, 256]),
                jnp.ones([1, 256]),
                jnp.zeros([1, 256])], axis=1),
            logits,
            -1e30),
        lambda _: logits,
        operand=None)
    logits = jax.lax.cond(
        channel == 2,
        lambda _: jnp.where(  # pylint: disable=g-long-lambda
            jnp.concatenate([jnp.ones([1, 3]),
                             jnp.zeros([1, 512]),
                             jnp.ones([1, 256])], axis=1),
            logits,
            -1e30),
        lambda _: logits,
        operand=None)
  elif modality == 'soundstream_12kbps':
    print('Adding channel masking for soundstream inference.')
    channel = (i - 1) % 24
    mask = jnp.zeros([24, 1024])
    mask = mask.at[channel, :].set(1.0)
    mask = jnp.concatenate(
        [jnp.ones([1, 3]), jnp.reshape(mask, newshape=(1, -1))], axis=1)
    logits = jnp.where(mask, logits, -1e30)
  elif modality == 'soundstream_22kbps':
    print('Adding channel masking for soundstream inference.')
    channel = (i - 1) % 44
    mask = jnp.zeros([44, 1024])
    mask = mask.at[channel, :].set(1.0)
    mask = jnp.concatenate(
        [jnp.ones([1, 3]), jnp.reshape(mask, newshape=(1, -1))], axis=1)
    logits = jnp.where(mask, logits, -1e30)
  else:
    pass

  if top_p < 1:
    print('Adding nucleus sampling.')
    sorted_logits = jax.lax.sort(logits, is_stable=False)
    sorted_probs = jax.nn.softmax(sorted_logits)
    threshold_idx = jnp.argmax(
        jnp.cumsum(sorted_probs, -1) >= 1 - top_p, axis=-1)
    threshold_largest_logits = jnp.take_along_axis(
        sorted_logits, threshold_idx[..., None], axis=-1)
    assert threshold_largest_logits.shape == logits.shape[:-1] + (1,)
    mask = logits >= threshold_largest_logits
    logits = jnp.where(mask, logits, -1e30)  # Set unused logits to -inf.

  logits /= jnp.maximum(temperature, 1e-12)
  return logits


def sample_position(
    i,
    events_rng_memory,
    forward_fn,
    condition_on,
    event_idxs,
    modality,
    temperature,
    top_p,
    use_memory):
  """Sample from a model given events and update the events.

  Args:
    i: The index used to update events.
    events_rng_memory: A tuple containing
      (a) events, rng, and memory if use_memory is True, or
      (b) events, rng if use_memory is False.
    forward_fn: A function that runs the model forward.
    condition_on: A string describing which events should be used as inputs to
      the model.
    event_idxs: An array of the positions of the event array.
    modality: A string giving the modality of events.
    temperature: The temperature used for sampling.
    top_p: The top_p parameter for nucleus sampling.
    use_memory: Whether to use memory (caching).
  Returns:
    The updated events, rng, and memory.
  """
  if use_memory:
    events, rng, memory = events_rng_memory
    # Generate only one output at a time:
    z_index_dim = 1
  else:
    events, rng = events_rng_memory
    memory = None
    z_index_dim = None
  rng, model_rng, sample_rng = jax.random.split(rng, 3)

  def run_forward(inputs, input_idxs, z_index_dim, memory):
    output, _ = forward_fn(
        model_rng,
        inputs,
        input_idxs,
        context=None,
        is_training=False,
        memory_type='fixed_size_kv',
        memory=memory,
        z_index_dim=z_index_dim,
    )

    return output.input_events_logits[:, -1, :], output.perceiver_state

  if condition_on == 'all_previous':
    logits, memory = run_forward(
        events[:, :i],
        event_idxs[:, :i],
        z_index_dim=z_index_dim,
        memory=memory)
  elif condition_on == 'most_recent_only':
    assert use_memory
    z_index_dim_memory = memory[1].memory_mask.shape[1]

    assert z_index_dim_memory % 2 == 0
    reset_buffer_size = z_index_dim_memory // 2

    # Caching introduces latent dependencies beyond what we use in training.
    # To counter this effect, whenever the memory cache is "full", rather than
    # just rotating old positions off, we reset the memory with a full run of
    # the model to fill half the memory and then proceed again until it is full.

    def infer_one(memory):
      """Infer next position using memory cache for all previous positions."""
      logits, memory = run_forward(
          events[:, None, i-1],
          event_idxs[:, None, i-1],
          z_index_dim=z_index_dim,
          memory=memory)
      return logits, memory

    def reset_memory(memory):
      """Infer next position without any cache and fill cache to half full."""

      # Zero out all positions for the cross-attend layer.
      memory[0] = jax.tree_map(jnp.zeros_like, memory[0])

      # For self-attend layers, create a memory buffer that is half the size
      # of the full memory buffer.
      memory[1:] = jax.tree_map(
          lambda x: jnp.zeros(  # pylint: disable=g-long-lambda
              x.shape[:1] + (reset_buffer_size,) + x.shape[2:], dtype=x.dtype),
          memory[1:])

      # Infer next step using the full sequence.
      # TODO(fjord): it's possible to use cross-attend memory for the portion
      # of the sequence that has already been inferred, but it's tricky to get
      # the masking to line up correctly and the results of the full
      # cross-attend query passed through to subsequent layers. For now, just
      # use the full sequence during these resets.
      logits, memory = run_forward(
          events[:, :-1],
          event_idxs[:, :-1],
          z_index_dim=reset_buffer_size,
          memory=memory)

      # Because we used the full sequence, the cross-attend memory is now
      # "full". Mask out all portions of the memory that will be occupied by
      # future positions and shift the memory so that it is left padded.
      def reset_ca_mem(x):
        mask = jnp.expand_dims(jnp.arange(x.shape[1]) < i, axis=0)
        while mask.ndim < x.ndim:
          mask = jnp.expand_dims(mask, axis=-1)
        x = jnp.where(
            jnp.broadcast_to(mask, x.shape),
            x, 0)
        x = jnp.roll(x, -i, axis=1)
        return x

      memory[0] = jax.tree_map(reset_ca_mem, memory[0])

      # Next inference will happen with the full z_index_dim, so left pad the
      # self-attend memory to be the full size.
      memory[1:] = jax.tree_map(
          lambda x: pad_axis(  # pylint: disable=g-long-lambda
              x, (z_index_dim_memory - reset_buffer_size, 0), axis=1),
          memory[1:])

      return logits, memory

    # If memory is full, reset to where it is half full, otherwise just
    # calculate logits for the next position using memory for previous positions
    logits, memory = jax.lax.cond(
        jnp.sum(memory[1].memory_mask[0]) == z_index_dim_memory,
        reset_memory,
        infer_one,
        operand=memory)
  elif condition_on == 'all':
    # TODO(drewjaegle): add max_context check?
    logits, memory = run_forward(
        events,
        event_idxs,
        z_index_dim=z_index_dim,
        memory=memory)
  else:
    raise ValueError(f'Unknown value for condition_on: {condition_on}')

  if temperature > 0 and top_p > 0:
    logits = process_logits(
        i, logits, top_p=top_p, temperature=temperature, modality=modality)
    tokens = jax.random.categorical(sample_rng, logits)
  else:
    tokens = jnp.argmax(logits, axis=-1)
  tokens = jnp.expand_dims(tokens, axis=-1)
  # events[:, i, ...] <- tokens  # tokens is shape [b, 1, ...]
  events = jax.lax.dynamic_update_slice_in_dim(events, tokens, i, 1)

  if use_memory:
    outputs = (events, rng, memory)
  else:
    outputs = (events, rng)
  return outputs
