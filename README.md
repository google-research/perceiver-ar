# Perceiver AR

Perceiver AR is an autoregressive, modality-agnostic architecture which uses
cross-attention to map long-range inputs to a small number of latents while also
maintaining end-to-end causal masking. Perceiver AR can directly attend to over
a hundred thousand tokens, enabling practical long-context density estimation
without the need for hand-crafted sparsity patterns or memory mechanisms.

For more details, see our ICML paper: https://arxiv.org/abs/2202.07765

## Setup

First, install dependencies following these instructions:

1. Create a virtual env: `python3 -m venv ~/.venv/perceiver-ar`
2. Switch to the virtual env: `source ~/.venv/perceiver-ar/bin/activate`
3. Follow instructions for installing JAX on your platform:
   https://github.com/google/jax#installation
4. Install other dependencies: `pip install -f requirements.txt`

## Training

```
python experiment.py  --config=experiment.py:random_mirrored_32
```

By default, checkpoints and events will be saved to `/tmp/perceiver_ar`.

Training metrics will be periodically written to Tensorboard event files which
can be viewed using:

```
tensorboard --logdir /tmp/perceiver_ar/
```

During training, use Ctrl+C to save a checkpoint and Ctrl+\ to save a checkpoint
and exit.

## Evaluation

To evaluate the latest saved checkpoint:

```
CHECKPOINTS="/tmp/perceiver_ar"
LATEST_CHECKPOINT="${CHECKPOINTS}/models/latest/$(ls -tr ${CHECKPOINTS}/models/latest/ | tail -n 1)"
echo "Evaluating ${LATEST_CHECKPOINT}"
python experiment.py  --config=experiment.py:random_mirrored_32 \
  --jaxline_mode=eval \
  --config.one_off_evaluate=True \
  --config.restore_path="${LATEST_CHECKPOINT}"
```

Results will be written to the console and can also be viewed from Tensorboard.

## Inference

To run inference in a local Jupyter notebook:

```
jupyter notebook
```

Load `inference.ipynb`.

Set `modality`, `sweep_name`, and `checkpoint_base` appropriately and run all
of the cells.

## Unit Tests

To run all unit tests:

```
pytest
```

## Disclaimer

This is not an officially supported Google product.
