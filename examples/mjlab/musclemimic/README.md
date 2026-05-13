# Musclemimic Example

Source: https://github.com/amathislab/musclemimic

Demo: WIP

Trains the [`myoMimicFullbody-v0`](https://github.com/ttktjmt/myosuite4/blob/colab/tutorials/mimic/train_colab.ipynb) task on mjlab and replays it in the browser. Policy: W&B run `ttktjmt-org/mjlab/zyklrroq`.

## Prerequisites

1. **Install extras** — this example needs `huggingface_hub` (clip download) and `musclemimic_models` (myosuite mjlab registration), which aren't pulled in by the default mjswan venv. Listed in [`requirements.txt`](requirements.txt):
   ```sh
   uv pip install -r examples/mjlab/musclemimic/requirements.txt
   ```

2. **Hugging Face access** — `amathislab/musclemimic-retargeted` is a gated dataset.
   - Accept the license at https://huggingface.co/datasets/amathislab/musclemimic-retargeted.
   - Authenticate locally: `huggingface-cli login` (a read-scope token is enough).

3. **W&B credentials** — needed to fetch the trained checkpoints and motion artifact:
   ```sh
   wandb login
   ```

## Run

```sh
uv run python -m examples.mjlab.musclemimic.main
```
