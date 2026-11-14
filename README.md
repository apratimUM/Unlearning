# Unlearning — GPT-2 / Pile Baseline

Code for training GPT-2 Small from scratch on the deduplicated Pile, with a separate official Pile validation set.

## Files

| File | Purpose |
| --- | --- |
| `prepare.py` | Streams and tokenizes the deduplicated Pile, writes `train.bin` and official validation `val.bin`, and supports resumable preprocessing. |
| `data.py` | Reads packed token data, constructs 1,024-token input/target pairs, and applies the saved training-block order. |
| `model.py` | Defines the GPT-2 Small model and text generation methods. |
| `runtime.py` | Provides CUDA, mixed-precision, and optimizer-step utilities. |
| `train.py` | Trains with four-GPU DDP, handles block ordering, evaluation, logging, checkpoints, and resume. |
| `token.sbatch` | Launches training on Frances and handles graceful checkpoint-and-resubmit between Slurm jobs. |
| `evaluation.py` | Evaluates checkpoints on WikiText-103, LAMBADA, BLiMP, and MMLU. |
| `checkpoints.py` | Checks dataset and checkpoint consistency and can invoke benchmark evaluation. |
| `requirements.txt` | Lists Python dependencies. |

Training data and model checkpoints are stored outside GitHub (on Turbo). The Slurm script requires its cluster paths and environment to be configured before use.
