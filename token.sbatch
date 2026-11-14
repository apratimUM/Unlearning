#!/bin/bash
#SBATCH --job-name=gpt2_pile_ddp4_full_v2
#SBATCH --partition=pdefault
#SBATCH --account=aroyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=/z/apratim/project/project/ddp4-full-%j.log

set -Eeuo pipefail
PROJECT=/z/apratim/project/project
DATA=/coe-aroyc/apratim/pile_full_data
RUN_ROOT=/coe-aroyc/apratim/runs25
# New lineage. Do NOT reuse the old batch=128, accum=1 checkpoint.
RUN_NAME=gpt2_pile_ddp4_docshuf42_samefolder_b512_v1
SCRIPT="$PROJECT/token.sbatch"
TRAINER="$PROJECT/train.py"
RUN_DIR="$RUN_ROOT/$RUN_NAME"

cd "$PROJECT"
PYTHON="$(tr -d '\r\n' < "$PROJECT/.python_path")"
for p in "$PYTHON" "$TRAINER" "$SCRIPT"; do
  [[ -e "$p" ]] || { echo "[ERROR] Missing: $p" >&2; exit 1; }
done
[[ -x "$PYTHON" ]] || { echo "[ERROR] Python is not executable: $PYTHON" >&2; exit 1; }
[[ -r "$TRAINER" && -r "$SCRIPT" ]] || { echo '[ERROR] Trainer or script unreadable' >&2; exit 1; }
for name in model.py data.py runtime.py; do
  [[ -r "$PROJECT/$name" ]] || { echo "[ERROR] Missing $name" >&2; exit 1; }
done
for name in manifest_shuffled.json train_shuffled.bin val.bin documents_shuffled.sqlite train_chunk_order_shuffled.u32 train_chunk_inverse_shuffled.u32; do
  [[ -r "$DATA/$name" ]] || { echo "[ERROR] Missing $DATA/$name" >&2; exit 1; }
done
mkdir -p "$RUN_ROOT"

RESUME=()
if [[ -e "$RUN_DIR" ]]; then
  [[ -f "$RUN_DIR/latest.pt" ]] || {
    echo "[ERROR] Run directory exists without latest.pt: $RUN_DIR" >&2
    exit 1
  }
  RESUME=(--resume)
fi

export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRAIN_STOP_FILE="$RUN_ROOT/.${RUN_NAME}.stop-request"
export TRAIN_SEGMENT_RESULT="$RUN_ROOT/.${RUN_NAME}.segment-${SLURM_JOB_ID}.json"
unset TRAIN_TEST_STOP_STEP TRAIN_EXPECTED_RESUME_STEP
rm -f "$TRAIN_STOP_FILE" "$TRAIN_SEGMENT_RESULT"

nvidia-smi -L
GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ')"
[[ "$GPU_COUNT" == 4 ]] || { echo "[ERROR] Expected 4 GPUs, found $GPU_COUNT" >&2; exit 1; }
echo "[START] job=$SLURM_JOB_ID host=$(hostname) date=$(date -Is) resume=${#RESUME[@]}"
echo "[RUN] $RUN_DIR"
echo '[CONFIG] DDP=4 batch=128 accum=4 effective_batch=512 block=1024 steps=0 precision=bfloat16 lr=6e-4 min_lr=6e-5 warmup=2000'
echo '[CONFIG] eval_every=10000 eval_batches=16 checkpoint_wallclock=6h max_job=12h stop_signal=USR1@1800s'

if [[ -d "$DATA/tokenizer" ]]; then
  CHECK_ARGS=(--check-begin 10000 --check-every 10000)
else
  echo "[WARNING] Missing $DATA/tokenizer; disabling generation checks (validation still enabled)"
  CHECK_ARGS=(--check-every 0)
fi

WALLTIME=0
on_walltime() {
  WALLTIME=1
  echo "[SIGNAL] $(date -Is) USR1: requesting safe checkpoint and shutdown"
  touch "$TRAIN_STOP_FILE"
}
trap on_walltime USR1

# Best-effort Linux page-cache warmup: not pinned, may be evicted, and reads
# compete for Turbo bandwidth while running. Requires sufficient node RAM.
echo "[CACHE] warming $DATA/train_shuffled.bin asynchronously"
dd if="$DATA/train_shuffled.bin" of=/dev/null bs=64M status=none &
WARM_PID=$!

"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -- "$TRAINER" \
  --data "$DATA" --shuffled --run-root "$RUN_ROOT" --run "$RUN_NAME" \
  --model small --block-size 1024 --batch 128 --accum 4 \
  --steps 0 --seed 42 --precision bfloat16 \
  --lr 6e-4 --min-lr 6e-5 --warmup 2000 --lr-schedule cosine \
  --weight-decay 0.1 --grad-clip 1.0 --beta1 0.9 --beta2 0.95 \
  --eval-every 10000 --eval-batches 16 \
  --save-every 0 --save-hours 6 \
  "${CHECK_ARGS[@]}" \
  --log-every 100 --eta-window 200 --max-loss 15 \
  "${RESUME[@]}" &
PY_PID=$!

set +e
while :; do
  wait "$PY_PID"
  RC=$?
  if (( RC > 128 )) && kill -0 "$PY_PID" 2>/dev/null; then
    continue
  fi
  break
done
set -e
echo "[EXIT] torchrun rc=$RC"
# Stop any unfinished cache warming before this job terminates.
if kill -0 "$WARM_PID" 2>/dev/null; then
  kill "$WARM_PID" 2>/dev/null || true
fi
wait "$WARM_PID" 2>/dev/null || true
(( RC == 0 )) || { echo '[ERROR] Training failed; no automatic successor submitted' >&2; exit "$RC"; }
[[ -f "$TRAIN_SEGMENT_RESULT" && -f "$RUN_DIR/latest.pt" ]] || {
  echo '[ERROR] Missing segment result or durable latest.pt; no successor submitted' >&2; exit 1;
}

read -r STATUS STEP TARGET < <("$PYTHON" - "$TRAIN_SEGMENT_RESULT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
status = r['status']
step, target = int(r['step']), int(r['target_steps'])
assert status in ('stopped', 'complete') and 0 < step <= target
assert (status == 'complete') == (step == target), r
assert int(r['optimizer_states']) > 0
print(status, step, target)
PY
)
echo "[RESULT] status=$STATUS step=$STEP target=$TARGET walltime_signal=$WALLTIME"
if [[ "$STATUS" == complete ]]; then
  echo "[FULL_PASS_COMPLETE] job=$SLURM_JOB_ID steps=$STEP"
  exit 0
fi

NEXT_JOB=$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" "$SCRIPT")
echo "[RESUBMIT] successor_job=$NEXT_JOB checkpoint_step=$STEP"
