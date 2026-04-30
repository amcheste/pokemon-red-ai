#!/usr/bin/env bash
#
# run_pilots.sh — launch the EWRL 2026 pilot grid (3 treatments × 3 seeds).
#
# Wraps scripts/train.py with consistent save-dir / W&B-run-name paths,
# caffeinate on macOS so the system doesn't sleep mid-run, log
# redirection per run, optional parallelism, and skip-completed
# detection so re-runs after a crash don't redo finished work.
#
# Usage:
#   scripts/run_pilots.sh --rom path/to/PokemonRed.gb [options]
#
# See --help for all options.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-./venv/bin/python3}"

ROM_PATH=""
SAVE_STATE="states/s0_post_intro.state"
TOTAL_TIMESTEPS="10000000"
TREATMENTS="pixel,symbolic,hybrid"
SEEDS="42,123,456"
SAVE_ROOT="./training_output"
LOG_DIR="./logs"
ALERTS_CONFIG="configs/alerts.yaml"
WANDB_PROJECT="pokemon-red-ai"
PARALLEL=1
N_ENVS=4
DEVICE="auto"
DRY_RUN=0
NO_CAFFEINATE=0
SKIP_COMPLETED=1
EXTRA_ARGS=()

# ──────────────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────────────

usage() {
  cat <<'EOF'
run_pilots.sh — launch the EWRL pilot grid (3 treatments × 3 seeds = 9 runs).

Required:
  --rom PATH                Pokemon Red ROM (.gb) file.

Optional:
  --save-state PATH         PyBoy save state to reset to.
                            Default: states/s0_post_intro.state
  --total-timesteps N       Steps per run.  Default: 10000000 (10M)
  --n-envs N                Parallel envs per pilot (SubprocVecEnv).
                            Default: 4.  Each env uses one CPU core.
  --device {auto,cpu,cuda,mps}
                            PyTorch device for the policy network.
                            Default: mps (Apple Silicon GPU).  Set to
                            'auto' on non-Apple-Silicon hardware.
  --treatments LIST         Comma-separated observation types.
                            Default: pixel,symbolic,hybrid
  --seeds LIST              Comma-separated seeds.
                            Default: 42,123,456
  --save-root DIR           Parent directory for per-run save dirs.
                            Default: ./training_output
  --log-dir DIR             Where to write per-run log files.
                            Default: ./logs
  --alerts-config PATH      YAML alerts config (passes --alerts-config
                            and --enable-alerts to train.py).
                            Default: configs/alerts.yaml (if exists)
  --wandb-project NAME      W&B project name.
                            Default: pokemon-red-ai
  --parallel N              Run N pilots concurrently (waits for each
                            batch before launching the next).
                            Default: 1 (fully sequential)
  --no-caffeinate           Don't wrap commands in `caffeinate -i`
                            (default ON when on macOS).
  --no-skip-completed       Re-run pilots that already have a
                            final_model.zip in their save dir.
                            Default: skip them.
  --dry-run                 Print the commands that would run without
                            executing them.
  --python PATH             Python interpreter override (otherwise
                            uses ./venv/bin/python3 or $PYTHON).
  --                        Anything after `--` is forwarded to every
                            train.py invocation as extra args.
  -h, --help                Show this help.

Examples:

  # Sequential (one at a time), default 9-run grid:
  scripts/run_pilots.sh --rom PokemonRed.gb

  # 3 concurrent runs (one per treatment × different seeds):
  scripts/run_pilots.sh --rom PokemonRed.gb --parallel 3

  # Subset — just the seed-42 runs across all 3 treatments:
  scripts/run_pilots.sh --rom PokemonRed.gb --seeds 42

  # Forward extra args to train.py (e.g. faster save-freq):
  scripts/run_pilots.sh --rom PokemonRed.gb -- --save-freq 100000

  # Dry run — print commands without executing:
  scripts/run_pilots.sh --rom PokemonRed.gb --parallel 3 --dry-run

EOF
}

# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rom)               ROM_PATH="$2";               shift 2 ;;
    --save-state)        SAVE_STATE="$2";             shift 2 ;;
    --total-timesteps)   TOTAL_TIMESTEPS="$2";        shift 2 ;;
    --n-envs)            N_ENVS="$2";                 shift 2 ;;
    --device)            DEVICE="$2";                 shift 2 ;;
    --treatments)        TREATMENTS="$2";             shift 2 ;;
    --seeds)             SEEDS="$2";                  shift 2 ;;
    --save-root)         SAVE_ROOT="$2";              shift 2 ;;
    --log-dir)           LOG_DIR="$2";                shift 2 ;;
    --alerts-config)     ALERTS_CONFIG="$2";          shift 2 ;;
    --wandb-project)     WANDB_PROJECT="$2";          shift 2 ;;
    --parallel)          PARALLEL="$2";               shift 2 ;;
    --no-caffeinate)     NO_CAFFEINATE=1;             shift ;;
    --no-skip-completed) SKIP_COMPLETED=0;            shift ;;
    --dry-run)           DRY_RUN=1;                   shift ;;
    --python)            PYTHON="$2";                 shift 2 ;;
    -h|--help)           usage; exit 0 ;;
    --)                  shift; EXTRA_ARGS=("$@");    break ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      echo "Run with --help for usage." >&2
      exit 2
      ;;
  esac
done

# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────

cd "$PROJECT_ROOT"

if [[ -z "$ROM_PATH" ]]; then
  echo "ERROR: --rom is required." >&2
  echo "Run with --help for usage." >&2
  exit 2
fi

if [[ ! -f "$ROM_PATH" ]]; then
  echo "ERROR: ROM not found: $ROM_PATH" >&2
  exit 2
fi

if [[ ! -f "$SAVE_STATE" ]]; then
  echo "WARNING: save state not found: $SAVE_STATE" >&2
  echo "         train.py will start episodes from a fresh boot —" >&2
  echo "         this is much slower.  Run scripts/create_save_states.py first." >&2
fi

if [[ ! -x "$PYTHON" && ! "$PYTHON" =~ ^python ]]; then
  echo "ERROR: Python not found or not executable: $PYTHON" >&2
  echo "       Override with --python or \$PYTHON env var." >&2
  exit 2
fi

if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [[ "$PARALLEL" -lt 1 ]]; then
  echo "ERROR: --parallel must be a positive integer (got: $PARALLEL)" >&2
  exit 2
fi

# Decide whether to wrap with caffeinate
USE_CAFFEINATE=0
if [[ "$NO_CAFFEINATE" -eq 0 && "$(uname)" == "Darwin" ]]; then
  if command -v caffeinate >/dev/null 2>&1; then
    USE_CAFFEINATE=1
  fi
fi

# Resolve alerts-config: pass it to train.py only if the file exists.
ALERTS_FLAGS=()
if [[ -f "$ALERTS_CONFIG" ]]; then
  ALERTS_FLAGS+=(--enable-alerts --alerts-config "$ALERTS_CONFIG")
fi

mkdir -p "$LOG_DIR" "$SAVE_ROOT"

# ──────────────────────────────────────────────────────────────────────
# Build the run grid
# ──────────────────────────────────────────────────────────────────────

IFS=',' read -ra TREATMENTS_ARR <<< "$TREATMENTS"
IFS=',' read -ra SEEDS_ARR <<< "$SEEDS"

declare -a GRID_TREATMENT GRID_SEED
for t in "${TREATMENTS_ARR[@]}"; do
  t_trim="$(echo -n "$t" | tr -d '[:space:]')"
  for s in "${SEEDS_ARR[@]}"; do
    s_trim="$(echo -n "$s" | tr -d '[:space:]')"
    GRID_TREATMENT+=("$t_trim")
    GRID_SEED+=("$s_trim")
  done
done

TOTAL_RUNS="${#GRID_TREATMENT[@]}"

# ──────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────

echo "──────────────────────────────────────────────────────────────"
echo "Pokemon Red RL — pilot launcher"
echo "──────────────────────────────────────────────────────────────"
echo "ROM:               $ROM_PATH"
echo "Save state:        $SAVE_STATE"
echo "Steps per run:     $TOTAL_TIMESTEPS"
echo "Treatments:        ${TREATMENTS_ARR[*]}"
echo "Seeds:             ${SEEDS_ARR[*]}"
echo "Total runs:        $TOTAL_RUNS"
echo "Parallelism:       $PARALLEL"
echo "Save root:         $SAVE_ROOT"
echo "Log dir:           $LOG_DIR"
echo "Alerts config:     ${ALERTS_CONFIG} $([[ -f "$ALERTS_CONFIG" ]] && echo '(found)' || echo '(not found — alerts disabled)')"
echo "W&B project:       $WANDB_PROJECT"
echo "Python:            $PYTHON"
echo "Caffeinate:        $([[ "$USE_CAFFEINATE" -eq 1 ]] && echo enabled || echo disabled)"
echo "Skip completed:    $([[ "$SKIP_COMPLETED" -eq 1 ]] && echo yes || echo no)"
echo "Dry run:           $([[ "$DRY_RUN" -eq 1 ]] && echo yes || echo no)"
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  echo "Extra args:        ${EXTRA_ARGS[*]}"
fi
echo "──────────────────────────────────────────────────────────────"

# ──────────────────────────────────────────────────────────────────────
# Build per-run command
# ──────────────────────────────────────────────────────────────────────

build_cmd() {
  local treatment="$1"
  local seed="$2"
  local save_dir="$SAVE_ROOT/${treatment}-seed${seed}"
  local run_name="${treatment}-seed${seed}"
  local log_file="$LOG_DIR/${run_name}.log"

  local -a cmd=()
  if [[ "$USE_CAFFEINATE" -eq 1 ]]; then
    cmd+=("caffeinate" "-i")
  fi
  cmd+=(
    "$PYTHON" "scripts/train.py"
    --rom "$ROM_PATH"
    --save-state "$SAVE_STATE"
    --observation-type "$treatment"
    --algorithm "RecurrentPPO"
    --reward-strategy "events"
    --total-timesteps "$TOTAL_TIMESTEPS"
    --n-envs "$N_ENVS"
    --device "$DEVICE"
    --seed "$seed"
    --save-dir "$save_dir"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$run_name"
    -v
  )
  if [[ "${#ALERTS_FLAGS[@]}" -gt 0 ]]; then
    cmd+=("${ALERTS_FLAGS[@]}")
  fi
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  # Print the resolved command + log path (callers handle &> redirection)
  printf '%q ' "${cmd[@]}"
  printf '> %q 2>&1\n' "$log_file"
}

is_completed() {
  local treatment="$1"
  local seed="$2"
  local save_dir="$SAVE_ROOT/${treatment}-seed${seed}"
  [[ -f "$save_dir/models/final_model.zip" ]]
}

run_one() {
  local treatment="$1"
  local seed="$2"
  local run_name="${treatment}-seed${seed}"
  local log_file="$LOG_DIR/${run_name}.log"

  echo "[$(date +%H:%M:%S)] starting: $run_name → $log_file"

  local -a cmd=()
  if [[ "$USE_CAFFEINATE" -eq 1 ]]; then
    cmd+=("caffeinate" "-i")
  fi
  cmd+=(
    "$PYTHON" "scripts/train.py"
    --rom "$ROM_PATH"
    --save-state "$SAVE_STATE"
    --observation-type "$treatment"
    --algorithm "RecurrentPPO"
    --reward-strategy "events"
    --total-timesteps "$TOTAL_TIMESTEPS"
    --n-envs "$N_ENVS"
    --device "$DEVICE"
    --seed "$seed"
    --save-dir "$SAVE_ROOT/${run_name}"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$run_name"
    -v
  )
  if [[ "${#ALERTS_FLAGS[@]}" -gt 0 ]]; then
    cmd+=("${ALERTS_FLAGS[@]}")
  fi
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  if "${cmd[@]}" > "$log_file" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK:       $run_name"
    return 0
  else
    local rc=$?
    echo "[$(date +%H:%M:%S)] FAILED:   $run_name (exit $rc) — see $log_file" >&2
    return "$rc"
  fi
}

# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────

declare -i SKIPPED=0 COMPLETED=0 FAILED=0
declare -a FAILED_RUNS=()

# Filter the grid through skip-completed
declare -a TODO_TREATMENT TODO_SEED
for ((i=0; i<TOTAL_RUNS; i++)); do
  t="${GRID_TREATMENT[$i]}"
  s="${GRID_SEED[$i]}"
  if [[ "$SKIP_COMPLETED" -eq 1 ]] && is_completed "$t" "$s"; then
    echo "skip:    ${t}-seed${s} (final_model.zip already present)"
    SKIPPED+=1
    continue
  fi
  TODO_TREATMENT+=("$t")
  TODO_SEED+=("$s")
done

TODO_COUNT="${#TODO_TREATMENT[@]}"

if [[ "$TODO_COUNT" -eq 0 ]]; then
  echo "All $TOTAL_RUNS runs already complete.  Nothing to do."
  exit 0
fi

echo "──────────────────────────────────────────────────────────────"
echo "Will run: $TODO_COUNT of $TOTAL_RUNS pilots ($SKIPPED skipped)"
echo "──────────────────────────────────────────────────────────────"

if [[ "$DRY_RUN" -eq 1 ]]; then
  for ((i=0; i<TODO_COUNT; i++)); do
    t="${TODO_TREATMENT[$i]}"
    s="${TODO_SEED[$i]}"
    echo
    echo "# Run $((i+1))/$TODO_COUNT — ${t}-seed${s}"
    build_cmd "$t" "$s"
  done
  echo
  echo "(dry run — no commands executed)"
  exit 0
fi

# Sequential or batched-parallel execution.
declare -a BATCH_PIDS=()
declare -a BATCH_NAMES=()

flush_batch() {
  local rc=0
  local -i j
  for ((j=0; j<${#BATCH_PIDS[@]}; j++)); do
    local pid="${BATCH_PIDS[$j]}"
    local name="${BATCH_NAMES[$j]}"
    if wait "$pid"; then
      COMPLETED+=1
    else
      local r=$?
      FAILED+=1
      FAILED_RUNS+=("$name (exit $r)")
      rc=1
    fi
  done
  BATCH_PIDS=()
  BATCH_NAMES=()
  return "$rc"
}

for ((i=0; i<TODO_COUNT; i++)); do
  t="${TODO_TREATMENT[$i]}"
  s="${TODO_SEED[$i]}"
  name="${t}-seed${s}"

  run_one "$t" "$s" &
  BATCH_PIDS+=("$!")
  BATCH_NAMES+=("$name")

  if [[ "${#BATCH_PIDS[@]}" -ge "$PARALLEL" ]]; then
    flush_batch || true
  fi
done
flush_batch || true

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

echo "──────────────────────────────────────────────────────────────"
echo "Pilot launch summary"
echo "──────────────────────────────────────────────────────────────"
echo "Completed:  $COMPLETED"
echo "Failed:     $FAILED"
echo "Skipped:    $SKIPPED  (already had final_model.zip)"
echo

if [[ "$FAILED" -gt 0 ]]; then
  echo "Failed runs:"
  for f in "${FAILED_RUNS[@]}"; do
    echo "  - $f"
  done
  echo
  echo "Inspect logs in $LOG_DIR/*.log to debug.  Re-run this script —"
  echo "by default, completed runs are skipped so only failures will retry."
  exit 1
fi

echo "All pilots done.  Next steps:"
echo "  streamlit run scripts/compare.py -- --runs-dir $SAVE_ROOT"
echo "  $PYTHON scripts/analyze.py --results-dir $SAVE_ROOT \\"
echo "      --output-dir paper/figures \\"
echo "      --plots aggregate profiles improvement efficiency \\"
echo "      --reps 10000 --format pdf"
