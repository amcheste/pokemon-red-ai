#!/usr/bin/env bash
#
# mirror_paper_to_overleaf.sh — push the local paper/ directory to Overleaf.
#
# Treats the git repo as canonical: this script copies the contents of
# `paper/` into the Overleaf project's local clone (managed by
# overleaf-mcp at ~/.cache/overleaf-mcp/<alias>/), commits, and pushes.
# Overleaf's web UI updates within seconds.
#
# Pulls from Overleaf first (`git pull --ff-only`) so manual edits in
# the web UI don't get clobbered — if someone has edited there since
# the last mirror, this script aborts and prints instructions for
# resolving the divergence.
#
# Usage:
#   scripts/mirror_paper_to_overleaf.sh [--alias NAME] [--source-dir DIR] [--dry-run]
#
# Defaults:
#   --alias       pokemon-rl-ewrl-2026
#   --source-dir  paper/
#
# Run after every paper edit, or wire into a git post-commit hook.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALIAS="pokemon-rl-ewrl-2026"
SOURCE_DIR="paper"
COMMIT_MESSAGE=""
DRY_RUN=0
CACHE_ROOT="${OVERLEAF_MCP_CACHE:-$HOME/.cache/overleaf-mcp}"

# ──────────────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────────────

usage() {
  cat <<'EOF'
Usage: mirror_paper_to_overleaf.sh [options]

Optional:
  --alias NAME         Overleaf project alias (default: pokemon-rl-ewrl-2026).
  --source-dir DIR     Local source directory (default: paper).
  --commit-message STR Override the auto-generated commit message.
  --dry-run            Preview the rsync changes without copying or pushing.
  -h, --help           Show this help.

Environment:
  OVERLEAF_MCP_CACHE   Override the local cache root.

Behaviour:
  1. cd into the Overleaf clone at <cache>/<alias>/
  2. git pull --ff-only (refuses to proceed if Overleaf has diverged)
  3. rsync paper/ → clone (with --delete, so paper/ is canonical)
  4. git add -A
  5. git commit (skipped if no changes)
  6. git push

Re-running with no local changes is a no-op.
EOF
}

# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --alias)          ALIAS="$2";          shift 2 ;;
    --source-dir)     SOURCE_DIR="$2";     shift 2 ;;
    --commit-message) COMMIT_MESSAGE="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=1;           shift ;;
    -h|--help)        usage; exit 0 ;;
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

err() { echo "ERROR: $*" >&2; exit 2; }

cd "$PROJECT_ROOT"

[[ -d "$SOURCE_DIR" ]] || err "source dir does not exist: $SOURCE_DIR"

CLONE_DIR="$CACHE_ROOT/$ALIAS"
[[ -d "$CLONE_DIR/.git" ]] || err "Overleaf clone not found at $CLONE_DIR.
       Run: bin/setup-overleaf-project.sh --alias $ALIAS --project-id <id>
       (set OVERLEAF_TOKEN first)"

command -v rsync >/dev/null 2>&1 || err "rsync not on PATH"

# Auto-generate commit message if none provided
if [[ -z "$COMMIT_MESSAGE" ]]; then
  git_sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  git_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  COMMIT_MESSAGE="Mirror paper/ from pokemon-red-ai@${git_sha} (${git_branch})"
fi

# ──────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────

echo "──────────────────────────────────────────────────────────────"
echo "Mirror paper/ → Overleaf"
echo "──────────────────────────────────────────────────────────────"
echo "Alias:           $ALIAS"
echo "Source:          $PROJECT_ROOT/$SOURCE_DIR"
echo "Clone:           $CLONE_DIR"
echo "Commit message:  $COMMIT_MESSAGE"
echo "Dry run:         $([[ "$DRY_RUN" -eq 1 ]] && echo yes || echo no)"
echo "──────────────────────────────────────────────────────────────"

# ──────────────────────────────────────────────────────────────────────
# 1. Pull from Overleaf to catch up with any web-UI edits
# ──────────────────────────────────────────────────────────────────────

echo
echo "[1/4] Pulling latest from Overleaf..."
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  (dry run — skipping)"
else
  if ! (cd "$CLONE_DIR" && git pull --ff-only 2>&1); then
    err "git pull --ff-only failed.  Possible causes:
       - someone edited the project in Overleaf's web UI since the last mirror,
         and that edit conflicts with files we're about to overwrite.
       - the cached clone is dirty (uncommitted changes from a previous failed mirror).

       Resolve manually:
         cd $CLONE_DIR
         git status
         git pull --no-rebase    # if a merge is needed
       Then re-run this script."
  fi
fi

# ──────────────────────────────────────────────────────────────────────
# 2. Rsync the paper/ directory into the clone
# ──────────────────────────────────────────────────────────────────────

echo
echo "[2/4] Syncing paper/ → clone..."

# --delete so removed files are pruned on the Overleaf side too.
# Exclude .git so we don't blow away the clone's git metadata, plus
# common LaTeX build artefacts that shouldn't go to Overleaf.
rsync_args=(
  -av
  --delete
  --exclude='.git/'
  --exclude='*.aux'
  --exclude='*.bbl'
  --exclude='*.blg'
  --exclude='*.log'
  --exclude='*.out'
  --exclude='*.toc'
  --exclude='*.fdb_latexmk'
  --exclude='*.fls'
  --exclude='*.synctex.gz'
)
if [[ "$DRY_RUN" -eq 1 ]]; then
  rsync_args+=(--dry-run)
fi

# Trailing slash on source = "copy contents of paper/", not "copy paper/ as a subdir"
rsync "${rsync_args[@]}" "$SOURCE_DIR/" "$CLONE_DIR/"

# ──────────────────────────────────────────────────────────────────────
# 3. Commit (skip if no changes)
# ──────────────────────────────────────────────────────────────────────

echo
echo "[3/4] Committing changes..."

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  (dry run — skipping)"
else
  cd "$CLONE_DIR"
  git add -A
  if git diff --cached --quiet; then
    echo "  no changes to commit — Overleaf is already in sync"
    exit 0
  fi
  git commit -m "$COMMIT_MESSAGE" 2>&1 | sed 's/^/    /'
fi

# ──────────────────────────────────────────────────────────────────────
# 4. Push to Overleaf
# ──────────────────────────────────────────────────────────────────────

echo
echo "[4/4] Pushing to Overleaf..."

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  (dry run — skipping)"
else
  if ! git push 2>&1 | sed 's/^/    /'; then
    err "git push failed.  Check the Overleaf project's git auth (token may have expired)."
  fi
fi

echo
echo "──────────────────────────────────────────────────────────────"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete.  Re-run without --dry-run to actually mirror."
else
  echo "Mirrored.  Refresh the Overleaf web UI to see the changes."
fi
echo "──────────────────────────────────────────────────────────────"
