#!/usr/bin/env bash
#
# setup-overleaf-project.sh — one-shot Overleaf project setup.
#
# Wires a new Overleaf project into the local overleaf-mcp installation
# in three steps that the upstream `overleaf-mcp` CLI handles
# individually:
#
#   1. overleaf-mcp init       — write the alias / project_id mapping
#   2. overleaf-mcp auth add   — store the token in the OS keychain
#   3. git clone               — populate the local cache (the MCP server
#                                deliberately doesn't do clone orchestration)
#
# Then verifies via `overleaf-mcp doctor`.
#
# Requires: overleaf-mcp >= 0.1.2 (for non-interactive --alias and
# --token-stdin / --token-from-env flags).
#
# Usage:
#   bin/setup-overleaf-project.sh \
#       --alias pokemon-rl-ewrl-2026 \
#       --project-id 6620abc123def \
#       [--display-name "Pokemon RL — EWRL 2026"]
#
# The token is read from the OVERLEAF_TOKEN environment variable so it
# never appears on the command line (no leak via `ps`).  Either:
#
#   export OVERLEAF_TOKEN=<paste>
#   bin/setup-overleaf-project.sh --alias foo --project-id 123
#
# or:
#
#   OVERLEAF_TOKEN=<paste> bin/setup-overleaf-project.sh --alias foo --project-id 123
#
# Re-run is safe: existing alias prompts for --force, existing clone is
# left alone, existing token is overwritten.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

ALIAS=""
PROJECT_ID=""
DISPLAY_NAME=""
FORCE=0
SKIP_DOCTOR=0

OVERLEAF_MCP="${OVERLEAF_MCP:-overleaf-mcp}"
CACHE_ROOT="${OVERLEAF_MCP_CACHE:-$HOME/.cache/overleaf-mcp}"
MIN_MCP_VERSION="0.1.2"

# ──────────────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────────────

usage() {
  cat <<'EOF'
Usage: setup-overleaf-project.sh [options]

Required:
  --alias NAME        Short nickname for the project (e.g. "pokemon-rl-ewrl-2026").
  --project-id ID     Overleaf project ID (the hex string in the project URL).
  $OVERLEAF_TOKEN     Overleaf Git token.  Must be exported in the environment.

Optional:
  --display-name STR  Human-friendly display name shown in `overleaf-mcp doctor`.
  --force             Overwrite an existing alias without prompting.
  --skip-doctor       Don't run `overleaf-mcp doctor` at the end.
  -h, --help          Show this help.

Environment:
  OVERLEAF_TOKEN      Required.  Mint at Overleaf → Account Settings →
                      Git Integration → New token.
  OVERLEAF_MCP        CLI binary (default: `overleaf-mcp`; override for
                      a venv-pinned install).
  OVERLEAF_MCP_CACHE  Local clone cache root (default: ~/.cache/overleaf-mcp).

Example:
  export OVERLEAF_TOKEN=olp_xxxx
  bin/setup-overleaf-project.sh \
      --alias pokemon-rl-ewrl-2026 \
      --project-id 66201abc \
      --display-name "Pokemon RL — EWRL 2026"
EOF
}

# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --alias)        ALIAS="$2";        shift 2 ;;
    --project-id)   PROJECT_ID="$2";   shift 2 ;;
    --display-name) DISPLAY_NAME="$2"; shift 2 ;;
    --force)        FORCE=1;           shift ;;
    --skip-doctor)  SKIP_DOCTOR=1;     shift ;;
    -h|--help)      usage; exit 0 ;;
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

[[ -n "$ALIAS"      ]] || err "--alias is required"
[[ -n "$PROJECT_ID" ]] || err "--project-id is required"
[[ -n "${OVERLEAF_TOKEN:-}" ]] || err "OVERLEAF_TOKEN must be set in the environment"

command -v "$OVERLEAF_MCP" >/dev/null 2>&1 || err "overleaf-mcp not on PATH (set OVERLEAF_MCP=<path> if installed elsewhere)"
command -v git             >/dev/null 2>&1 || err "git not on PATH"

# Version check — bail early with a clear message rather than letting
# click choke on an unknown flag.
version_line="$("$OVERLEAF_MCP" --version 2>&1 | tr -d '\r')"
installed_version="$(echo "$version_line" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+/) { print $i; exit } }')"
if [[ -z "$installed_version" ]]; then
  err "could not parse overleaf-mcp version from: $version_line"
fi

# Sort versions: if the installed version sorts before the minimum, fail.
lowest="$(printf '%s\n%s\n' "$installed_version" "$MIN_MCP_VERSION" | sort -V | head -1)"
if [[ "$lowest" != "$MIN_MCP_VERSION" ]]; then
  err "overleaf-mcp ${installed_version} is too old; need >= ${MIN_MCP_VERSION}.
       Upgrade with:  pipx upgrade overleaf-mcp-server
       (or to test from source: pipx install --force git+https://github.com/amcheste/overleaf-mcp@develop)"
fi

# ──────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────

CLONE_DIR="$CACHE_ROOT/$ALIAS"

echo "──────────────────────────────────────────────────────────────"
echo "Setting up Overleaf project"
echo "──────────────────────────────────────────────────────────────"
echo "Alias:         $ALIAS"
echo "Project ID:    $PROJECT_ID"
echo "Display name:  ${DISPLAY_NAME:-(none)}"
echo "Clone path:    $CLONE_DIR"
echo "Tool:          $OVERLEAF_MCP ($installed_version)"
echo "──────────────────────────────────────────────────────────────"

# ──────────────────────────────────────────────────────────────────────
# 1. Register the alias in the config file (idempotent with --force)
# ──────────────────────────────────────────────────────────────────────

echo
echo "[1/4] Registering project alias in config..."

init_args=(init --alias "$ALIAS" --project-id "$PROJECT_ID")
if [[ -n "$DISPLAY_NAME" ]]; then
  init_args+=(--display-name "$DISPLAY_NAME")
fi
if [[ "$FORCE" -eq 1 ]]; then
  init_args+=(--force)
fi

if ! "$OVERLEAF_MCP" "${init_args[@]}"; then
  err "overleaf-mcp init failed.  If the alias already exists, re-run with --force."
fi

# ──────────────────────────────────────────────────────────────────────
# 2. Store the token in the keychain (via stdin so it doesn't show in ps)
# ──────────────────────────────────────────────────────────────────────

echo
echo "[2/4] Storing token in OS keychain..."

if ! printf '%s' "$OVERLEAF_TOKEN" | "$OVERLEAF_MCP" auth add --project "$ALIAS" --token-stdin; then
  err "overleaf-mcp auth add failed."
fi

# ──────────────────────────────────────────────────────────────────────
# 3. Clone the project locally if it doesn't already exist
# ──────────────────────────────────────────────────────────────────────

echo
echo "[3/4] Setting up local clone..."

if [[ -d "$CLONE_DIR/.git" ]]; then
  echo "  clone already exists at $CLONE_DIR — skipping (use git pull there to refresh)"
else
  mkdir -p "$(dirname "$CLONE_DIR")"
  # Token-embedded URL — git stores this in .git/config, which means
  # subsequent pull/push don't need any further auth helper.  Overleaf
  # requires the username to be literally "git" (not "x" or
  # "anything") — the server returns 403 otherwise.  See
  # https://www.overleaf.com/learn/how-to/Git_integration_authentication_tokens
  clone_url="https://git:${OVERLEAF_TOKEN}@git.overleaf.com/${PROJECT_ID}"
  if ! git clone "$clone_url" "$CLONE_DIR" 2>&1 \
        | sed "s|${OVERLEAF_TOKEN}|<REDACTED>|g"; then
    err "git clone failed.  Common causes:
       - wrong project ID
       - token revoked or expired
       - network firewall blocking git.overleaf.com"
  fi
  echo "  cloned into $CLONE_DIR"
fi

# ──────────────────────────────────────────────────────────────────────
# 4. Sanity check
# ──────────────────────────────────────────────────────────────────────

echo
if [[ "$SKIP_DOCTOR" -eq 1 ]]; then
  echo "[4/4] Skipping overleaf-mcp doctor (--skip-doctor)."
else
  echo "[4/4] Running overleaf-mcp doctor..."
  if ! "$OVERLEAF_MCP" doctor; then
    err "doctor reported failures.  Review the output above and fix before continuing."
  fi
fi

echo
echo "──────────────────────────────────────────────────────────────"
echo "Done.  Restart Claude Code (cmd-Q + relaunch) so the MCP server"
echo "picks up the new project, then in a new conversation try:"
echo
echo "    use overleaf list_projects"
echo
echo "You should see '${ALIAS}' in the response."
echo "──────────────────────────────────────────────────────────────"
