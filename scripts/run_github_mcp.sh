#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT_DIR/.tools/github-mcp-server-v1.5.0/github-mcp-server"

if [[ ! -x "$SERVER" ]]; then
  echo "GitHub MCP server binary is missing: $SERVER" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI 'gh' is required for token lookup." >&2
  exit 1
fi

export GITHUB_PERSONAL_ACCESS_TOKEN="${GITHUB_PERSONAL_ACCESS_TOKEN:-$(gh auth token)}"

exec "$SERVER" stdio --read-only --toolsets=default,repos,issues,pull_requests,actions
