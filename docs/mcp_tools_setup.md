# MCP Tools Setup

This project is configured for three Codex MCP servers:

- `context7`: current library docs lookup through `@upstash/context7-mcp`
- `github`: local GitHub MCP server, read-only, authenticated through `gh auth token`
- `playwright`: browser automation and screenshot support through `@playwright/mcp`

## Current Local Runtime

Node is installed locally under:

```text
.tools/node-v22.23.1-linux-x64/
```

It is intentionally not added to the global shell PATH. Codex MCP entries use the full local path and set PATH only for the MCP process.

GitHub MCP server is installed locally under:

```text
.tools/github-mcp-server-v1.5.0/
```

The `.tools/` directory is ignored by git.

## Status Commands

```bash
codex mcp list
codex mcp get context7
codex mcp get github
codex mcp get playwright
```

## Configured Servers

### Context7

```bash
codex mcp add context7 \
  --env PATH=/home/ruan/research/Hackthon/.tools/bin:/usr/bin:/bin \
  -- /home/ruan/research/Hackthon/.tools/bin/npx -y @upstash/context7-mcp@latest
```

Use this for current docs and API examples for libraries such as LeRobot, OpenVLA-style repos, ManiSkill, Genesis, Open3D, gsplat, and Nerfstudio.

### GitHub

```bash
codex mcp add github -- /home/ruan/research/Hackthon/scripts/run_github_mcp.sh
```

The wrapper script:

- Uses the installed GitHub MCP server binary
- Reads the token from `gh auth token`
- Does not write the token into Codex config
- Runs the server in read-only mode
- Enables default repo, issue, pull request, and Actions toolsets

Check GitHub auth:

```bash
gh auth status
```

### Playwright

```bash
codex mcp add playwright \
  --env PATH=/home/ruan/research/Hackthon/.tools/bin:/usr/bin:/bin \
  -- /home/ruan/research/Hackthon/.tools/bin/npx -y @playwright/mcp@latest \
  --browser chrome \
  --headless \
  --output-dir /home/ruan/research/Hackthon/.tools/playwright-output
```

The machine already has:

```text
/usr/bin/google-chrome
```

Playwright outputs are saved under:

```text
.tools/playwright-output/
```

## Rebuild Notes

If `.tools/` is deleted, reinstall lightweight local tools only. Do not use `sudo`.

Node source:

```text
https://nodejs.org/dist/latest-v22.x/node-v22.23.1-linux-x64.tar.xz
```

GitHub MCP source:

```text
https://github.com/github/github-mcp-server/releases/tag/v1.5.0
```

Large model and dataset downloads are still out of scope unless explicitly approved.

