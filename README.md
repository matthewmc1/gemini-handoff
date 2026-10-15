# claude-gemini-tool

MCP bridge that lets Claude Code hand read-heavy work (task synthesis, code review,
post-implementation verification) to the Gemini CLI running in read-only
`--approval-mode=plan`. All edits stay in Claude Code. See the PRD for background.

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
gemini -m gemini-3.8-flash --approval-mode=plan --output-format json -p "PONG"   # smoke test auth + model
claude mcp add gemini-worker -- "$PWD/.venv/bin/python" "$PWD/scripts/gemini_cli_mcp.py"   # or use the checked-in .mcp.json
```

Then confirm with `/mcp` in Claude Code. Copy `docs/CLAUDE.md.snippet` into your
project's `CLAUDE.md` and `.claude/settings.json` into the project to pre-approve the tool.

`mcp` is pinned `<2`: v2 renamed `FastMCP`.

## Tool: `delegate_to_gemini`

| arg | notes |
|---|---|
| `task_type` | `synthesize` \| `review` \| `verify` |
| `instructions` | what Gemini should analyze |
| `target_paths` | files/dirs relative to `working_directory`; converted to `@path`. Must exist and stay inside the root |
| `shell_context_cmd` | read-only commands (see below); run without a shell, output piped on stdin |
| `model` | default `gemini-3.8-flash` |
| `working_directory` | project root (default: server cwd) |

### `shell_context_cmd` allowlist

Programs: read-only `git` subcommands (`diff status log show ls-files blame rev-parse diff-tree shortlog grep merge-base describe`), `cat`, `ls`, `grep`, `fzf`.
Operators: `&&` (stop at first failing pipeline) and `|`. Anything else (`;`, `||`, `>`, `$()`, globs) is rejected.

Guards:
- `cat`/`ls`/`grep`/`fzf` path arguments are resolved (following symlinks) and must stay inside `working_directory`. `~` is never expanded.
- `grep`: `-f/--file` and `-R` (follows symlinks) rejected.
- `fzf`: must be non-interactive (`--filter`/`-f`); `--bind`, `--preview*`, `--with-shell`, `--listen`, `--history`, `--tmux`, `--walker-root` rejected. fzf only works at the end of a pipeline (it has no stdin otherwise), e.g. `git ls-files | fzf --filter=auth`.
- Commands get `/dev/null` as stdin when they start a pipeline; the MCP server's own stdin is never inherited.

Output over 25,000 chars is written to `.gemini_last_handoff.md` and a 4,000-char preview is returned.

Env overrides: `GEMINI_BIN`, `GEMINI_HANDOFF_MODEL`, `GEMINI_HANDOFF_TIMEOUT`.

## Differences from the PRD sketch

- `shell_context_cmd` is not run via `shell=True`. The PRD's arbitrary shell contradicts "strictly read-only", so only an allowlist (`git` read-only subcommands, `cat`, `ls`, `grep`, `fzf`) is accepted, with `&&`/`|` and path confinement.
- `target_paths` are validated (exist, inside root, spaces escaped for `@path`).
- `model` is validated so it can't inject CLI flags (e.g. `--yolo`).
- Errors (non-zero exit, timeout, Gemini API error JSON, missing binary) come back as clear tool text; the server never crashes.
- No mutable default argument; tolerates log noise before Gemini's JSON.

## Tests

```sh
.venv/bin/python -m pytest -q   # uses a fake `gemini`; no network/auth needed
```
