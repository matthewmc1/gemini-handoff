"""MCP bridge that lets Claude Code hand read-heavy work to the Gemini CLI.

Exposes one tool, ``delegate_to_gemini``. Gemini runs with
``--approval-mode=plan`` (read-only); all edits stay in Claude Code.

Configuration (environment variables, all optional):
    GEMINI_BIN                 gemini executable           (default: gemini)
    GEMINI_HANDOFF_MODEL       default model               (default: gemini-3.8-flash)
    GEMINI_HANDOFF_TIMEOUT     gemini timeout, seconds     (default: 300)
"""

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gemini-enterprise-worker")

DEFAULT_MODEL = os.environ.get("GEMINI_HANDOFF_MODEL", "gemini-3.8-flash")
GEMINI_BIN = os.environ.get("GEMINI_BIN", "gemini")
GEMINI_TIMEOUT = int(os.environ.get("GEMINI_HANDOFF_TIMEOUT", "300"))
CONTEXT_CMD_TIMEOUT = 120
MAX_CLAUDE_CHARS = 25000  # Keep safely below Claude Code's ~30k char tool truncation cap
PREVIEW_CHARS = 4000
MAX_CONTEXT_CHARS = 2_000_000  # Cap on piped shell output (Gemini has a 1M-token window)
OVERFLOW_FILE = ".gemini_last_handoff.md"

TASK_TYPES = ("synthesize", "review", "verify")

# Read-only git subcommands permitted in shell_context_cmd.
ALLOWED_GIT_SUBCOMMANDS = frozenset(
    {"diff", "status", "log", "show", "ls-files", "blame", "rev-parse",
     "diff-tree", "shortlog", "grep", "merge-base", "describe"}
)
# Flags that make an otherwise read-only git command write to disk or run code.
FORBIDDEN_GIT_FLAGS = ("--output", "--ext-diff", "--textconv", "--exec", "-O", "--open-files-in-pager")

# Other read-only commands permitted in shell_context_cmd (git is handled separately above).
ALLOWED_COMMANDS = frozenset({"git", "cat", "ls", "grep", "fzf"})
# Flag prefixes rejected per command: file-reading (-f, -R follows symlinks out of the root)
# for grep; command execution / file writes / interactive UI for fzf.
FORBIDDEN_FLAG_PREFIXES = {
    "grep": ("--file", "--dereference-recursive"),
    "fzf": ("--bind", "--preview", "--with-shell", "--listen", "--history", "--tmux", "--walker-root"),
}
# Single-dash flag letters rejected inside short-flag clusters (e.g. grep -rf).
FORBIDDEN_SHORT_LETTERS = {"grep": "fR"}

_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")

SYSTEM_CONTRACT = """
You are an expert read-only sub-agent invoked by an automated orchestrator (Claude Code).
Follow this strict output contract:
1. Be dense and actionable. Omit conversational filler, greetings, or sign-offs.
2. Always cite exact relative file paths and line numbers (e.g., `src/services/user.ts:114`).
3. For code reviews (REVIEW) and verification (VERIFY): group findings by severity (CRITICAL, WARNING, NIT), verify requirement coverage, and provide concise replacement code blocks.
4. For complex task synthesis (SYNTHESIZE): return a numbered execution plan specifying exact files to modify, dependency order, and architectural edge cases.
""".strip()


class HandoffError(Exception):
    """A user-facing validation or execution failure."""


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def resolve_root(working_directory: str) -> Path:
    root = Path(working_directory).expanduser().resolve()
    if not root.is_dir():
        raise HandoffError(f"working_directory is not a directory: {root}")
    return root


def resolve_target_paths(target_paths: list[str], root: Path) -> list[str]:
    """Return @-safe relative paths, rejecting anything outside ``root``."""
    refs = []
    for raw in target_paths:
        if not raw or any(c in raw for c in "\n\r\0"):
            raise HandoffError(f"Invalid target path: {raw!r}")
        resolved = (root / raw).resolve()
        if resolved != root and root not in resolved.parents:
            raise HandoffError(f"target path escapes working_directory: {raw}")
        if not resolved.exists():
            raise HandoffError(f"target path does not exist: {raw}")
        rel = os.path.relpath(resolved, root)
        # Gemini's @path syntax treats unescaped whitespace as the end of the path.
        refs.append(rel.replace(" ", "\\ "))
    return refs


def _check_in_root(value: str, root: Path, original: str) -> None:
    """Reject ``value`` if it names a location outside ``root`` (checked after resolving symlinks)."""
    if not value:
        return
    candidate = root / value  # an absolute ``value`` replaces ``root`` here
    suspicious = value.startswith(("/", "~")) or ".." in Path(value).parts or candidate.exists()
    if not suspicious:
        return  # a pattern / number / flag value, not a path
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise HandoffError(f"shell_context_cmd: argument {original!r} points outside working_directory")


def _validate_command(argv: list[str], root: Path) -> None:
    prog = argv[0]
    if prog not in ALLOWED_COMMANDS:
        raise HandoffError(
            f"shell_context_cmd: command {prog!r} not allowed (allowed: {', '.join(sorted(ALLOWED_COMMANDS))})"
        )
    if prog == "git":
        if len(argv) < 2 or argv[1] not in ALLOWED_GIT_SUBCOMMANDS:
            sub = argv[1] if len(argv) > 1 else ""
            raise HandoffError(
                f"shell_context_cmd: git subcommand {sub!r} not allowed "
                f"(allowed: {', '.join(sorted(ALLOWED_GIT_SUBCOMMANDS))})"
            )
    args = argv[2:] if prog == "git" else argv[1:]
    for arg in args:
        if arg == "--" and prog == "git":
            break  # everything after is a pathspec
        if prog == "git" and any(arg == f or arg.startswith(f + "=") for f in FORBIDDEN_GIT_FLAGS):
            raise HandoffError(f"shell_context_cmd: flag {arg!r} not allowed")
        if arg.startswith("--") and arg.startswith(FORBIDDEN_FLAG_PREFIXES.get(prog, ())):
            raise HandoffError(f"shell_context_cmd: flag {arg!r} not allowed for {prog}")
        if arg.startswith("-") and not arg.startswith("--") and any(
            c in arg[1:] for c in FORBIDDEN_SHORT_LETTERS.get(prog, "")
        ):
            raise HandoffError(f"shell_context_cmd: flag {arg!r} not allowed for {prog}")
        if prog != "git":  # git reads only the repo; the others take arbitrary paths
            if not arg.startswith("-"):
                _check_in_root(arg, root, arg)
            elif "=" in arg:
                _check_in_root(arg.split("=", 1)[1], root, arg)
    if prog == "fzf" and not any(
        a == "--filter" or a.startswith("--filter=") or (a.startswith("-") and not a.startswith("--") and "f" in a)
        for a in args
    ):
        raise HandoffError("shell_context_cmd: fzf must be non-interactive; pass --filter=QUERY (-f QUERY)")


def parse_context_cmd(cmd: str, root: Path) -> list[list[list[str]]]:
    """Parse ``cmd`` into ``&&``-joined chains of ``|``-joined argv pipelines, enforcing the allowlist.

    Returns ``chains[chain_index][stage_index] -> argv``.
    """
    lexer = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError as e:
        raise HandoffError(f"shell_context_cmd could not be parsed: {e}")

    chains: list[list[list[str]]] = [[[]]]
    for tok in tokens:
        if tok == "&&":
            chains.append([[]])
        elif tok == "|":
            chains[-1].append([])
        elif set(tok) <= set("&|;<>()"):
            raise HandoffError(
                f"shell_context_cmd: operator {tok!r} not allowed (only '&&' and '|' may be used)"
            )
        else:
            chains[-1][-1].append(tok)

    for pipeline in chains:
        for argv in pipeline:
            if not argv:
                raise HandoffError("shell_context_cmd contains an empty command")
            _validate_command(argv, root)
    return chains


def _run_pipeline(pipeline: list[list[str]], cwd: Path, env: dict) -> tuple[int, str, str]:
    """Run stages sequentially, feeding each stage's stdout to the next; no shell involved."""
    data = None
    stderrs = []
    proc = None
    for argv in pipeline:
        # Never inherit stdin: it is the MCP JSON-RPC stream.
        extra = {"input": data} if data is not None else {"stdin": subprocess.DEVNULL}
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, errors="replace",
            env=env, timeout=CONTEXT_CMD_TIMEOUT, **extra,
        )
        data = proc.stdout
        if proc.stderr.strip():
            stderrs.append(proc.stderr)
    return proc.returncode, data, "".join(stderrs)


def run_context_cmd(cmd: str, cwd: Path) -> str:
    """Run the validated command chain (no shell) and format output for Gemini's stdin."""
    chains = parse_context_cmd(cmd, cwd)
    env = {k: v for k, v in os.environ.items() if k != "GIT_EXTERNAL_DIFF"}
    env.update(GIT_PAGER="cat", GIT_OPTIONAL_LOCKS="0")
    sections = []
    for pipeline in chains:
        shown = " | ".join(shlex.join(a) for a in pipeline)
        try:
            code, out, err = _run_pipeline(pipeline, cwd, env)
        except subprocess.TimeoutExpired:
            sections.append(f"`{shown}` timed out after {CONTEXT_CMD_TIMEOUT}s")
            break
        except OSError as e:
            sections.append(f"`{shown}` failed to start: {e}")
            break
        body = out + (f"\n[stderr]\n{err}" if err.strip() else "")
        sections.append(f"Output of `{shown}` (exit code {code}):\n{body}")
        if code != 0:  # `&&` semantics: stop at first failing pipeline
            break
    payload = "\n\n".join(sections)
    if len(payload) > MAX_CONTEXT_CHARS:
        payload = payload[:MAX_CONTEXT_CHARS] + f"\n[context truncated at {MAX_CONTEXT_CHARS} chars]"
    return payload


# --------------------------------------------------------------------------
# Core handoff
# --------------------------------------------------------------------------

def build_prompt(task_type: str, instructions: str, at_refs: list[str]) -> str:
    prompt = (
        f"{SYSTEM_CONTRACT}\n\n"
        f"### TASK TYPE: {task_type.upper()}\n"
        f"{instructions}\n"
    )
    if at_refs:
        prompt += "\n### TARGET FILES/DIRS\n" + " ".join(f"@{r}" for r in at_refs) + "\n"
    return prompt


def extract_response(stdout: str) -> str:
    """Pull `.response` out of Gemini's JSON output, tolerating leading log noise."""
    data = None
    for candidate in (stdout, stdout[stdout.find("{"):] if "{" in stdout else ""):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        return stdout
    if data.get("response"):
        return data["response"]
    if data.get("error"):
        err = data["error"]
        msg = err.get("message", err) if isinstance(err, dict) else err
        raise HandoffError(f"Gemini CLI reported an error: {msg}")
    return stdout


def guard_payload(text: str, root: Path) -> str:
    """Keep the tool result under Claude Code's truncation cap, spilling to disk if needed."""
    if len(text) <= MAX_CLAUDE_CHARS:
        return text
    (root / OVERFLOW_FILE).write_text(text, encoding="utf-8")
    return (
        f"[Output was {len(text)} chars (> {MAX_CLAUDE_CHARS}); full text saved to `{OVERFLOW_FILE}`]\n\n"
        f"Preview (first {PREVIEW_CHARS} chars):\n{text[:PREVIEW_CHARS]}\n\n"
        f"Use Grep or Read with line offsets on `{OVERFLOW_FILE}` for the rest."
    )


def handoff(
    task_type: str,
    instructions: str,
    target_paths: list[str] | None = None,
    shell_context_cmd: str = "",
    model: str = DEFAULT_MODEL,
    working_directory: str = ".",
) -> str:
    task_type = task_type.strip().lower()
    if task_type not in TASK_TYPES:
        raise HandoffError(f"task_type must be one of {', '.join(TASK_TYPES)}; got {task_type!r}")
    if not instructions.strip():
        raise HandoffError("instructions must not be empty")
    if not _MODEL_RE.match(model):
        raise HandoffError(f"invalid model name: {model!r}")
    gemini = shutil.which(GEMINI_BIN)
    if gemini is None:
        raise HandoffError(f"Gemini CLI executable {GEMINI_BIN!r} not found on PATH (set GEMINI_BIN)")

    root = resolve_root(working_directory)
    at_refs = resolve_target_paths(target_paths or [], root)
    stdin_payload = run_context_cmd(shell_context_cmd, root) if shell_context_cmd.strip() else None

    cmd = [
        gemini, "-m", model,
        "--approval-mode=plan",
        "--output-format", "json",
        "-p", build_prompt(task_type, instructions, at_refs),
    ]
    try:
        result = subprocess.run(
            cmd, input=stdin_payload, cwd=root,
            capture_output=True, text=True, timeout=GEMINI_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise HandoffError(f"Gemini CLI timed out after {GEMINI_TIMEOUT}s")
    except OSError as e:
        raise HandoffError(f"Failed to start Gemini CLI: {e}")

    if result.returncode != 0:
        raise HandoffError(f"Gemini CLI exited {result.returncode}:\n{result.stderr}\n{result.stdout}".strip())

    return guard_payload(extract_response(result.stdout), root)


@mcp.tool()
def delegate_to_gemini(
    task_type: str,
    instructions: str,
    target_paths: list[str] | None = None,
    shell_context_cmd: str = "",
    model: str = DEFAULT_MODEL,
    working_directory: str = ".",
) -> str:
    """
    Delegate read-only code review, verification, or complex multi-file synthesis to Gemini CLI.

    Gemini reads files from disk itself, so do NOT Read the target files into Claude first.

    Args:
        task_type: One of 'synthesize', 'review', or 'verify'.
        instructions: What Gemini should analyze, review, or verify.
        target_paths: Files or directories relative to working_directory (e.g. ['src/auth/', 'src/api/']).
        shell_context_cmd: Optional read-only command(s) whose output is piped to Gemini, e.g.
            'git diff main...HEAD', 'git status --short && git diff', or
            'git ls-files | fzf --filter=auth'. Allowed programs: read-only git subcommands,
            cat, ls, grep, and fzf (non-interactive: --filter required). Chain with '&&' or
            pipe with '|'; no redirects, globs, or paths outside working_directory.
        model: Gemini model; defaults to 'gemini-3.8-flash'.
        working_directory: Project root directory.
    """
    try:
        return handoff(task_type, instructions, target_paths, shell_context_cmd, model, working_directory)
    except HandoffError as e:
        return f"delegate_to_gemini error: {e}"
    except Exception as e:  # never let the MCP server crash on a handoff
        return f"delegate_to_gemini unexpected error ({type(e).__name__}): {e}"


if __name__ == "__main__":
    mcp.run()
