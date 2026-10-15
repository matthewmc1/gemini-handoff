import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gemini_cli_mcp as g  # noqa: E402

FAKE_GEMINI = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys
    mode = os.environ.get("FAKE_MODE", "ok")
    stdin = "" if sys.stdin.isatty() else sys.stdin.read()
    if mode == "fail":
        sys.stderr.write("auth expired"); sys.exit(41)
    if mode == "big":
        print(json.dumps({{"response": "x" * 30000}})); sys.exit(0)
    if mode == "noise":
        print("Loaded cached credentials."); print(json.dumps({{"response": "noisy ok"}})); sys.exit(0)
    if mode == "plain":
        print("not json"); sys.exit(0)
    if mode == "apierror":
        print(json.dumps({{"error": {{"message": "quota exceeded"}}}})); sys.exit(0)
    print(json.dumps({{"response": json.dumps({{"argv": sys.argv[1:], "stdin": stdin, "cwd": os.getcwd()}})}}))
    """
)


@pytest.fixture
def project(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "gemini"
    fake.write_text(FAKE_GEMINI.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setattr(g, "GEMINI_BIN", str(fake))

    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    (proj / "src" / "a.py").write_text("print('a')\n")
    (proj / "my dir").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=proj, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "."], cwd=proj, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], cwd=proj, check=True)
    (proj / "src" / "a.py").write_text("print('changed')\n")
    return proj


def call(project, **kw):
    kw.setdefault("task_type", "review")
    kw.setdefault("instructions", "look")
    return g.delegate_to_gemini(working_directory=str(project), **kw)


def seen(out):
    return json.loads(out)


def test_happy_path_builds_read_only_command(project):
    d = seen(call(project, target_paths=["src/"]))
    argv = d["argv"]
    assert "--approval-mode=plan" in argv
    assert argv[argv.index("-m") + 1] == "gemini-3.8-flash"
    assert argv[argv.index("--output-format") + 1] == "json"
    prompt = argv[argv.index("-p") + 1]
    assert "TASK TYPE: REVIEW" in prompt and "@src" in prompt
    assert Path(d["cwd"]).resolve() == project.resolve()
    assert d["stdin"] == ""


def test_shell_context_piped_to_stdin(project):
    d = seen(call(project, shell_context_cmd="git status --short && git diff"))
    assert "Output of `git status --short`" in d["stdin"]
    assert "M src/a.py" in d["stdin"]
    assert "+print('changed')" in d["stdin"]


def test_spaces_in_path_escaped(project):
    d = seen(call(project, target_paths=["my dir"]))
    assert "@my\\ dir" in d["argv"][d["argv"].index("-p") + 1]


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", "src/../../x"])
def test_path_escape_rejected(project, path):
    assert "error" in call(project, target_paths=[path])


def test_missing_path_rejected(project):
    assert "does not exist" in call(project, target_paths=["nope/"])


@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf /",
        "git push origin main",
        "git commit -am x",
        "git diff; rm x",
        "git diff | sh",
        "git diff > out.txt",
        "git diff $(whoami)",
        "git diff --output=pwn.txt",
        "git diff && rm -rf .",
        "git -c core.pager=sh diff",
        "git diff ||",
    ],
)
def test_unsafe_context_cmd_rejected(project, cmd):
    out = call(project, shell_context_cmd=cmd)
    assert out.startswith("delegate_to_gemini error"), out
    assert not (project / "pwn.txt").exists() and not (project / "out.txt").exists()


def test_shell_not_interpreted(project):
    # `$(...)` must reach git as a literal pathspec, never be executed.
    out = call(project, shell_context_cmd="git diff -- '$(touch pwned)'")
    assert not (project / "pwned").exists()
    assert not out.startswith("delegate_to_gemini error")


def test_failed_context_cmd_stops_chain(project):
    d = seen(call(project, shell_context_cmd="git show deadbeef && git status"))
    assert "exit code 128" in d["stdin"] and "git status" not in d["stdin"]


def test_bad_task_type_and_model(project):
    assert "task_type must be" in call(project, task_type="write")
    assert "invalid model" in call(project, model="--yolo")
    assert "instructions must not be empty" in call(project, instructions="  ")


def test_missing_binary(project, monkeypatch):
    monkeypatch.setattr(g, "GEMINI_BIN", "definitely-not-installed-xyz")
    assert "not found on PATH" in call(project)


def test_nonzero_exit_surfaced(project, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    out = call(project)
    assert "exited 41" in out and "auth expired" in out


def test_api_error_surfaced(project, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "apierror")
    assert "quota exceeded" in call(project)


def test_log_noise_before_json_tolerated(project, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "noise")
    assert call(project) == "noisy ok"


def test_non_json_falls_back_to_raw(project, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "plain")
    assert call(project).strip() == "not json"


def test_overflow_spills_to_file(project, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "big")
    out = call(project)
    assert len(out) < g.MAX_CLAUDE_CHARS
    assert ".gemini_last_handoff.md" in out
    assert (project / ".gemini_last_handoff.md").read_text() == "x" * 30000


def test_timeout(project, monkeypatch):
    monkeypatch.setattr(g, "GEMINI_TIMEOUT", 0.001)
    # fake gemini is a Python script; startup alone exceeds 1ms
    assert "timed out" in call(project)


def test_registered_tool_schema():
    import asyncio

    tools = asyncio.run(g.mcp.list_tools())
    assert [t.name for t in tools] == ["delegate_to_gemini"]
    props = tools[0].inputSchema["properties"]
    assert set(props) == {"task_type", "instructions", "target_paths", "shell_context_cmd", "model", "working_directory"}
    assert tools[0].inputSchema["required"] == ["task_type", "instructions"]


# --- cat / ls / grep / fzf -------------------------------------------------

needs_fzf = pytest.mark.skipif(shutil.which("fzf") is None, reason="fzf not installed")


def stdin_of(project, cmd):
    return seen(call(project, shell_context_cmd=cmd))["stdin"]


def test_cat_ls_grep_work(project):
    assert "print('changed')" in stdin_of(project, "cat src/a.py")
    assert "a.py" in stdin_of(project, "ls src")
    assert "src/a.py:1:print('changed')" in stdin_of(project, "grep -rn changed src")


def test_pipeline_and_chain(project):
    out = stdin_of(project, "ls -R src | grep a.py && cat src/a.py")
    assert "Output of `ls -R src | grep a.py`" in out and "Output of `cat src/a.py`" in out


def test_grep_no_match_stops_chain(project):
    out = stdin_of(project, "grep -r zzzz src && cat src/a.py")
    assert "exit code 1" in out and "Output of `cat src/a.py`" not in out


@needs_fzf
def test_fzf_filter_in_pipeline(project):
    out = stdin_of(project, "git ls-files | fzf --filter=a.py")
    assert "src/a.py" in out


@pytest.mark.parametrize(
    "cmd",
    [
        "fzf",                                   # interactive
        "fzf -f x --bind 'enter:execute(rm -rf .)'",
        "ls | fzf -f x --preview 'cat {}'",
        "ls | fzf --filter=x --history=h.txt",
        "cat /etc/passwd",
        "cat ../secret",
        "cat src/../../secret",
        "ls /",
        "grep -r root /etc",
        "grep -rf /etc/passwd src",
        "grep --file=/etc/passwd src/a.py",
        "grep -R x src",
        "grep x --include=../*",
        "cat link-out",                          # symlink to outside root
        "ls link-out",
        "cat src/a.py | sh",
        "cat src/a.py | tee out.txt",
        "find . -delete",
        "sed -i x src/a.py",
        "cat src/a.py > x",
        "cat -- /etc/passwd",
    ],
)
def test_unsafe_new_commands_rejected(project, tmp_path, cmd):
    (tmp_path / "secret").write_text("s3cret")
    (project / "link-out").symlink_to(tmp_path)
    out = call(project, shell_context_cmd=cmd)
    assert out.startswith("delegate_to_gemini error"), out
    assert not (project / "out.txt").exists() and not (project / "x").exists()


def test_symlink_inside_root_allowed(project):
    (project / "alias.py").symlink_to(project / "src" / "a.py")
    assert "changed" in stdin_of(project, "cat alias.py")


def test_tilde_is_not_expanded(project):
    out = stdin_of(project, "ls ~")  # literal file named "~" under the root, never $HOME
    assert "No such file" in out
