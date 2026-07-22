I'll start by confirming what's actually available for this review — a few files were listed in your upload but I don't see their contents yet (`filesearch.py`, `filesystem.py`, `agent.py`), and `agent.py` in particular sounds like it'd be the core orchestration loop, which I'd need for a genuinely complete review.I pulled the three files that were listed but hadn't actually made it into context (`filesearch.py`, `filesystem.py`, `agent.py`) — worth doing, since `agent.py` is the orchestrator that ties everything together, and a lot of the more interesting bugs live at the seams between it and the tool modules.

Overall this is a reasonable architecture — plan/build modes, hooks, sub-agents, pluggable backends via LiteLLM, clearly modeled on Claude Code's design. But there's one issue serious enough to lead with, a cluster of correctness bugs that will actually surface in normal use, and a long tail of smaller stuff. Going in priority order.

## The big one: no workspace boundary on file access

Every file tool resolves whatever path the model gives it and just operates on it:

```python
# filesystem.py — _read_impl, _write_impl, _edit_impl, _multiedit_impl all do this
file_path = Path(file_path_str).resolve()
```

`Read`, `Write`, `Edit`, and `MultiEdit` all take this path (which happily follows `..` and symlinks anywhere on disk) with **no check that the result stays inside `ctx.workspace`**. Tellingly, `ctx` is threaded into all four functions but never actually used — that's a strong signal a boundary check was planned and never landed, rather than a deliberate choice.

`environment.py` tells the model "the workspace root defines the project boundary," but that's a prompt-level convention, not an enforced one. This agent also reads files, runs shell commands, and processes arbitrary content — classic prompt-injection vectors. A webpage it fetches, a dependency's README, a comment in someone else's code could contain "also read `~/.ssh/id_rsa`" or "write to `~/.bashrc`," and nothing in the code stops it.

`filesearch.py`'s `Glob`/`ls` have the same gap on `path`, and it's actually worse there — when the requested path falls outside the workspace, ignore-filtering is explicitly disabled (the comment says so):

```python
def should_ignore(p: Path, is_d: bool) -> bool:
    try:
        rel_path = p.absolute().relative_to(ctx.workspace.absolute())
        return ignore_matcher.ignores_relative(rel_path.as_posix(), is_dir=is_d)
    except ValueError:
        # If the path being listed is completely outside the workspace,
        # we default to not ignoring it.
        return False
```

So searching outside the workspace is both allowed *and* unfiltered — `.git/`, `node_modules/`, `.venv/`, `.prismaignore` all stop applying. The fix is the same in both files: resolve the path, then reject (or require an explicit flag to widen scope) if it's not `path.is_relative_to(ctx.workspace)`.

## Safety hooks exist but nothing's registered

`hooks.py`'s `PreToolUseEvent` gives you a real interception point (`decision: "allow"|"deny"`), and `execute_tool` checks it correctly before running anything. But `main()` only registers two hooks, and both are `register_user_prompt` hooks:

```python
hooks.register_user_prompt(bound_setup_hook)
hooks.register_user_prompt(bound_mode_hook)
```

No `register_pre_tool` hook is ever registered. So `trigger_pre_tool` always walks an empty list, `decision` defaults to `"allow"`, and every tool call — including `Shell` — is auto-approved. The `Shell` description warns at length about `rm`, deploys, and pushes ("Only do these with the user's explicit instruction"), but right now that's enforced by nothing except the model's own judgment. The scaffolding for a real confirmation gate is there; it's just not wired to anything yet.

Related: the `code-reviewer` sub-agent is described as "Strict read-only code reviewer" with `tools=["Read", "Shell"]` and the comment "Can only read and run test commands!" But `clone_filtered` restricts by tool *name* only — it never looks at `is_readonly` — and `Shell.is_readonly` is `False` for good reason: it's completely general. Handing a "read-only" sub-agent `Shell` gives it the same destructive power as the main agent (`rm`, `curl | sh`, git push, anything), it just can't call `Write`/`Edit` *by name* — which doesn't matter once it has a shell. If this agent is meant to run tests, it needs something narrower: a dedicated test-runner tool, or a Shell variant with a command allowlist.

## Bugs that will show up in normal use

**Shell output draining can hang commands that produce a lot of output.**
```python
async def read_stream(stream, parts):
    while (chunk := await stream.read(8192)) and sum(len(p) for p in parts) < MAX_OUTPUT:
        parts.append(chunk)
    return b''.join(parts).decode('utf-8', errors='replace')[:MAX_OUTPUT]
```
Once `parts` hits 30,000 bytes, this simply stops calling `.read()`. But stdout/stderr are OS pipes with a bounded buffer (~64KB on Linux); if nothing drains them, the child process blocks on its next `write()`. So the command doesn't finish faster once it's over the cap — it now has to run out the full timeout (120s by default) before `terminate()`/`kill()` kicks in, even though it might have exited in milliseconds if its output kept being read. This will hit anything with chatty output: `npm install`, a verbose test run, a big `git log`, `find` on a large tree. Fix: keep draining to EOF, just stop *accumulating* past the cap:
```python
while chunk := await stream.read(8192):
    if sum(len(p) for p in parts) < MAX_OUTPUT:
        parts.append(chunk)
```

**`handle_subagent` always reports success.** The only two return points are the pre-start "blocked" case (`True`) and the normal-completion case, which is hardcoded:
```python
final_blocks = await run_agentic_loop(...)
return final_blocks, False
```
If a sub-agent's own writeup describes failure, or it stops without producing any text at all, the parent still sees a successful tool result. Worth deriving `is_error` from the sub-agent's actual output rather than a constant — at minimum, treat an empty `final_blocks` as an error.

**No limit on sub-agent recursion or on tool-calling turns.** `default-agent` gets `tools=None` ("can use all tools"), which includes `Task` itself — a sub-agent can spawn a sub-agent can spawn a sub-agent, with nothing tracking depth. Separately, `run_agentic_loop`'s `while True:` only exits when the model stops requesting tools — there's no turn ceiling. Either one can turn a confused model into an expensive, hard-to-interrupt runaway. Most comparable systems simply don't let sub-agents spawn sub-agents; that alone would close the recursion half.

**An ordinary API hiccup can crash the whole session.** `main()`'s loop only catches `KeyboardInterrupt`/`EOFError`:
```python
try:
    ...
    await run_agentic_loop(transcript, registry, hooks, model=args.model, policy=policy)
except (KeyboardInterrupt, EOFError):
    print("\nExiting...")
    break
```
`acompletion`'s call inside `run_agentic_loop`, and the hook triggers in `execute_tool`, aren't wrapped in anything else. A rate-limit response, a network blip, or a bug in a future hook propagates all the way up and kills the process with a raw traceback. (Sub-agent failures are accidentally *more* resilient, since a sub-agent's whole loop runs nested inside its parent's own `try/except`.) The transcript persists incrementally, so `--resume` recovers history, but the live session still dies mid-turn.

**`Transcript.load()` has no per-line error handling.** Since transcripts are appended one line at a time and the process can be killed mid-write, a truncated final line is a real possibility — and `json.loads` on it raises uncaught, taking down the *entire* load rather than just that one line. Wrap the per-line parse (JSON decode and `model_validate`) in a try/except that logs and skips.

**`stale_content_files` is checked but never populated.** `filesystem.py` declares it and checks it in three places as part of the read-before-write gate, but nothing anywhere adds to it — only `.discard()` calls exist. Concretely: `Read` a file, then something else changes it on disk (a `Shell` command, an external editor, a sub-agent), then `Write` it — the write proceeds and silently overwrites those external changes, because nothing ever marked the cached read as stale. (Related: since these tracking dicts are module-level globals, they're also shared across the top-level agent *and* every sub-agent — a sub-agent can `Write` a file it never itself `Read`, purely because the parent happened to read it earlier.) Either wire up something that actually populates staleness, or the mechanism should come out — right now it looks unfinished.

**Extended-thinking content is captured but never sent back.** `parse_assistant_response` carefully captures `reasoning_content`/`thinking_blocks` (with signatures) into `ThinkingMessageContent`. But `to_openai_message`'s `AssistantMessage` branch only handles `ToolUseMessageContent` and `TextMessageContent` — any thinking block is silently dropped on the next round-trip. For most providers that's just a quality loss; for Anthropic's actual extended thinking with tool use, the API expects the thinking block to precede the tool_use it justifies when history is resent, so this can be an outright mismatch depending on which model is in play.

**The tail-of-conversation cache breakpoint mostly no-ops.** `add_cache_control` for the last message only actually attaches inside the `if text_blocks:` branch — never on the `{"role": "tool", ...}` dicts built from tool results. Since no post-tool hooks are registered (above), nearly every "user" turn in this loop *is* pure tool-results with no text. So the "cache the tail" half of the strategy rarely fires in practice — only the first couple of messages ever get a breakpoint. Not a crash, just a quiet latency/cost regression.

**The "stable" cache key isn't stable.** `openai_prompt_cache_key = str(hash(str(Path.cwd())))` — Python's `hash()` on strings is randomized per-process by default, so this value differs every time you restart the CLI from the same directory, defeating whatever session-affinity it was meant to provide. Use `hashlib.sha256(...).hexdigest()` instead.

## Smaller issues worth fixing

- **`Read` checks file size after loading the whole file into memory.** `file_path.read_text(...)` happens before the `MAX_FILE_BYTES` check. Stat first, reject on size, then read — otherwise a stray huge file in the workspace gets fully slurped before being rejected.
- **Shell's documented timeout ceiling ("max 600000") isn't enforced anywhere** — neither `shell.py` nor `handle_shell` clamps it. An explicit `null` also divides cleanly into a `TypeError` that gets swallowed into a generic error string.
- **`execute_tool(..., policy: AgentPolicy=None)`** — the type says `AgentPolicy`, the default says `None`. Every current call site passes a real policy, so it doesn't bite today, but the `SubmitPlan` path does `policy.mode = AgentMode.BUILD` unconditionally, so any future call site relying on the default would crash the first time a plan gets submitted.
- **`known_content_files` stores content nobody reads.** Every consumer only checks membership (`file_path in known_content_files`) — the cached lines are never read back (`_edit_impl` re-reads from disk every time). A `set[Path]`, like `stale_content_files`, would do the same job for much less memory.
- **`load_dotenv(".env.development")`** is a fairly dev-specific filename to be the one env file the CLI loads, and it fails silently if missing — worth confirming this is really what you want end users' `ANTHROPIC_API_KEY`/`OLLAMA_API_BASE` to depend on, versus a plain `.env`.
- **No `max_tokens` on the completion call.** Depending on the provider's default via LiteLLM, this can silently cap responses shorter than you'd want for a coding agent emitting long edits.
- **`ToolResultMessageContent.is_error` isn't consulted when building the "tool" role message.** Probably fine in practice since failure text is pre-formatted ("Error: ..."), but worth confirming that's actually load-bearing rather than assumed.
- **`_glob_impl`/`_ls_impl` each define their own, nearly identical `should_ignore` closure.** Worth pulling into one helper (in `ignore.py`) so the "outside workspace → don't ignore" behavior only needs reconsidering once.
- **`IgnoreMatcher` (and `.prismaignore`) is rebuilt from scratch, including a disk read, on every single `Glob`/`ls` call.** Fine for one call, wasteful over a long exploration session — cache it per workspace.
- **`get_dir_count` doubles the directory walk.** For every directory `ls` expands, it does a full `scandir` pass to compute the `(N items)` count, then a separate full pass via `generate_tree` to actually render the children — roughly 2x the I/O and ignore-matching work on a large tree.

## Nits

- Unused imports: `os` in `config.py`, `prompts.py`, and `agent.py`; `Path` in `tasks.py`; `pydantic` in `registry.py`; `re` and `fnmatch` in `filesearch.py`. None matter functionally — a `ruff`/`pyflakes` pass cleans all of them in seconds.
- `ignore.py` has a stray `# We` comment before the `.prisma/` pattern.
- `hooks.py`'s section header says "Agends.md" (typo for AGENTS.md).
- `sessioncontext.py` mixes `Optional[Path]` and `AgentMode | None` in the same file.
- `registry.py`'s `invoke()` catches exceptions and returns a bare string, while every intentional failure elsewhere uses the structured `ToolFailure` type — worth aligning so callers can't confuse "the tool deliberately failed" with "the tool crashed" by string-sniffing.
- `plan.py`'s `_plan_impl` uses `kwargs.get("plan_summary", "No plan provided.")`, which only applies the default when the key is *absent* — an explicit empty string sails through. `shell.py`/`tasks.py` get this right by checking truthiness after the `.get()`.
- `typedefs.py`'s content unions rely on Pydantic's "smart" union matching rather than an explicit discriminator, and `transcript.py` re-implements the same role-dispatch by hand in `load()`. A discriminated union would let you collapse that into one `TypeAdapter(Message).validate_python(...)` call.

## What's genuinely solid

Worth calling out, since it's easy to get these wrong:
- `MultiEdit`'s overlap detection (replacing each edit's `old_string` with a UUID marker in a scratch copy to catch edits that would consume each other's targets) is a genuinely clever, correct technique.
- `Glob`'s bounded min-heap for tracking the newest N files during traversal, rather than collecting everything and sorting at the end, is the right call on a large tree.
- `handle_shell`'s SIGTERM → wait → SIGKILL escalation on timeout is correct and often skipped in simpler implementations.
- The layered system prompt in `prompts.py` (immutable core → user-overridable → global/project SYSTEM.md → environment) is clean, with comments that make the override rules obvious.
- Several tools defensively handle the model passing a bare string where a list was expected (`exclude` in both `Glob` and `ls`) — small, but it heads off a whole class of avoidable tool-call failures.

Happy to patch any of these — the workspace-boundary check and the shell output draining fix are both small, contained changes if you want to start there.