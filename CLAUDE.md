# CLAUDE.md

## Overview
`speechToText` transcribes interview recordings into Traditional Chinese text.
`transcribe.py` reads `./records/*.mp3` and writes `./output/*.txt` using
faster-whisper `large-v3` on CUDA (RTX 4070, WSL2; CUDA libs come from the
pip `nvidia-*-cu12` wheels, no system toolkit). Python 3.12, managed with `uv`
(`uv run transcribe.py`, `uv add <pkg>`; never edit `uv.lock` by hand).

The CI runner has no GPU. Anything that needs CUDA or real audio is a check
for James to run on his machine: list it in your report, don't fake it.

## Workflow
James (the owner) does not write code. A reviewer bot writes scoped issues,
you (Claude Code) plan and implement them, the reviewer bot reviews your
plan and result, and James merges. Stay inside the issue's scope, files, and
done criteria; if something outside scope looks necessary, ask instead.

## Agent rules
- Recordings and transcripts are private interview data. Never commit, print,
  or upload anything from `records/` or `output/`, and never add test fixtures
  derived from them. Use synthetic or generated audio for tests.
- Never read `.env*`, credentials, tokens, or shell history; never print or
  commit secrets.
- You can't push changes under `.github/workflows/` or `.claude/`. If one is
  needed, describe the exact diff in your report and the reviewer bot applies it.

## Reporting protocol
Every comment that ends a run must end with exactly one status marker on its
own line, after human-readable prose explaining it:

- `<!-- agent-status: plan-ready -->`: for every new issue, first post a plan
  (approach, files to touch, how you'll validate, risks/open questions) and
  stop. Don't implement until a comment says the plan is approved, unless the
  triggering comment explicitly says to skip planning.
- `<!-- agent-status: done -->`: implementation finished. State the branch,
  what changed, what you tested and how, and any checks James must run on real
  hardware.
- `<!-- agent-status: needs-decision -->`: you need a choice from a human;
  list the options and your recommendation.
- `<!-- agent-status: blocked -->`: you can't proceed (missing access, tool
  refused, failing environment); give the exact error.
