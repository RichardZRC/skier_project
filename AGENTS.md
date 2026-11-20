# AGENTS.md

Instructions for AI coding assistants working on this repository: a CS230 project by Ruichao Zhang on skier
tracking and 2D body, ski and pole landmarks. The final report is due Fri 2026-11-20.

Read these first, in order. They were written for Claude Code, but they apply to any assistant: read "Claude" as
"the AI assistant".
1. `CLAUDE.md`: the working rules (data discipline, when to commit, the environment, code conventions, known
   gotchas) and the module map.
2. `PROJECT_NOTES.md`, starting with the Handoff subsection "2026-11-19: the final report, its state and how to
   continue". It has the current state, the to-do list, how Ruichao and the assistant split the report work, the
   course AI policy, how to build and measure the report, and the key numbers.
3. `docs/decisions.md`: decisions D1–D30. Append only.

The three rules that matter most right now:
- **The report prose is Ruichao's.** The assistant proofreads, checks numbers and typesets, and lists every change.
  It doesn't write the analysis, interpretation or conclusions (course AI policy, quoted in the handoff).
- **Commit and push only when Ruichao asks.** Never commit dataset images, labels, checkpoints, or Ruichao's
  untracked files in the repo root.
- **No more test-set evaluations.** Each test stage has run its one time, and the scripts refuse to run again.
