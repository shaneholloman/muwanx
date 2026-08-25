---
description: Review the comments in a given range and cut them down to only what is needed
---

Target: $ARGUMENTS (nothing given → the current diff)

Using the `ponytail:ponytail-review` skill, review every comment in the target range: delete the ones that are not needed, and rewrite the ones that are so they carry only the information a reader actually needs.

- Scope: docstrings and comment lines only. Do not change code.
- If a comment fits on one line, make it one line. Multiple lines are fine when a developer genuinely needs that much context, but strip anything that is not carrying its weight.
- If the range is a PR, branch, or commit, first get its diff with `gh` / `git` and review only the comments inside that diff.
