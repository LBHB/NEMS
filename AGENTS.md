# AGENTS.md

This file provides guidance to AI agents working in this repository. See CLAUDE.md for full project context.

## Conventions

- **Agent code edits:** Wrap any new or modified block of code introduced by an agent with start/end comment tags:
  ```python
  # [AGENT EDIT START | agent: <agent> | user: <user> | reason: <description> | date: YYYY-MM-DD]
  ... new/edited code ...
  # [AGENT EDIT END]
  ```
  This makes agent-introduced changes easy to search for (e.g., `grep -r "AGENT EDIT"`), audit, and revert.
