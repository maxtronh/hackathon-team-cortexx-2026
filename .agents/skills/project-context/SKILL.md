---
name: project-context
description: >-
  Use this skill when starting a new conversation or when the agent needs to
  understand the current project state. Provides a quick overview of the
  codebase structure and tech stack.
---

# Project Context Loader

## Steps

1. Read `package.json` to determine dependencies and scripts.
2. List the project structure (excluding `node_modules/` and `dist/`).
3. Read `GEMINI.md` for team rules.
4. Summarize the current state to the user before proceeding.
