# CLAUDE.md — ES-rlvr

This file provides guidance for AI assistants (e.g. Claude Code) working in this repository.

---

## Repository Overview

**Name:** ES-rlvr
**Owner:** joejajo
**Remote:** `http://local_proxy@127.0.0.1:41211/git/joejajo/ES-rlvr`

This repository appears to be in its initial state (no committed source code yet). Update this file once the project structure is established.

---

## Git Workflow

### Branching Convention

- Feature branches: `feature/<short-description>`
- Bug fixes: `fix/<short-description>`
- AI/automated work: `claude/<task-id>` (e.g. `claude/claude-md-mmjmsyal1uqqzz0l-l304z`)

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`

Examples:
```
feat(training): add PPO rollout buffer
fix(env): correct reward normalization bug
docs: update CLAUDE.md with project structure
```

### Push Instructions

Always use:
```bash
git push -u origin <branch-name>
```

Branch names for AI-generated work must start with `claude/` and end with the session ID.

---

## Development Setup

> This section should be updated once the project is initialized.

Expected setup steps (fill in when project is scaffolded):

```bash
# Clone
git clone <repo-url>
cd ES-rlvr

# Install dependencies (update as appropriate)
# pip install -e .       # Python project
# npm install            # Node project
# make install           # Makefile-driven project
```

---

## Project Structure

> To be filled in once source code is committed.

Expected layout for an RL/training project:

```
ES-rlvr/
├── CLAUDE.md           # This file
├── README.md           # Project description
├── pyproject.toml      # Python packaging & dependencies
├── src/                # Main source code
│   └── es_rlvr/
│       ├── __init__.py
│       ├── trainer.py
│       ├── models/
│       └── envs/
├── scripts/            # Training/evaluation entry points
├── tests/              # Unit and integration tests
├── configs/            # Hyperparameter / experiment configs
└── .github/
    └── workflows/      # CI/CD pipelines
```

---

## Testing

> Update once tests are established.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run a specific test file
pytest tests/test_trainer.py
```

Conventions:
- Tests live in `tests/` mirroring `src/` structure.
- Use `pytest` as the test runner.
- All new code should include corresponding tests.

---

## Code Style

> Update based on project linting configuration.

- Python: follow [PEP 8](https://peps.python.org/pep-0008/) with `ruff` or `black` for formatting.
- Imports: stdlib → third-party → local, separated by blank lines.
- Type hints encouraged for all public functions.
- Docstrings: NumPy or Google style.

Lint / format commands (update as configured):
```bash
ruff check .
ruff format .
mypy src/
```

---

## AI Assistant Guidelines

When working in this repository:

1. **Read before writing.** Always read existing files before modifying them.
2. **Minimal changes.** Only change what is necessary for the task at hand.
3. **No speculative abstractions.** Don't add helpers or utilities beyond what is immediately needed.
4. **No unused imports or variables.**
5. **No security vulnerabilities.** Avoid command injection, SQL injection, unsafe deserialization, etc.
6. **Commit incrementally.** Prefer small, focused commits over large monolithic ones.
7. **Update this file.** When the project structure changes significantly, keep CLAUDE.md current.
8. **Do not push to `main` or `master` directly.** Always use a feature branch and PR.

---

## Key Contacts / References

- Repository owner: `joejajo`
- Update this section with links to design docs, issue trackers, or runbooks as they are created.
