# CloakOS notes

Read `AGENTS.md` first: the full handoff (what this is, the architecture, how the user works,
the training pipeline, the current state and next steps). The short rules:

- Bump the version on every PR that ships: `pyproject.toml` and `cloakos/__init__.py` (`__version__`) must match.
- When trained policies change, bump Nous (`NOUS`/`NOUS_VERSION` in `cloakos/__init__.py`) and run
  `python scripts/nous_manifest.py <nous version> <min app version>`. The Update Nous button reads that manifest.
- When work is done: open a PR, squash-merge it yourself, and tell the user to run `cloakos update`.
- Keep test runs short (targeted `pytest -k`); the user doesn't want long tests or training runs in the container.
- Talk to the user briefly and plainly; outline changes, don't explain code.
