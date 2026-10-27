# CloakOS notes

- Bump the version on every PR that ships: `pyproject.toml` and `cloakos/__init__.py` (`__version__`) must match.
- When trained policies change, bump Nous (`NOUS`/`NOUS_VERSION` in `cloakos/__init__.py`) and run
  `python scripts/nous_manifest.py <nous version> <min app version>`. The Update Nous button reads that manifest.
