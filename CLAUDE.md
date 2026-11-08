# CloakOS notes

The repo root is the new CloakOS. The Vector robot brain lives in `vector-edition/`
(CloakOS Vector Edition), kept out of the default download. For anything in there, read
`vector-edition/AGENTS.md` first.

- Root `install.sh`/`uninstall.sh` are shims to `vector-edition/`; installed Macs update through them. Keep them until those installs have moved over.
- Bump the version on every Vector Edition PR that ships: `vector-edition/pyproject.toml` and `vector-edition/cloakos/__init__.py` (`__version__`) must match.
- When trained policies change, bump Nous (`NOUS`/`NOUS_VERSION`) and run
  `python scripts/nous_manifest.py <nous version> <min app version>` from `vector-edition/`.
- When work is done: open a PR, squash-merge it yourself, and tell the user to run `cloakos update`.
- Keep test runs short (targeted `pytest -k`); the user doesn't want long tests or training runs in the container.
- Talk to the user briefly and plainly; outline changes, don't explain code.
