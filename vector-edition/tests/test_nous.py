import hashlib
import json

import cloakos
from cloakos import nous
from cloakos.rl.skills import PRETRAINED


def test_versions_match():
    import re
    assert re.search(rf'^version = "{cloakos.__version__}"', open("pyproject.toml").read(), re.M)


def test_manifest_matches_shipped_files():
    m = json.loads((PRETRAINED / "manifest.json").read_text())
    assert m["nous"] == cloakos.NOUS_VERSION
    for skill, files in m["skills"].items():
        for name, digest in files.items():
            assert hashlib.sha256((PRETRAINED / skill / name).read_bytes()).hexdigest() == digest


def test_newer_download_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOAK_HOME", str(tmp_path))
    assert nous.active_dir() == PRETRAINED
    (tmp_path / "nous").mkdir()
    (tmp_path / "nous" / "manifest.json").write_text(json.dumps({"nous": "9.0.0", "epoch": 1, "min_app": "0.1.0", "skills": {}}))
    assert nous.active_dir() == tmp_path / "nous"
    (tmp_path / "nous" / "manifest.json").write_text(json.dumps({"nous": "9.0.0", "epoch": 1, "min_app": "99.0.0", "skills": {}}))
    assert nous.active_dir() == PRETRAINED          # needs a newer app than this one
    (tmp_path / "nous" / "manifest.json").write_text(json.dumps({"nous": "0.0.2", "min_app": "0.1.0", "skills": {}}))
    assert nous.active_dir() == PRETRAINED          # from before the reset to v0.0.0: ignored


def test_update_installs_and_verifies(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOAK_HOME", str(tmp_path))
    blob = b"policy"
    man = {"nous": "9.0.0", "epoch": 1, "min_app": "0.1.0", "skills": {"goto": {"policy.npz": hashlib.sha256(blob).hexdigest()}}}

    class Resp:
        def __init__(self, data): self.data = data
        @property
        def content(self): return self.data
        @property
        def text(self): return "a" * 40
        def json(self): return man

    class Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **k):
            return Resp(blob if url.endswith("policy.npz") else b"")
    monkeypatch.setattr(nous.httpx, "Client", Client)
    out = nous.update()
    assert out["updated"] and nous.installed() == "9.0.0"
    assert (tmp_path / "nous" / "goto" / "policy.npz").read_bytes() == blob
    man["skills"]["goto"]["policy.npz"] = "0" * 64      # a bad download is refused
    man["nous"] = "9.1.0"
    out = nous.update()
    assert "checksum" in out["error"] and nous.installed() == "9.0.0"
