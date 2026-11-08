import json

from cloakos import share


def test_upload_run_sends_run_then_files(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text(json.dumps({"task": "goto", "sensors": "map", "trained_steps": 1000}))
    (tmp_path / "progress.jsonl").write_text("\n".join(json.dumps({"steps": i * 100, "t": i, "fps": 50}) for i in range(1, 11)))
    (tmp_path / "exam.json").write_text(json.dumps({"passed": True}))
    (tmp_path / "policy.npz").write_bytes(b"\x01\x02")
    sent = []

    class R:
        status_code = 201
        text = ""
    monkeypatch.setattr(share.httpx, "post", lambda url, json, **k: sent.append((url.rsplit("/", 1)[1], json)) or R())
    rid = share.upload_run(tmp_path, "local")
    assert [t for t, _ in sent] == ["training_runs", "training_policies"]
    run, pol = sent[0][1], sent[1][1]
    assert run["id"] == rid and run["task"] == "goto" and run["steps"] == 1000 and run["exam"] == {"passed": True}
    assert pol["run_id"] == rid and pol["data"] == "\\x0102"


def test_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("CLOAK_SHARE", "0")
    assert not share.enabled()
