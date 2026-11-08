import json
import wave

import numpy as np
from PIL import Image

from cloakos.datalog import Recorder
from cloakos.voice.audio import to_vector_wav
from cloakos.voice.llm import clean_line


def test_clean_line():
    assert clean_line('"Hello there, friend!" *beeps happily*') == "Hello there, friend!"
    assert clean_line("one two three four five six seven eight") == "one two three four five six"
    assert clean_line("Hi 🤖 you\nsecond line") == "Hi you"
    assert clean_line("   ") == ""


def test_vector_wav_format(tmp_path):
    sr = 24000
    x = np.sin(np.linspace(0, 2 * np.pi * 440, sr)).astype(np.float32)
    p = to_vector_wav(x, sr, tmp_path / "a.wav", pitch=1.0)
    with wave.open(str(p)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert abs(w.getnframes() - (16000 + 800)) < 5
    p2 = to_vector_wav(x, sr, tmp_path / "b.wav", pitch=1.25)
    with wave.open(str(p2)) as w:
        assert w.getnframes() < 16000          # higher pitch = shorter clip


def test_recorder_writes_episode(tmp_path):
    rec = Recorder({"datalog": {"dir": str(tmp_path), "frame_size": [32, 18]}, "llm": {"api_key": "sekret"}})
    d = rec.start("teleop", 10)
    img = Image.new("RGB", (640, 360), "red")
    for i in range(5):
        rec.record({"t": i, "action": {"left_mmps": i}}, frame=img, frame_id=i // 2)
    rec.stop()
    lines = [json.loads(l) for l in open(d / "steps.jsonl")]
    assert [l["i"] for l in lines] == [0, 1, 2, 3, 4]
    assert [l["frame"] is not None for l in lines] == [True, False, True, False, True]
    assert Image.open(d / lines[0]["frame"]).size == (32, 18)
    meta = json.loads((d / "meta.json").read_text())
    assert meta["config"]["llm"]["api_key"] == "***"
    assert rec.list_episodes()[0]["steps"] == 5


def test_chirps_and_eyes():
    from cloakos.expression import Eyes, chirp
    from cloakos.expression.chirps import CHIRPS

    for name in CHIRPS:
        pcm = chirp(name)
        assert len(pcm) % 2 == 0 and 0.1 < len(pcm) / 2 / 16000 < 1.5
    eyes = Eyes({})
    d = {"energy": 0.8, "curiosity": 0.5, "boredom": 0.1, "attachment": 0.9, "fear": 0.0}
    happy = eyes.shape(d, "approach_face", (0.8, 0.5), now=1.0)
    assert happy.happy > 0.3 and happy.gaze_x > 0.5
    nap = eyes.shape(d, "nap", None, now=1.0)
    assert nap.open < 0.1
    img = eyes.render(happy)   # white rounded eyes on black
    assert img.size == (184, 96)
    assert img.getpixel((92, 2)) == (0, 0, 0)                 # background stays black


def test_setup_env_roundtrip(tmp_path):
    from cloakos.web.setup_api import read_env, write_env

    p = tmp_path / ".env"
    p.write_text("# comment\n# LLM_MODEL=old\nWIREPOD_URL=http://a:8080\nOTHER=keep\n")
    write_env(p, {"WIREPOD_URL": "http://b:8080", "LLM_MODEL": "llama3.2:3b", "VECTOR_SERIAL": "00e2"})
    text = p.read_text()
    assert "# comment" in text and "OTHER=keep" in text
    assert read_env(p) == {"WIREPOD_URL": "http://b:8080", "LLM_MODEL": "llama3.2:3b", "OTHER": "keep", "VECTOR_SERIAL": "00e2"}


def test_robot_ota_url_never_localhost(monkeypatch):
    from cloakos.web import setup_api

    monkeypatch.setattr(setup_api, "lan_ip", lambda: "192.168.1.20")
    assert setup_api.robot_ota_url("http://localhost:8080") == "http://192.168.1.20:8080/api/get_ota/vicos-2.0.1.6076ep.ota"
    assert setup_api.robot_ota_url("http://10.0.0.5:8090") == "http://10.0.0.5:8090/api/get_ota/vicos-2.0.1.6076ep.ota"
