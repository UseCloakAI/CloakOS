import json
import wave

import numpy as np
from PIL import Image

from cloak_vector.datalog import Recorder
from cloak_vector.voice.audio import to_vector_wav
from cloak_vector.voice.llm import clean_line


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
