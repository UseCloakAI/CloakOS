"""End-to-end: the whole brain against a fake Vector over real TLS gRPC (no hardware)."""
import asyncio
import json

from cloakos.app import Brain
from cloakos.config import Config, load_config

from .fake_vector import GUID, SERIAL, make_cert, start_fake


async def test_brain_end_to_end(tmp_path):
    cert, key = make_cert()
    server, fv, port = await start_fake(cert, key)
    (tmp_path / "certs").mkdir()
    (tmp_path / "certs" / f"{SERIAL}.pem").write_bytes(cert)

    cfg = Config(load_config("config.yaml"))
    cfg["tick_hz"] = 20
    cfg["datalog"]["dir"] = str(tmp_path)
    cfg["voice"]["enabled"] = False
    cfg["expression"]["stock_blend"] = 0.0
    # serial + ip + guid + cached cert: discovery skips wire-pod's HTTP API entirely
    cfg["robot"].update(serial=SERIAL, ip=f"127.0.0.1:{port}", guid=GUID, cert_dir=str(tmp_path / "certs"), heartbeat_s=0.2)
    brain = Brain(cfg)
    seen = []
    brain.bus.on("*", lambda ev: seen.append(ev.topic))

    task = asyncio.create_task(brain.run(mode="auto", web=False))
    for _ in range(60):
        await asyncio.sleep(0.1)
        if brain.link.has_control and brain.world.battery_volts:
            break
    assert brain.link.has_control and brain.world.battery_volts == 3.9
    await asyncio.sleep(0.4)
    assert brain.policy.current != "-"
    assert brain.world.face is not None and "sense.face_seen" in seen
    assert brain.world.frame is not None and brain.world.head_angle == 0.3
    assert any(c[0] == "DisplayFaceImageRGB" for c in fv.calls)            # Cloak's eyes on the screen

    fv.picked_up = True
    await asyncio.sleep(0.3)
    assert "sense.picked_up" in seen and brain.drives["fear"] > 0.3

    await brain.set_mode("teleop")
    assert brain.recorder.active
    brain.teleop.input(throttle=1.0)
    await asyncio.sleep(0.3)
    assert ("DriveWheels", (150.0, 150.0)) in fv.calls

    await brain.link.set_volume(40)                                         # master volume + scaled clip volume
    assert ("MasterVolume", 2) in fv.calls
    await brain.body.play_pcm(b"\0" * 2048, volume=75)
    assert ("AudioVolume", 30) in fv.calls
    await brain.link.set_volume(0)                                          # muted: nothing goes to the speaker
    n = len(fv.calls)
    await brain.body.play_pcm(b"\0" * 2048, volume=75)
    assert not any(c[0] == "AudioVolume" for c in fv.calls[n:])

    await brain.set_mode("idle")
    await asyncio.sleep(0.2)
    assert not fv.control                                                   # stock Vector gets control back

    brain.request_stop()
    await asyncio.wait_for(task, 10)
    ep = next((tmp_path / "episodes").iterdir())
    steps = [json.loads(l) for l in open(ep / "steps.jsonl")]
    assert any(s["action"]["left_mmps"] == 150.0 for s in steps)
    assert any(s["frame"] for s in steps)
    assert {"obs", "drives", "behavior", "action", "events"} <= set(steps[0])
    assert "robot.connection" in seen and "mode.changed" in seen
    await server.stop(0)
