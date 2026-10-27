"""Our gRPC client against a fake Vector over real TLS + token auth, and wire-pod discovery."""
import asyncio

import httpx
import pytest

from cloakos.vector import BotInfo, VectorClient, discover, rgb565
from cloakos.vector.discovery import cert_name

from .fake_vector import GUID, NAME, SERIAL, make_cert, start_fake


@pytest.fixture
async def fake():
    cert, key = make_cert()
    server, fv, port = await start_fake(cert, key)
    yield fv, BotInfo(serial=SERIAL, host=f"127.0.0.1:{port}", guid=GUID, cert=cert, name=NAME)
    await server.stop(0)


async def test_client_roundtrip(fake):
    fv, bot = fake
    c = VectorClient(bot)
    await c.connect(5)
    c.spawn(c.control_loop(), "control")
    events = []
    c.spawn(c.events(lambda k, msg: events.append(k)), "events")
    frames = []
    c.spawn(c.camera(lambda img, i: frames.append(img.size)), "camera")

    assert await c.request_control(timeout=3)
    await c.drive_wheels(100, -100)
    await c.head_angle(20)
    await c.show_face(b"\x00" * 184 * 96 * 2, 300)
    await c.play_pcm(b"\x01\x00" * 4000, rate=16000)      # 8000 bytes -> 8 chunks
    assert await c.anim_triggers() == ["GoToSleepGetIn", "GreetAfterLongTime", "Happy"]
    await asyncio.sleep(0.3)

    assert ("DriveWheels", (100.0, -100.0)) in fv.calls
    assert ("DisplayFaceImageRGB", 184 * 96 * 2) in fv.calls
    assert ("Audio", (16000, 8000)) in fv.calls
    assert "robot_state" in events and "robot_observed_face" in events
    assert frames and frames[0] == (640, 360)
    assert c.state is not None and not c.status("picked_up")
    fv.picked_up = True
    await asyncio.sleep(0.15)
    assert c.status("picked_up")

    await c.release_control()
    await asyncio.sleep(0.1)
    assert not fv.control
    await c.close()


async def test_bad_token_is_rejected(fake):
    _, bot = fake
    bot.guid = "wrong"
    c = VectorClient(bot)
    with pytest.raises(Exception):
        await c.connect(5)
    await c.close()


def test_rgb565_big_endian():
    from PIL import Image
    data = rgb565(Image.new("RGB", (184, 96), (255, 0, 0)))
    assert len(data) == 184 * 96 * 2 and data[:2] == b"\xf8\x00"


async def test_discover_from_wirepod(tmp_path):
    cert, _ = make_cert("Vector-D1SC")
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if req.url.path == "/api-sdk/get_sdk_info":
            return httpx.Response(200, json={"global_guid": "g", "robots": [
                {"esn": "00e2aaaa", "ip_address": "10.0.0.9", "guid": "tok", "activated": True}]})
        if req.url.path == "/session-certs/00e2aaaa":
            return httpx.Response(200, content=cert)
        return httpx.Response(404)

    cfg = {"wirepod": {"url": "http://pod:8080"}, "robot": {}}
    bot = await discover(cfg, cert_dir=tmp_path, transport=httpx.MockTransport(handler))
    assert (bot.serial, bot.host, bot.guid, bot.name) == ("00e2aaaa", "10.0.0.9:443", "tok", "Vector-D1SC")
    assert (tmp_path / "00e2aaaa.pem").read_bytes() == cert
    assert cert_name(cert) == "Vector-D1SC"

    with pytest.raises(LookupError, match="00e2aaaa"):
        await discover({"wirepod": {"url": "http://pod:8080"}, "robot": {"serial": "00e2bbbb"}},
                       cert_dir=tmp_path, transport=httpx.MockTransport(handler))


def test_connect_error_hints():
    from cloakos.connection.robot import hint_for

    grpc_msg = ("failed to connect to all addresses; last error: UNKNOWN: ipv4:192.168.1.67:443: "
                "Failed to connect to remote host: No route to host")
    assert hint_for(grpc_msg) == "local_network"
    assert hint_for("can't reach wire-pod at http://escapepod.local:8080 (...)") == "no_wirepod"
    assert hint_for("wire-pod has no cert for 003091c9 yet; finish setting the bot up") == "not_paired"
    assert hint_for("heartbeat lost") is None
