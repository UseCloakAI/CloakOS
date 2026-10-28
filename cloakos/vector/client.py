"""VectorClient: a small async gRPC client for Vector's external interface. Replaces the Anki SDK.

Only what Cloak uses: behavior control, the event stream (robot state, faces, wake word),
camera feed, motors, a few actions, animations, the speaker and the face screen.
Everything runs on our own asyncio loop, so there are no threads and no futures to bridge.
"""
from __future__ import annotations

import asyncio
import io
import logging
import math
from typing import Any, AsyncIterator, Callable

import grpc

from .discovery import BotInfo
from .pb import behavior_pb2, messages_pb2, shared_pb2
from .pb import external_interface_pb2_grpc as rpc

log = logging.getLogger(__name__)


class _Messages:
    """One namespace over the generated modules: m.DriveWheelsRequest, m.ControlRequest, ..."""

    def __getattr__(self, name: str) -> Any:
        for mod in (messages_pb2, shared_pb2, behavior_pb2):
            if hasattr(mod, name):
                return getattr(mod, name)
        raise AttributeError(name)


m = _Messages()

AUDIO_CHUNK = 1024   # bytes per chunk (engine max)
# RobotStatus bits (messages.proto)
STATUS = {
    "moving": 0x1, "picked_up": 0x8, "button": 0x10, "falling": 0x20, "animating": 0x40,
    "on_charger": 0x1000, "charging": 0x2000, "cliff": 0x4000, "wheels_moving": 0x8000, "held": 0x10000,
}
# No gRPC keepalive pings: Vector's Go server allows one per 5 min and drops the connection
# (GOAWAY too_many_pings) if we ping faster. RobotLink's battery heartbeat does that job.
CHANNEL_OPTIONS = [("grpc.keepalive_time_ms", 2**31 - 1), ("grpc.max_receive_message_length", 8 * 1024 * 1024)]
PRIORITY = {"override": m.ControlRequest.OVERRIDE_BEHAVIORS, "default": m.ControlRequest.DEFAULT,
            "reserve": m.ControlRequest.RESERVE_CONTROL}


class VectorClient:
    def __init__(self, bot: BotInfo) -> None:
        self.bot = bot
        self.channel: grpc.aio.Channel | None = None
        self.stub: rpc.ExternalInterfaceStub | None = None
        self.state: Any = None                   # latest RobotState message
        self.has_control = False
        self.control_changed = asyncio.Event()
        self._control_q: asyncio.Queue = asyncio.Queue()   # requests queue up until control_loop runs
        self._tasks: list[asyncio.Task] = []
        self._audio_lock = asyncio.Lock()
        self._next_id = 1
        self._bad_frames = 0

    # ── lifecycle ──────────────────────────────────────────────

    async def connect(self, timeout: float = 10) -> None:
        creds = grpc.composite_channel_credentials(
            grpc.ssl_channel_credentials(root_certificates=self.bot.cert),
            grpc.access_token_call_credentials(self.bot.guid),
        )
        self.channel = grpc.aio.secure_channel(
            self.bot.host, creds,
            options=CHANNEL_OPTIONS + [("grpc.ssl_target_name_override", self.bot.name)],
        )
        await asyncio.wait_for(self.channel.channel_ready(), timeout)
        self.stub = rpc.ExternalInterfaceStub(self.channel)
        await self.stub.ProtocolVersion(m.ProtocolVersionRequest(client_version=4, min_host_version=0), timeout=5)

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self.channel is not None:
            await self.channel.close()
            self.channel = None
        self.has_control = False

    def spawn(self, coro, name: str) -> asyncio.Task:
        t = asyncio.create_task(coro, name=name)
        self._tasks.append(t)
        return t

    # ── streams ────────────────────────────────────────────────

    async def events(self, on_event: Callable[[str, Any], None]) -> None:
        """Run the event stream; calls on_event(kind, msg) for every event. Returns when the stream dies."""
        req = m.EventRequest(connection_id="cloak")
        async for resp in self.stub.EventStream(req):
            ev = resp.event
            kind = ev.WhichOneof("event_type")
            if kind == "robot_state":
                self.state = ev.robot_state
            if kind:
                on_event(kind, getattr(ev, kind))

    async def camera(self, on_frame: Callable[[Any, int], None]) -> None:
        """Run the camera feed; calls on_frame(PIL image, image_id)."""
        from PIL import Image

        async for resp in self.stub.CameraFeed(m.CameraFeedRequest()):
            try:
                img = Image.open(io.BytesIO(resp.data))
                img.load()
                on_frame(img, resp.image_id)
            except Exception as e:
                self._bad_frames += 1
                if self._bad_frames in (1, 100):
                    log.warning("camera frame %d undecodable (encoding %s): %s", self._bad_frames, resp.image_encoding, e)

    async def control_loop(self, priority: str = "default") -> None:
        """Hold the BehaviorControl stream. Use request_control()/release_control() to drive it."""

        async def requests() -> AsyncIterator[m.BehaviorControlRequest]:
            while True:
                yield await self._control_q.get()

        async for resp in self.stub.BehaviorControl(requests()):
            kind = resp.WhichOneof("response_type")
            if kind == "control_granted_response":
                self.has_control = True
            elif kind in ("control_lost_event", "reserved_control_lost_event"):
                self.has_control = False
            else:
                continue
            log.info("behavior control: %s", kind.replace("_", " "))
            self.control_changed.set()
            self.control_changed = asyncio.Event()

    async def request_control(self, priority: str = "default", timeout: float = 10) -> bool:
        if self.has_control:
            return True
        waiter = self.control_changed
        await self._control_q.put(m.BehaviorControlRequest(control_request=m.ControlRequest(priority=PRIORITY[priority])))
        try:
            await asyncio.wait_for(waiter.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return self.has_control

    async def release_control(self) -> None:
        await self._control_q.put(m.BehaviorControlRequest(control_release=m.ControlRelease()))
        self.has_control = False

    # ── state helpers ──────────────────────────────────────────

    def status(self, flag: str) -> bool:
        return bool(self.state is not None and self.state.status & STATUS[flag])

    # ── motors + actions ───────────────────────────────────────

    def _tag(self) -> int:
        self._next_id = self._next_id % 2_000_000 + 1
        return self._next_id

    async def drive_wheels(self, left: float, right: float, accel: float = 0.0) -> None:
        await self.stub.DriveWheels(m.DriveWheelsRequest(left_wheel_mmps=left, right_wheel_mmps=right,
                                                         left_wheel_mmps2=accel, right_wheel_mmps2=accel), timeout=2)

    async def move_head(self, rad_s: float) -> None:
        await self.stub.MoveHead(m.MoveHeadRequest(speed_rad_per_sec=rad_s), timeout=2)

    async def move_lift(self, rad_s: float) -> None:
        await self.stub.MoveLift(m.MoveLiftRequest(speed_rad_per_sec=rad_s), timeout=2)

    async def stop_all(self) -> None:
        await self.stub.StopAllMotors(m.StopAllMotorsRequest(), timeout=2)

    async def head_angle(self, deg: float, speed: float = 10.0, duration: float = 0.0) -> None:
        await self.stub.SetHeadAngle(m.SetHeadAngleRequest(
            angle_rad=math.radians(deg), max_speed_rad_per_sec=speed, accel_rad_per_sec2=10.0,
            duration_sec=duration, id_tag=self._tag()), timeout=10)

    async def lift_height(self, frac: float, speed: float = 10.0, duration: float = 0.0) -> None:
        mm = 32.0 + max(0.0, min(1.0, frac)) * (92.0 - 32.0)   # lift travel: 32..92 mm
        await self.stub.SetLiftHeight(m.SetLiftHeightRequest(
            height_mm=mm, max_speed_rad_per_sec=speed, accel_rad_per_sec2=10.0,
            duration_sec=duration, id_tag=self._tag()), timeout=10)

    async def turn(self, deg: float, speed_dps: float = 90.0) -> None:
        await self.stub.TurnInPlace(m.TurnInPlaceRequest(
            angle_rad=math.radians(deg), speed_rad_per_sec=math.radians(speed_dps), accel_rad_per_sec2=math.radians(360),
            tol_rad=math.radians(2), id_tag=self._tag()), timeout=15)

    async def straight(self, mm: float, speed_mmps: float = 80.0) -> None:
        await self.stub.DriveStraight(m.DriveStraightRequest(
            speed_mmps=abs(speed_mmps), dist_mm=mm, should_play_animation=False, id_tag=self._tag()), timeout=15)

    async def drive_on_charger(self) -> bool:
        """Vector's own dock: finds the charger's marker and backs on. Can take a minute or two."""
        r = await self.stub.DriveOnCharger(m.DriveOnChargerRequest(), timeout=180)
        return getattr(getattr(r, "result", None), "code", 0) in (0, 1)

    async def drive_off_charger(self) -> None:
        await self.stub.DriveOffCharger(m.DriveOffChargerRequest(), timeout=30)

    # ── animations, screen, speaker, misc ──────────────────────

    async def anim_triggers(self) -> list[str]:
        r = await self.stub.ListAnimationTriggers(m.ListAnimationTriggersRequest(), timeout=15)
        return sorted(t.name for t in r.animation_trigger_names)

    async def play_trigger(self, name: str) -> None:
        await self.stub.PlayAnimationTrigger(m.PlayAnimationTriggerRequest(
            animation_trigger=m.AnimationTrigger(name=name), loops=1), timeout=20)

    async def show_face(self, rgb565: bytes, duration_ms: int, interrupt: bool = True) -> None:
        await self.stub.DisplayFaceImageRGB(m.DisplayFaceImageRGBRequest(
            face_data=rgb565, duration_ms=duration_ms, interrupt_running=interrupt), timeout=2)

    async def eye_color(self, hue: float, saturation: float) -> None:
        await self.stub.SetEyeColor(m.SetEyeColorRequest(hue=hue, saturation=saturation), timeout=2)

    async def master_volume(self, level: int) -> None:
        """Vector's own speaker volume (stock sounds + ours): 0 low .. 4 high."""
        await self.stub.SetMasterVolume(m.MasterVolumeRequest(volume_level=max(0, min(4, level))), timeout=5)

    async def face_detection(self, on: bool = True) -> None:
        await self.stub.EnableFaceDetection(m.EnableFaceDetectionRequest(enable=on), timeout=5)

    async def battery(self) -> Any:
        return await self.stub.BatteryState(m.BatteryStateRequest(), timeout=5)

    async def play_pcm(self, pcm: bytes, rate: int = 16000, volume: int = 75) -> None:
        """Stream 16-bit mono PCM to the speaker. Paced to real time so the robot's buffer never overruns."""
        async with self._audio_lock:
            async def reqs():
                yield m.ExternalAudioStreamRequest(audio_stream_prepare=m.ExternalAudioStreamPrepare(
                    audio_frame_rate=rate, audio_volume=max(0, min(100, volume))))
                loop = asyncio.get_running_loop()
                t0, sent = loop.time(), 0
                for i in range(0, len(pcm), AUDIO_CHUNK):
                    chunk = pcm[i:i + AUDIO_CHUNK]
                    yield m.ExternalAudioStreamRequest(audio_stream_chunk=m.ExternalAudioStreamChunk(
                        audio_chunk_size_bytes=len(chunk), audio_chunk_samples=chunk))
                    sent += len(chunk) // 2
                    ahead = sent / rate - (loop.time() - t0)
                    if ahead > 0.5:                      # keep ~0.5 s buffered on the robot
                        await asyncio.sleep(ahead - 0.5)
                yield m.ExternalAudioStreamRequest(audio_stream_complete=m.ExternalAudioStreamComplete())

            async for resp in self.stub.ExternalAudioStreamPlayback(reqs()):
                kind = resp.WhichOneof("audio_response_type")
                if kind == "audio_stream_playback_complete":
                    return
                if kind:
                    raise RuntimeError(f"audio playback: {kind}")


def rgb565(img: Any) -> bytes:
    """PIL image (184x96) -> Vector screen bytes (RGB565, big-endian)."""
    import numpy as np

    a = np.asarray(img.convert("RGB"), dtype=np.uint16)
    v = ((a[..., 0] >> 3) << 11) | ((a[..., 1] >> 2) << 5) | (a[..., 2] >> 3)
    return v.astype(">u2").tobytes()

