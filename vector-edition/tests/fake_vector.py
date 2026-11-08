"""A fake Vector: real TLS gRPC server speaking the robot's external interface.

Checks the bearer token like the robot does, grants behavior control, streams robot
state, a face and camera frames, and records every motor/screen/audio call.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import io

import grpc
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from PIL import Image

from cloakos.vector.client import m
from cloakos.vector.pb import external_interface_pb2_grpc as rpc

NAME, GUID, SERIAL = "Vector-T3ST", "test-guid-123", "00e2feed"


def make_cert(name: str = NAME) -> tuple[bytes, bytes]:
    key = ec.generate_private_key(ec.SECP256R1())
    subj = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subj).issuer_name(subj).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=30))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(name)]), critical=False)
            .sign(key, hashes.SHA256()))
    key_pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return cert.public_bytes(serialization.Encoding.PEM), key_pem


class FakeVector(rpc.ExternalInterfaceServicer):
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.audio_bytes = 0
        self.control = False
        self.picked_up = False

    async def _auth(self, ctx) -> None:
        md = dict(ctx.invocation_metadata())
        if md.get("authorization") != f"Bearer {GUID}":
            await ctx.abort(grpc.StatusCode.UNAUTHENTICATED, "bad token")

    async def ProtocolVersion(self, req, ctx):
        await self._auth(ctx)
        return m.ProtocolVersionResponse(host_version=4)

    async def BehaviorControl(self, reqs, ctx):
        await self._auth(ctx)
        async for r in reqs:
            if r.WhichOneof("request_type") == "control_request":
                self.control = True
                yield m.BehaviorControlResponse(control_granted_response=m.ControlGrantedResponse())
            else:
                self.control = False
                yield m.BehaviorControlResponse(control_lost_event=m.ControlLostResponse())

    async def EventStream(self, req, ctx):
        await self._auth(ctx)
        while True:
            status = 0x8 if self.picked_up else 0
            state = m.RobotState(head_angle_rad=0.3, lift_height_mm=40.0, status=status,
                                 touch_data=m.TouchData(is_being_touched=False),
                                 prox_data=m.ProxData(distance_mm=150, found_object=True))
            yield m.EventResponse(event=m.Event(robot_state=state))
            face = m.RobotObservedFace(face_id=1, img_rect=m.CladRect(x_top_left=280, y_top_left=120, width=80, height=80))
            yield m.EventResponse(event=m.Event(robot_observed_face=face))
            await asyncio.sleep(0.05)

    async def CameraFeed(self, req, ctx):
        await self._auth(ctx)
        buf = io.BytesIO()
        Image.new("RGB", (640, 360), "blue").save(buf, "JPEG")
        i = 0
        while True:
            i += 1
            yield m.CameraFeedResponse(image_id=i, image_encoding=7, data=buf.getvalue())
            await asyncio.sleep(0.1)

    async def DriveWheels(self, req, ctx):
        self.calls.append(("DriveWheels", (req.left_wheel_mmps, req.right_wheel_mmps)))
        return m.DriveWheelsResponse()

    async def MoveHead(self, req, ctx):
        self.calls.append(("MoveHead", req.speed_rad_per_sec))
        return m.MoveHeadResponse()

    async def MoveLift(self, req, ctx):
        self.calls.append(("MoveLift", req.speed_rad_per_sec))
        return m.MoveLiftResponse()

    async def StopAllMotors(self, req, ctx):
        self.calls.append(("StopAllMotors", None))
        return m.StopAllMotorsResponse()

    async def SetHeadAngle(self, req, ctx):
        self.calls.append(("SetHeadAngle", round(req.angle_rad, 3)))
        return m.SetHeadAngleResponse()

    async def SetLiftHeight(self, req, ctx):
        self.calls.append(("SetLiftHeight", round(req.height_mm, 1)))
        return m.SetLiftHeightResponse()

    async def TurnInPlace(self, req, ctx):
        self.calls.append(("TurnInPlace", round(req.angle_rad, 3)))
        return m.TurnInPlaceResponse()

    async def DriveStraight(self, req, ctx):
        self.calls.append(("DriveStraight", req.dist_mm))
        return m.DriveStraightResponse()

    async def DisplayFaceImageRGB(self, req, ctx):
        self.calls.append(("DisplayFaceImageRGB", len(req.face_data)))
        return m.DisplayFaceImageRGBResponse()

    async def SetMasterVolume(self, req, ctx):
        self.calls.append(("MasterVolume", req.volume_level))
        return m.MasterVolumeResponse()

    async def EnableFaceDetection(self, req, ctx):
        return m.EnableFaceDetectionResponse()

    async def ListAnimationTriggers(self, req, ctx):
        names = ["GreetAfterLongTime", "Happy", "GoToSleepGetIn"]
        return m.ListAnimationTriggersResponse(animation_trigger_names=[m.AnimationTrigger(name=n) for n in names])

    async def PlayAnimationTrigger(self, req, ctx):
        self.calls.append(("PlayAnimationTrigger", req.animation_trigger.name))
        return m.PlayAnimationResponse()

    async def BatteryState(self, req, ctx):
        await self._auth(ctx)
        return m.BatteryStateResponse(battery_level=2, battery_volts=3.9)

    async def ExternalAudioStreamPlayback(self, reqs, ctx):
        prepared = None
        async for r in reqs:
            kind = r.WhichOneof("audio_request_type")
            if kind == "audio_stream_prepare":
                prepared = r.audio_stream_prepare.audio_frame_rate
                self.calls.append(("AudioVolume", r.audio_stream_prepare.audio_volume))
            elif kind == "audio_stream_chunk":
                assert r.audio_stream_chunk.audio_chunk_size_bytes <= 1024
                self.audio_bytes += len(r.audio_stream_chunk.audio_chunk_samples)
            elif kind == "audio_stream_complete":
                self.calls.append(("Audio", (prepared, self.audio_bytes)))
                yield m.ExternalAudioStreamResponse(audio_stream_playback_complete=m.ExternalAudioStreamPlaybackComplete())
                return


async def start_fake(cert: bytes, key: bytes) -> tuple[grpc.aio.Server, FakeVector, int]:
    server = grpc.aio.server()
    fake = FakeVector()
    rpc.add_ExternalInterfaceServicer_to_server(fake, server)
    port = server.add_secure_port("127.0.0.1:0", grpc.ssl_server_credentials([(key, cert)]))
    await server.start()
    return server, fake, port
