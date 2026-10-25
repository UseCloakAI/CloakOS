"""End-to-end: the whole brain against a fake SDK robot (no hardware, no network)."""
import asyncio
import concurrent.futures
import json
from types import SimpleNamespace as NS

from cloak_vector.app import Brain
from cloak_vector.config import Config, load_config


def done(value=None):
    f = concurrent.futures.Future()
    f.set_result(value)
    return f


class Recorder:
    """Any attribute is a callable that logs the call and returns a finished future."""

    def __init__(self, log, prefix):
        self._log, self._prefix = log, prefix

    def __getattr__(self, name):
        def call(*a, **k):
            self._log.append((f"{self._prefix}.{name}", a))
            return done()
        return call


class FakeRobot:
    def __init__(self):
        self.calls = []
        granted = NS(is_set=lambda: self.control)
        self.control = False
        self.conn = NS(control_granted_event=granted,
                       request_control=lambda timeout=10: setattr(self, "control", True),
                       release_control=lambda timeout=10: setattr(self, "control", False))
        self.status = NS(is_picked_up=False, is_being_held=False, is_cliff_detected=False, is_button_pressed=False,
                         is_on_charger=False, is_charging=False, are_wheels_moving=False, is_animating=False)
        self.touch = NS(last_sensor_reading=NS(is_being_touched=False))
        self.proximity = NS(last_sensor_reading=None)
        self.head_angle_rad, self.lift_height_mm = 0.1, 32.0
        self.left_wheel_speed_mmps = self.right_wheel_speed_mmps = 0.0
        self.accel = self.gyro = NS(x=0.0, y=0.0, z=9.8)
        self.pose = NS(position=NS(x=1.0, y=2.0), rotation=NS(angle_z=NS(radians=0.5)))
        self.camera = NS(latest_image=None, latest_image_id=0)
        self.motors = Recorder(self.calls, "motors")
        self.behavior = Recorder(self.calls, "behavior")
        self.audio = Recorder(self.calls, "audio")
        self.anim = Recorder(self.calls, "anim")

    def get_battery_state(self):
        return done(NS(battery_level=2, battery_volts=3.91))


async def test_brain_end_to_end(tmp_path):
    cfg = Config(load_config("config.yaml"))
    cfg["tick_hz"] = 20
    cfg["datalog"]["dir"] = str(tmp_path)
    cfg["voice"]["enabled"] = False
    cfg["robot"]["heartbeat_s"] = 0.2
    brain = Brain(cfg)
    fake = FakeRobot()

    async def fake_connect():
        brain.link.robot = fake
        brain.link.anim_triggers = ["GreetAfterLongTime", "GoToSleepGetIn", "Frustrated"]
        brain.link._set_state("connected")
        if brain.link._want_control:
            await brain.link._request_control()

    brain.link._connect = fake_connect
    seen = []
    brain.bus.on("*", lambda ev: seen.append(ev.topic))

    task = asyncio.create_task(brain.run(mode="auto", web=False))
    await asyncio.sleep(0.6)
    assert brain.link.has_control and brain.world.battery_volts == 3.91
    assert brain.policy.current != "-"                       # a behavior is running

    brain.bus.emit("sense.picked_up")
    assert brain.drives["fear"] > 0.3

    await brain.set_mode("teleop")
    assert brain.recorder.active                             # auto_record: [teleop]
    brain.teleop.input(throttle=1.0)
    await asyncio.sleep(0.2)
    assert ("motors.set_wheel_motors", (150.0, 150.0, 400, 400)) in fake.calls

    await brain.set_mode("idle")
    assert not fake.control                                   # stock Vector gets control back

    brain.request_stop()
    await asyncio.wait_for(task, 10)
    assert "motors.stop_all_motors" in [c[0] for c in fake.calls]
    ep = next((tmp_path / "episodes").iterdir())
    steps = [json.loads(l) for l in open(ep / "steps.jsonl")]
    assert steps and any(s["action"]["left_mmps"] == 150.0 for s in steps)
    assert {"obs", "drives", "behavior", "action", "events"} <= set(steps[0])
    assert "robot.connection" in seen and "mode.changed" in seen
