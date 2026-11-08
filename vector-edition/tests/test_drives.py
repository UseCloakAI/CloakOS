import math

from cloakos.config import load_config
from cloakos.drives import Context, DriveModel


def model(**over):
    cfg = {"drives": {
        "fear": {"init": 0.0, "baseline": 0.0, "tau": 8, "high": 0.6, "hysteresis": 0.2},
        "boredom": {"init": 0.0, "baseline": 0.0, "tau": 400, "high": 0.7},
        "energy": {"init": 0.8, "baseline": 0.6, "tau": 1800, "low": 0.25},
    }, "impulses": {"picked_up": {"fear": 0.7}, "face_seen": {"boredom": -0.3}}}
    cfg.update(over)
    return DriveModel(cfg)


def test_decays_toward_baseline_exponentially():
    m = model()
    m.set("fear", 1.0)
    m.update(8.0)  # one tau
    assert math.isclose(m["fear"], math.exp(-1), rel_tol=1e-6)


def test_dt_split_is_equivalent():
    a, b = model(), model()
    a.set("fear", 1.0), b.set("fear", 1.0)
    a.update(5.0)
    for _ in range(50):
        b.update(0.1)
    assert math.isclose(a["fear"], b["fear"], rel_tol=1e-6)


def test_impulse_clamps_and_crosses_once_with_hysteresis():
    m = model()
    assert m.impulse("picked_up") == ["fear.high"]
    assert m.impulse("picked_up") == []          # still armed, value clamped
    assert m["fear"] == 1.0
    m.set("fear", 0.5)                          # inside hysteresis band: stays armed
    assert m.impulse("picked_up") == []
    m.set("fear", 0.3)                          # below high - hysteresis: re-arms
    assert m.impulse("picked_up") == ["fear.high"]


def test_unknown_event_is_noop():
    m = model()
    before = m.snapshot()
    assert m.impulse("nope") == []
    assert m.snapshot() == before


def test_boredom_creeps_when_idle_only():
    m = model()
    m.update(60, Context(idle_s=0))
    assert m["boredom"] == 0.0
    m.update(60, Context(idle_s=60))
    assert m["boredom"] > 0.15


def test_energy_drains_moving_recovers_resting_and_battery_caps():
    m = model()
    e0 = m["energy"]
    m.update(60, Context(moving=True))
    assert m["energy"] < e0
    e1 = m["energy"]
    m.update(60, Context(resting=True))
    assert m["energy"] > e1
    m.update(0.1, Context(battery_low=True))
    assert m["energy"] <= 0.3


def test_low_threshold():
    m = model()
    assert "energy.low" in m.set("energy", 0.2)


def test_shipped_config_loads():
    cfg = load_config("config.yaml")
    m = DriveModel(cfg)
    assert set(m.values) == {"energy", "curiosity", "boredom", "attachment", "fear"}
    assert "face_seen" in m.impulses
