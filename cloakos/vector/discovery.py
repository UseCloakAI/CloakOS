"""Find Vector through wire-pod: serial, IP, auth token and TLS cert, with no manual setup.

wire-pod already holds everything a client needs:
  GET <pod>/api-sdk/get_sdk_info   -> {"global_guid", "robots": [{"esn", "ip_address", "guid", "activated"}]}
  GET <pod>/session-certs/<esn>    -> the robot's TLS cert (PEM)
The cert's name (SAN or CN, e.g. "Vector-A1B2") is what TLS must verify against.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


@dataclass
class BotInfo:
    serial: str
    host: str          # ip:port
    guid: str
    cert: bytes
    name: str          # TLS target name, from the cert


def cert_name(pem: bytes) -> str | None:
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    c = x509.load_pem_x509_certificate(pem)
    try:
        san = c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        names = san.get_values_for_type(x509.DNSName)
        if names:
            return names[0]
    except x509.ExtensionNotFound:
        pass
    cn = c.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    return cn[0].value if cn else None


def robot_cert(ip: str, timeout: float = 5.0) -> bytes | None:
    """The robot's TLS certificate (PEM), read straight from its gRPC port."""
    import ssl
    host, _, port = ip.partition(":")
    try:
        return ssl.get_server_certificate((host, int(port or 443)), timeout=timeout).encode()
    except (OSError, ValueError) as e:
        log.debug("couldn't read cert from %s: %s", ip, e)
        return None


async def discover(cfg: dict, cert_dir: str | Path = "data/certs", transport: httpx.AsyncBaseTransport | None = None) -> BotInfo:
    """Resolve the robot from wire-pod, letting any robot.* value in config override it."""
    r = cfg.get("robot", {}) or {}
    pod = ((cfg.get("wirepod", {}) or {}).get("url") or "http://escapepod.local:8080").rstrip("/")
    serial = r.get("serial") or None
    ip, guid, name = r.get("ip"), r.get("guid"), r.get("name")

    cert = None
    cert_file = Path(cert_dir) / f"{serial}.pem" if serial else None
    async with httpx.AsyncClient(timeout=5, transport=transport) as http:
        if not (ip and guid and serial):
            try:
                info = (await http.get(f"{pod}/api-sdk/get_sdk_info")).raise_for_status().json()
            except httpx.HTTPError as e:
                raise ConnectionError(f"can't reach wire-pod at {pod} ({e or type(e).__name__}); set WIREPOD_URL") from e
            bots = info.get("robots") or []
            bot = next((b for b in bots if not serial or b.get("esn", "").lower() == serial.lower()), None)
            if bot is None:
                have = ", ".join(b.get("esn", "?") for b in bots) or "none"
                raise LookupError(f"wire-pod at {pod} has no bot {serial or ''} (authenticated: {have})")
            serial = bot["esn"]          # keep wire-pod's spelling: cert paths are case-sensitive
            ip = ip or bot["ip_address"]
            guid = guid or bot.get("guid") or info.get("global_guid")
            cert_file = Path(cert_dir) / f"{serial}.pem"
        if cert_file and cert_file.exists():
            cert = cert_file.read_bytes()
        else:
            resp = await http.get(f"{pod}/session-certs/{serial}")
            if resp.status_code == 404:
                # wire-pod only saves the cert while the robot is being activated, so it can know the
                # bot but not its cert (e.g. after its data was reset). The cert is the robot's own TLS
                # certificate, so read it from the robot (wire-pod itself doesn't verify it at all).
                cert = await asyncio.to_thread(robot_cert, ip)
                if not cert:
                    raise LookupError(f"wire-pod has no cert for {serial} and the robot didn't answer at {ip}; "
                                      "pair it again (Pair Vector → Activate)")
                log.info("got %s's certificate from the robot itself", serial)
            else:
                resp.raise_for_status()
                cert = resp.content
            cert_file.parent.mkdir(parents=True, exist_ok=True)
            cert_file.write_bytes(cert)

    name = name or cert_name(cert)
    if not name:
        raise ValueError("couldn't read the robot name from its cert; set robot.name (e.g. Vector-A1B2)")
    host = ip if ":" in ip else f"{ip}:443"
    log.info("found %s (%s) at %s via %s", name, serial, host, pod)
    return BotInfo(serial=serial, host=host, guid=guid, cert=cert, name=name)
