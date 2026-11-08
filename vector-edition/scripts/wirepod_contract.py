"""Does a wire-pod release still offer everything Cloak relies on?

    python scripts/wirepod_contract.py path/to/wire-pod

Checks the wire-pod source for the exact HTTP endpoints and JSON fields Cloak uses
(discovery, certs, first-run setup). Exits 1 and lists what's missing if anything
changed. The robot's own gRPC interface lives on Vector, not in wire-pod, so it isn't
affected by wire-pod updates.
"""
import re
import sys
from pathlib import Path

# (what Cloak needs, file glob under chipper/pkg, regex that must match)
CONTRACT = [
    ("GET /api-sdk/get_sdk_info (bot discovery)", "wirepod/sdkapp/*.go", r'"/api-sdk/get_sdk_info"'),
    ("bot list JSON: esn", "vars/*.go", r'json:"esn"'),
    ("bot list JSON: ip_address", "vars/*.go", r'json:"ip_address"'),
    ("bot list JSON: guid", "vars/*.go", r'json:"guid"'),
    ("bot list JSON: global_guid", "vars/*.go", r'json:"global_guid"'),
    ("bot list JSON: robots", "vars/*.go", r'json:"robots"'),
    ("GET /session-certs/<serial> (robot TLS cert)", "wirepod/config-ws/*.go", r'"/session-certs/"'),
    ("GET /api/is_running", "wirepod/config-ws/*.go", r'case "is_running"'),
    ("GET /api/get_config", "wirepod/config-ws/*.go", r'case "get_config"'),
    ("config JSON: pastinitialsetup", "vars/*.go", r'json:"pastinitialsetup"'),
    ("POST /api/set_stt_info {language}", "wirepod/config-ws/*.go", r'case "set_stt_info"'),
    ("set_stt_info body: language", "wirepod/config-ws/*.go", r'Language string `json:"language"`'),
    ("GET /api/get_download_status", "wirepod/config-ws/*.go", r'case "get_download_status"'),
]


def check(root: Path) -> list[str]:
    pkg = root / "chipper" / "pkg"
    if not pkg.is_dir():
        return [f"not a wire-pod checkout: {pkg} missing"]
    missing = []
    for what, glob, pattern in CONTRACT:
        text = "\n".join(p.read_text(errors="ignore") for p in pkg.glob(glob))
        if not re.search(pattern, text):
            missing.append(what)
    return missing


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "wire-pod")
    missing = check(root)
    if missing:
        print("wire-pod is missing things Cloak needs:")
        for m in missing:
            print(f"  - {m}")
        return 1
    print(f"ok: all {len(CONTRACT)} checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
