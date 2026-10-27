import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("contract", Path(__file__).parents[1] / "scripts" / "wirepod_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)

GOOD = {
    "wirepod/sdkapp/server.go": 'case r.URL.Path == "/api-sdk/get_sdk_info":',
    "vars/vars.go": 'GlobalGUID string `json:"global_guid"`\nRobots []struct{} `json:"robots"`\nEsn `json:"esn"` '
                    'IPAddress `json:"ip_address"` GUID `json:"guid"`',
    "vars/config.go": 'PastInitialSetup bool `json:"pastinitialsetup"`',
    "wirepod/config-ws/webserver.go": 'case "is_running":\ncase "get_config":\ncase "set_stt_info":\n'
                                      'case "get_download_status":\nLanguage string `json:"language"`\n"/session-certs/"',
}


def make(tmp, files):
    for rel, text in files.items():
        p = tmp / "chipper" / "pkg" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp


def test_contract_passes_and_catches_breaks(tmp_path):
    assert contract.check(make(tmp_path / "ok", GOOD)) == []
    broken = dict(GOOD, **{"vars/vars.go": GOOD["vars/vars.go"].replace('json:"guid"', 'json:"token"')})
    assert contract.check(make(tmp_path / "bad", broken)) == ["bot list JSON: guid"]
    assert contract.check(tmp_path / "nothing")[0].startswith("not a wire-pod checkout")
