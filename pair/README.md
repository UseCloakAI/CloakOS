# Pairing (vector-web-setup, Cloak edition)

`src/` is the Bluetooth setup code from Digital Dream Labs' **vector-web-setup** (MIT, see
`LICENSE`), taken from kercre123's fork at commit `b33baf1`. It's bundled into
`cloak_vector/web/static/pair/rts.js` and driven by Cloak's own page,
`cloak_vector/web/static/pair.html`.

Changes from upstream (search `CLOAK:` in `src/main.js`):
- settings load from `/static/pair/settings.json` (one stack, so the stack picker is skipped)
- the firmware URL comes from `?otaUrl=`, which Cloak's `/pair` route fills with wire-pod's escape-pod OTA
  (`http://<wire-pod LAN IP>:8080/api/get_ota/vicos-2.0.1.6076ep.ota`, the same one wire-pod uses)
- "sign in" is replaced by wire-pod activation: the fixed session token wire-pod itself uses
  (`chipper/pkg/wirepod/setup/ble.go` → `AuthRobot`), retried 4× like wire-pod does
- Wi-Fi network names are escaped before they're put in the page

Rebuild after editing `src/`:

```bash
scripts/build_pair.sh
```
