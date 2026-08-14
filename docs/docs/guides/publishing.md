---
icon: octicons/cloud-16
---

# Publishing to mjswan Cloud

[Deployment](deployment.md) covers hosting a built `dist/` yourself. mjswan Cloud is the
alternative: upload only the *data* files and get a hosted page back, with no static host
to configure and no `base_path` to get right.

!!! note "Beta"
    mjswan Cloud is in beta. The `publish` verb, the token env vars, and the URL shape are
    stable enough to script against; the hosted UI is still moving.

## What gets uploaded

Only data files travel — the compiled JavaScript never does. Cloud renders your
simulation with its own pinned copy of the engine, loaded from a CDN.

| Uploaded | Not uploaded |
|---|---|
| `assets/config.json` | `index.html`, `logo.svg`, `manifest.json`, `robots.txt` |
| `scene.mjz` / `scene.mjb` | the compiled JS/CSS bundle and its WASM |
| `<policy>.onnx`, `<policy>.json` | `_headers`, `coi-serviceworker.js` |
| traced graphs (`obs/`, `term/`, `command/`, `event/`) | |
| `<motion>.npz`, `<splat>.spz`, `.ply` colliders | |

Limits: 50 MB per file, 200 MB total, 64 files.

!!! warning "Custom-JavaScript builds are rejected"
    A build whose `config.json` carries `uses_custom_js: true` — one using a
    `*Binding` with `ts_src`, i.e. an author-written TypeScript term class — cannot be
    published. Cloud will not execute author-supplied code in its own origin. Traced ONNX
    terms are fine; they are inert data run by a fixed runtime. This is checked locally
    before any upload starts.

## From the CLI

```bash
python build.py                # writes dist/
mjswan publish dist --title "G1 Locomotion"
```

The first `publish` runs a GitHub sign-in automatically (a `gh`-style loopback OAuth flow)
and prints the resulting page URL:

```
Published! https://mjswan.com/s/<id>
```

Manage the session explicitly when you want to:

```bash
mjswan login             # or --no-open to print the URL instead, e.g. over SSH
mjswan whoami
mjswan logout
```

## From Python

`Builder.build()` returns an `MjswanApp`, which publishes directly — no intermediate
`dist/` path to pass around:

```python
app = builder.build()
result = app.publish(title="G1 Locomotion", tags=["locomotion", "g1"])
print(result.id)
```

## In CI

Interactive login has no place in a pipeline. Set `MJSWAN_TOKEN` instead and the browser
flow is skipped:

```yaml
- name: Build and publish
  run: |
    uv run python build.py
    uv run mjswan publish dist --title "Nightly"
  env:
    MJSWAN_TOKEN: ${{ secrets.MJSWAN_TOKEN }}
    MJSWAN_NO_LAUNCH: "1"
```

| Variable | Effect |
|---|---|
| `MJSWAN_TOKEN` | Access token. Equivalent to `--token`; skips the interactive login. |
| `MJSWAN_API_BASE` | Cloud API base URL. Defaults to `https://api.mjswan.com`. |
| `MJSWAN_WEB_BASE` | Web app base used to build the printed page URL. Defaults to `https://mjswan.com`. |

## Embedding a published simulation

A published page can be embedded like any other, and the npm engine can load a published
`config.json` directly. See [Embedding](embedding.md) and the
[Engine API](../api/engine.md).
