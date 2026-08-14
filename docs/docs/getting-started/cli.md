---
icon: octicons/terminal-16
---

# Command-line interface

`pip install mjswan` exposes a single Typer-based command:

```bash
mjswan --help
```

| Subcommand | Purpose |
|---|---|
| [`mjswan view`](#mjswan-view) | Build and launch a viewer for a MuJoCo XML / MJCF file |
| [`mjswan serve`](#mjswan-serve) | Serve a pre-built `dist/` directory |
| [`mjswan new`](#mjswan-new) | Scaffold a new project from a template |
| [`mjswan demo`](#mjswan-demo) | Run a built-in demo |
| [`mjswan info`](#mjswan-info) | Show a tree of projects / scenes / policies and asset sizes |
| [`mjswan publish`](#mjswan-publish) | Upload a built `dist/` to mjswan Cloud |
| [`mjswan login`](#mjswan-login-whoami-logout) | Sign in to mjswan Cloud via GitHub |
| [`mjswan whoami`](#mjswan-login-whoami-logout) | Show the signed-in account |
| [`mjswan logout`](#mjswan-login-whoami-logout) | Remove the stored session |

The legacy entry points `main`, `simple`, `mjlab`, and `serve <dist-dir>` are kept for backward compatibility — prefer the subcommands above for new scripts.

## `mjswan view`

```bash
mjswan view <model.xml> [--name SCENE] [--port 8080] [--host localhost] [--no-open]
```

Build a temporary app around the supplied MuJoCo XML/MJCF file and launch a local viewer.

| Option | Default | Description |
|---|---|---|
| `<model>` | — | Path to a MuJoCo XML / MJCF file. |
| `--name` | `Scene` | Display name shown in the viewer. |
| `--port` | `8080` | HTTP server port. |
| `--host` | `localhost` | HTTP bind address. |
| `--no-open` | `false` | Do not open the browser automatically. |

The build is written to a temporary directory and discarded when the server exits.

## `mjswan serve`

```bash
mjswan serve <dist-dir> [--port 8080] [--host localhost] [--no-open] [--height 600]
```

Serve a pre-built `dist/` directory with the COOP/COEP headers set. Use this to re-launch an app you previously built with `builder.build()`.

| Option | Default | Description |
|---|---|---|
| `<dist-dir>` | — | Path to a built mjswan `dist/` directory. |
| `--port` | `8080` | HTTP server port. |
| `--host` | `localhost` | HTTP bind address. |
| `--no-open` | `false` | Do not open the browser automatically. |
| `--height` | `600` | Colab iframe height in pixels (ignored outside Colab). |

## `mjswan new`

```bash
mjswan new <name> [--template hello-world|policy|mjlab]
```

Scaffold a new mjswan project in a directory named `<name>`.

| Template | What you get |
|---|---|
| `hello-world` (default) | A `main.py` that builds a single scene from `model.xml` and a minimal `model.xml`. |
| `policy` | The hello-world template plus an `add_policy(name="Policy", policy=...)` call expecting a `policy.onnx` file alongside `main.py`. |
| `mjlab` | A one-line `mjswan.Builder.from_mjlab("go2_flat").build()` script. Requires [`mjlab`](../guides/mjlab.md) installed. |

After scaffolding:

```bash
cd <name>
python main.py
```

## `mjswan demo`

```bash
mjswan demo [name] [--list]
```

Run one of the bundled demos under `examples/`.

| Demo | Source |
|---|---|
| `simple` (default) | `examples/demo/simple.py` |
| `main` | `examples/demo/main.py` (the demo deployed to GitHub Pages) |
| `mjlab` | `examples/mjlab/defaults/main.py` |

Use `mjswan demo --list` to enumerate them.

## `mjswan info`

```bash
mjswan info <dist-dir>
```

Print a tree summary of a built `dist/` directory: projects, scenes (with `.mjz`/`.mjb` size), and policies (with `.onnx` size). Useful for spotting which assets are pushing your build toward the GitHub Pages 1 GB limit.

Example output:

```
mjswan app — dist  v0.8.2
└── My Robots  [main]
    └── G1  scene.mjz  (3.2 MB)
        └── Policy: Locomotion  (1.1 MB)
Total scene+policy assets: 4.3 MB
```

## `mjswan publish`

```bash
mjswan publish <dist-dir> [--title TITLE] [--description TEXT] [--tag TAG]... \
                          [--token TOKEN] [--api-base URL]
```

Upload a built `dist/` directory's data files to [mjswan Cloud](../guides/publishing.md)
and print the resulting simulation URL. Only data files travel — `config.json`, the
scene/policy/motion/splat assets and traced graphs — never the compiled JavaScript, since
Cloud renders them with its own copy of the engine.

| Option | Default | Description |
|---|---|---|
| `<dist-dir>` | — | Path to a built mjswan `dist/` directory. |
| `--title` | first project's name | Simulation title. |
| `--description` | — | Optional description. |
| `--tag` | — | Tag to attach. Repeatable. |
| `--token` | `$MJSWAN_TOKEN` | Access token, for CI. Skips the interactive login. |
| `--api-base` | `$MJSWAN_API_BASE`, then `https://api.mjswan.com` | Cloud API base URL. |

If you are not signed in and no token is supplied, the browser login runs first
automatically. Builds that use custom-JavaScript MDP terms are rejected: Cloud cannot
execute author-supplied code.

## `mjswan login` / `whoami` / `logout`

```bash
mjswan login [--no-open]   # GitHub OAuth over a loopback redirect
mjswan whoami              # print the signed-in account
mjswan logout              # remove the stored session
```

`--no-open` prints the authorization URL instead of opening a browser — useful over SSH.
The session is stored locally; `mjswan whoami` exits non-zero when there is none, or when
one exists locally but is no longer valid server-side.
