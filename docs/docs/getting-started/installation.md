---
icon: octicons/download-16
---

# Installation

mjswan can be installed as a Python package (the primary workflow) or as an npm package for JavaScript/TypeScript projects.

<div class="grid cards" markdown>

-   [:simple-python: &nbsp; __Python package__](#python-installation){ style="text-decoration: none; color: inherit;" }

    ---

    Install via pip to quickly build and share interactive MuJoCo simulations

-   [:simple-javascript: &nbsp; __JavaScript package__](#javascript-installation){ style="text-decoration: none; color: inherit;" }

    ---

    Install via npm for custom web applications with TypeScript support

-   [:simple-github: &nbsp; __GitHub Source__](#github-source){ style="text-decoration: none; color: inherit;" }

    ---

    Clone the repository for development and contributing

>   :simple-docker: &nbsp; __Docker / Cluster__
>   ---
>   Not supported.

</div>

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 – 3.12 (3.13+ not yet supported) |
| Platform | macOS (Apple Silicon) or Linux (x86-64) |
| Browser | Any modern browser with WebAssembly and WebGL |
| Node.js | 24+ (npm installation only) |

!!! note "Python 3.13"
    A transitive dependency (`labmaze`, pulled in via MyoSuite) does not yet publish a Python 3.13 wheel. Until it does, mjswan requires Python ≤ 3.12.

## Python Installation

```bash
pip install mjswan
```

That is everything needed to bundle MuJoCo models — `mujoco`, `onnx`, `typer`, `rich`,
`wandb`, and `nodeenv` for the frontend build. Extra dependency sets:

```bash
pip install 'mjswan[dev]'       # type checking, linting, test tools
pip install 'mjswan[examples]'  # mjlab, torch, MyoSuite, Playground, …
```

!!! warning "Policies with MDP terms need the `examples` extra"
    mjswan compiles observation, termination, event and command terms to ONNX at build
    time, which runs `torch.onnx.export` against a live mjlab environment — so a policy
    carrying any of those needs `mjlab` and `torch` installed. Both are **build-time
    only**; neither ships to the browser, and a model-only scene needs neither. See
    [How the Build Works](../guides/how-it-works.md).

    ```bash
    pip install 'mjswan[examples]'
    ```

The `examples` extra also pulls in MyoSuite, MuJoCo Playground, `robot_descriptions`,
`onnxruntime` (for the numeric parity checks), and `gymnasium`. It can take several
minutes to install and requires Python ≤ 3.12.

## JavaScript Installation

```bash
npm install mjswan
```

Or with yarn:

```bash
yarn add mjswan
```

The npm package provides the browser-side runtime — MuJoCo WASM, three.js rendering, and
ONNX Runtime Web behind a `createEngine` API. It is independent of the Python package and
has no Python dependency. Use it to embed a simulation in an app you own, or to author
custom MDP terms in TypeScript; see the [Engine API](../api/engine.md).

## GitHub Source

Clone the repository and install all dependencies with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/ttktjmt/mjswan.git
cd mjswan
uv sync --all-extras
```

To run the bundled demo after cloning:

```bash
mjswan demo          # runs the default demo
mjswan demo --list   # see all available demos
```

Common Makefile targets while developing:

| Target | What it does |
|---|---|
| `make sync` | Install/refresh all dependencies with `uv` |
| `make check` | Lint, format check, and type check |
| `make test` | Full pytest suite (`test-all` runs `check` first) |
| `make docs-serve` | Live-reloading documentation server |

See [docs/README.md](https://github.com/ttktjmt/mjswan/blob/main/docs/README.md){:target="_blank"} for the documentation workflow.
