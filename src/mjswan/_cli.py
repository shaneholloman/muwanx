"""CLI entry points for mjswan scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="mjswan",
    help="Browser-based MuJoCo simulation with real-time policy control.",
    no_args_is_help=True,
)
console = Console()


def _run_module(module_path: str) -> None:
    project_root = Path(__file__).parent.parent.parent
    result = subprocess.run(
        [sys.executable, "-m", module_path],
        check=False,
        cwd=project_root,
    )
    sys.exit(result.returncode)


def _fmt_size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


# ── view ──────────────────────────────────────────────────────


@app.command("view")
def view_cmd(
    model: Annotated[Path, typer.Argument(help="Path to MuJoCo XML/MJCF file.")],
    name: Annotated[
        str, typer.Option(help="Scene name shown in the viewer.")
    ] = "Scene",
    port: Annotated[int, typer.Option(help="HTTP server port.")] = 8080,
    host: Annotated[str, typer.Option(help="HTTP server host.")] = "localhost",
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Do not open browser automatically.")
    ] = False,
) -> None:
    """View a MuJoCo XML/MJCF file in the browser."""
    import tempfile

    import mujoco

    from mjswan import Builder

    if not model.exists():
        console.print(f"[red]Error:[/red] File not found: {model}")
        raise typer.Exit(1)

    spec = mujoco.MjSpec.from_file(str(model.resolve()))
    builder = Builder()
    builder.add_project(name=model.stem).add_scene(spec=spec, name=name)

    with tempfile.TemporaryDirectory() as tmp:
        built_app = builder.build(output_dir=tmp)
        built_app.launch(host=host, port=port, open_browser=not no_open)


# ── serve ─────────────────────────────────────────────────────


@app.command("serve")
def serve_cmd(
    dist_dir: Annotated[
        Path, typer.Argument(help="Path to a built mjswan dist directory.")
    ],
    port: Annotated[int, typer.Option(help="HTTP server port.")] = 8080,
    host: Annotated[str, typer.Option(help="HTTP server host.")] = "localhost",
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Do not open browser automatically.")
    ] = False,
    height: Annotated[int, typer.Option(help="Colab iframe height in pixels.")] = 600,
) -> None:
    """Serve a pre-built mjswan app from a dist directory."""
    from mjswan.app import mjswanApp

    resolved = dist_dir.resolve()
    if not resolved.exists():
        console.print(f"[red]Error:[/red] Directory not found: {dist_dir}")
        raise typer.Exit(1)

    mjswanApp(resolved).launch(
        host=host, port=port, open_browser=not no_open, height=height
    )


# ── publish ───────────────────────────────────────────────────


@app.command("publish")
def publish_cmd(
    dist_dir: Annotated[
        Path, typer.Argument(help="Path to a built mjswan dist directory.")
    ],
    title: Annotated[
        Optional[str],
        typer.Option(help="Simulation title. Defaults to the first project's name."),
    ] = None,
    description: Annotated[
        Optional[str], typer.Option(help="Optional description.")
    ] = None,
    tag: Annotated[
        Optional[list[str]],
        typer.Option(help="Tag to attach (repeatable)."),
    ] = None,
    token: Annotated[
        Optional[str],
        typer.Option(help="Supabase access token. Falls back to $MJSWAN_TOKEN."),
    ] = None,
    api_base: Annotated[
        Optional[str],
        typer.Option(
            help="Cloud API base URL. Falls back to $MJSWAN_API_BASE, then "
            "https://api-v2.mjswan.com."
        ),
    ] = None,
) -> None:
    """Publish a built dist directory's data files to mjswan Cloud."""
    from mjswan.publish import TOKEN_ENV_VAR, PublishError, publish_dist

    resolved = dist_dir.expanduser().resolve()
    if not resolved.exists():
        console.print(f"[red]Error:[/red] Directory not found: {dist_dir}")
        raise typer.Exit(1)

    # Auto-login when there is no token to use (no flag, no env, no stored
    # session). The browser flow runs first, then publish proceeds normally.
    import os

    from mjswan import auth

    if not token and not os.environ.get(TOKEN_ENV_VAR) and not auth.load_credentials():
        console.print("[dim]Not logged in — signing in to mjswan Cloud first…[/dim]")
        if not _do_login(open_browser=True):
            raise typer.Exit(1)

    try:
        result = publish_dist(
            resolved,
            title=title,
            description=description,
            tags=list(tag) if tag else None,
            token=token,
            api_base=api_base,
            on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        )
    except PublishError as exc:
        location = f" [dim]({exc.file})[/dim]" if exc.file else ""
        console.print(f"[red]Publish failed:[/red] {exc}{location}")
        raise typer.Exit(1)

    console.print(f"[green]Published![/green] Simulation id: [bold]{result.id}[/bold]")


# ── login / logout / whoami ────────────────────────────────────


def _do_login(*, open_browser: bool) -> bool:
    """Run the OAuth flow and report which account signed in.

    Returns ``True`` on success, ``False`` (with an error printed) on failure.
    Shared by ``mjswan login`` and ``mjswan publish``'s auto-login.
    """
    from mjswan.auth import AuthError, login

    try:
        creds = login(
            open_browser=open_browser,
            on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        )
    except AuthError as exc:
        console.print(f"[red]Login failed:[/red] {exc}")
        return False

    who = f" as [bold]{creds.username}[/bold]" if creds.username else ""
    console.print(f"[green]Logged in to mjswan Cloud{who}.[/green]")
    return True


@app.command("login")
def login_cmd(
    no_open: Annotated[
        bool,
        typer.Option(
            "--no-open", help="Do not open the browser; print the URL instead."
        ),
    ] = False,
) -> None:
    """Sign in to mjswan Cloud via GitHub (loopback OAuth)."""
    if not _do_login(open_browser=not no_open):
        raise typer.Exit(1)


@app.command("whoami")
def whoami_cmd() -> None:
    """Show the mjswan Cloud account you are signed in as."""
    from mjswan.auth import AuthError, fetch_identity

    try:
        identity = fetch_identity()
    except AuthError as exc:
        # Session exists locally but is no longer valid server-side.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if identity is None:
        console.print("[dim]Not logged in. Run [bold]mjswan login[/bold].[/dim]")
        raise typer.Exit(1)

    name = identity.username or identity.user_id
    detail = f" [dim]({identity.email})[/dim]" if identity.email else ""
    console.print(f"Logged in as [bold]{name}[/bold]{detail}")


@app.command("logout")
def logout_cmd() -> None:
    """Remove the stored mjswan Cloud session."""
    from mjswan.auth import clear_credentials, credentials_path

    if clear_credentials():
        console.print("[green]Logged out.[/green]")
    else:
        console.print(
            f"[dim]Not logged in (no credentials at {credentials_path()}).[/dim]"
        )


# ── new ───────────────────────────────────────────────────────

_TEMPLATES: dict[str, dict[str, str]] = {
    "hello-world": {
        "main.py": """\
import mujoco

import mjswan


def main() -> None:
    builder = mjswan.Builder()
    project = builder.add_project(name="{name}")

    spec = mujoco.MjSpec.from_file("model.xml")
    project.add_scene(spec=spec, name="Scene")

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
""",
        "model.xml": """\
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
""",
    },
    "policy": {
        "main.py": """\
import mujoco
import onnx

import mjswan


def main() -> None:
    builder = mjswan.Builder()
    project = builder.add_project(name="{name}")

    spec = mujoco.MjSpec.from_file("model.xml")
    scene = project.add_scene(spec=spec, name="Scene")

    # Replace with your ONNX policy file
    policy_model = onnx.load("policy.onnx")
    scene.add_policy(policy=policy_model, name="Policy")

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
""",
        "model.xml": """\
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
""",
    },
    "mjlab": {
        "main.py": """\
import mjswan


def main() -> None:
    # Replace "go2_flat" with your mjlab task ID
    app = mjswan.Builder.from_mjlab("go2_flat").build()
    app.launch()


if __name__ == "__main__":
    main()
""",
    },
}


@app.command("new")
def new_cmd(
    name: Annotated[
        str, typer.Argument(help="Project name (also used as directory name).")
    ],
    template: Annotated[
        str, typer.Option(help="Template to use: hello-world | policy | mjlab.")
    ] = "hello-world",
) -> None:
    """Scaffold a new mjswan project from a template."""
    if template not in _TEMPLATES:
        console.print(
            f"[red]Error:[/red] Unknown template '{template}'. "
            f"Available: {', '.join(_TEMPLATES)}"
        )
        raise typer.Exit(1)

    project_dir = Path(name)
    if project_dir.exists():
        console.print(f"[red]Error:[/red] Directory '{name}' already exists.")
        raise typer.Exit(1)

    project_dir.mkdir()
    for filename, content in _TEMPLATES[template].items():
        (project_dir / filename).write_text(content.format(name=name))
        console.print(f"  [green]created[/green]  {name}/{filename}")

    console.print(f"\n[bold]Done![/bold] Start with:\n  cd {name}\n  python main.py")


# ── demo ──────────────────────────────────────────────────────

_DEMOS: dict[str, str] = {
    "simple": "examples.demo.simple",
    "main": "examples.demo.main",
    "mjlab": "examples.mjlab.defaults.main",
}


@app.command("demo")
def demo_cmd(
    name: Annotated[
        Optional[str], typer.Argument(help="Demo name. Omit to run 'simple'.")
    ] = None,
    list_: Annotated[
        bool, typer.Option("--list", "-l", help="List available demos.")
    ] = False,
) -> None:
    """Run a built-in mjswan demo."""
    if list_:
        console.print("[bold]Available demos:[/bold]")
        for demo_name in _DEMOS:
            console.print(f"  {demo_name}")
        return

    demo_name = name or "simple"
    if demo_name not in _DEMOS:
        console.print(
            f"[red]Error:[/red] Unknown demo '{demo_name}'. "
            "Run [bold]mjswan demo --list[/bold] to see available demos."
        )
        raise typer.Exit(1)

    _run_module(_DEMOS[demo_name])


# ── info ──────────────────────────────────────────────────────


@app.command("info")
def info_cmd(
    dist_dir: Annotated[
        Path, typer.Argument(help="Path to a built mjswan dist directory.")
    ],
) -> None:
    """Show information about a built mjswan app."""
    import json

    from rich.tree import Tree

    from mjswan.utils import name2id

    config_path = dist_dir / "assets" / "config.json"
    if not config_path.exists():
        console.print(f"[red]Error:[/red] No assets/config.json found in {dist_dir}")
        raise typer.Exit(1)

    config = json.loads(config_path.read_text())
    version = config.get("version", "unknown")

    tree = Tree(f"[bold]mjswan app[/bold] — {dist_dir}  [dim]v{version}[/dim]")

    total_bytes = 0
    for project in config.get("projects", []):
        project_dir_name = project.get("id") or "main"
        p_node = tree.add(
            f"[cyan]{project['name']}[/cyan]  [dim][{project_dir_name}][/dim]"
        )
        for scene in project.get("scenes", []):
            scene_rel = scene.get("path", "")
            scene_path = dist_dir / project_dir_name / "assets" / scene_rel
            scene_size = scene_path.stat().st_size if scene_path.exists() else 0
            total_bytes += scene_size
            s_node = p_node.add(
                f"[green]{scene['name']}[/green]  "
                f"[dim]{scene_rel}  ({_fmt_size(scene_size)})[/dim]"
            )
            for policy in scene.get("policies", []):
                onnx_path = scene_path.parent / f"{name2id(policy['name'])}.onnx"
                policy_size = onnx_path.stat().st_size if onnx_path.exists() else 0
                total_bytes += policy_size
                size_str = f"  ({_fmt_size(policy_size)})" if policy_size else ""
                s_node.add(
                    f"Policy: [yellow]{policy['name']}[/yellow][dim]{size_str}[/dim]"
                )

    tree.add(f"[dim]Total scene+policy assets: {_fmt_size(total_bytes)}[/dim]")
    console.print(tree)


# ── Legacy entry points (backward compatibility) ──────────────


def main() -> None:
    """Run examples/demo/main.py"""
    _run_module("examples.demo.main")


def simple() -> None:
    """Run examples/demo/simple.py"""
    _run_module("examples.demo.simple")


def mjlab() -> None:
    """Run examples/mjlab/defaults/main.py"""
    _run_module("examples.mjlab.defaults.main")


def serve() -> None:
    """Launch a pre-built mjswan app from a dist directory.

    Usage: serve <dist-dir>
    """
    if len(sys.argv) < 2:
        print("Usage: serve <dist-dir>", file=sys.stderr)
        sys.exit(1)

    from mjswan.app import mjswanApp

    mjswan_app = mjswanApp(Path(sys.argv[1]).resolve())
    mjswan_app.launch()
