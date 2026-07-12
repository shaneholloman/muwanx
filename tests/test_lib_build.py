"""Tests for the Vite library build (`mjswan.js` createEngine entry).

L3 slow (triggers a frontend build): TestLibBuild
Run with: pytest -m slow -k LibBuild

The library build produces a single self-contained ESM (the headless engine,
no React/Mantine) consumed by mjswan Cloud from a CDN. These tests enforce the
load-bearing invariants:
  - `dist/mjswan.js` exists and exports `createEngine` (and default),
  - every dependency is bundled (no bare imports left to resolve from a CDN),
  - the MuJoCo/ONNX WASM is emitted as co-located files (NOT inlined as base64
    data URLs) and referenced relative to the bundle.
See vite.lib.config.ts and docs/adr/0004-headless-engine-core.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import mjswan
from mjswan._build_client import ClientBuilder

TEMPLATE_DIR = Path(mjswan.__file__).parent / "template"

# Real ESM `import ... from "spec"` / `export ... from "spec"` and `import("spec")`,
# anchored at a statement boundary so matches inside string literals (e.g. React's
# "import it from \"react-dom/client\"" warning) are not mistaken for imports.
_IMPORT_FROM = re.compile(
    r"""(?:^|[;}\n])\s*(?:import|export)\b[^;{}\n]*?\bfrom\s*["']([^"']+)["']"""
)
_DYNAMIC_IMPORT = re.compile(
    r"""(?:^|[;}\n=(,:?&|])\s*import\(\s*["']([^"']+)["']\s*\)"""
)


def _is_bare(spec: str) -> bool:
    return not (
        spec.startswith("./")
        or spec.startswith("../")
        or spec.startswith("data:")
        or spec.startswith("http://")
        or spec.startswith("https://")
    )


@pytest.fixture(scope="class")
def lib_dist() -> Path:
    """Build only the library bundle once for the whole test class."""
    builder = ClientBuilder(TEMPLATE_DIR)
    builder.create_env()
    builder.sync_version_from_python()
    # The core imports these generated modules; ensure they exist as empty stubs
    # (custom terms load at runtime via plugins.js, not the engine bundle).
    builder.generate_empty_custom_stubs()
    builder.generate_viewer_config_defaults()
    builder.install_dependencies()
    builder.run_build_script("build:lib")
    return TEMPLATE_DIR / "dist"


@pytest.mark.slow
class TestLibBuild:
    def test_mjswan_js_emitted(self, lib_dist: Path):
        assert (lib_dist / "mjswan.js").is_file()

    def test_exports_create_engine(self, lib_dist: Path):
        code = (lib_dist / "mjswan.js").read_text()
        assert re.search(r"\bas createEngine\b", code) or re.search(
            r"export\s*\{[^}]*\bcreateEngine\b", code
        )

    def test_no_bare_imports(self, lib_dist: Path):
        """No `import 'three'`-style specifiers — all resolvable from the CDN."""
        offenders: list[str] = []
        for js in lib_dist.glob("*.js"):
            code = js.read_text()
            specs = _IMPORT_FROM.findall(code) + _DYNAMIC_IMPORT.findall(code)
            offenders.extend(f"{js.name} -> {spec}" for spec in specs if _is_bare(spec))
        assert not offenders, f"bare imports found: {offenders}"

    def test_wasm_co_located_not_inlined(self, lib_dist: Path):
        """Main-thread WASM is co-located, never inlined as a base64 data URL.

        `extractInlinedWasmPlugin` in vite.lib.config.ts pulls every
        `new URL('data:application/wasm…', import.meta.url)` back out into a
        flat dist/ file. It deliberately leaves ONE class inlined: Spark's
        Gaussian Splat sorter runs in a classic Blob worker whose base is
        `self.location.href`, where `import.meta` is a syntax error — that
        dormant single-threaded worker keeps its base64 WASM. So forbid only
        `import.meta.url`-based (MuJoCo/ONNX main-thread) inlining.
        """
        wasm_files = list(lib_dist.glob("*.wasm"))
        assert wasm_files, "no co-located .wasm files emitted in dist/"
        inlined = re.compile(
            r"""new URL\(\s*(["'`])data:application/wasm;base64,"""
            r"""[A-Za-z0-9+/=]+\1\s*,\s*([^)]*)\)"""
        )
        for js in lib_dist.glob("*.js"):
            for m in inlined.finditer(js.read_text()):
                base = m.group(2)
                assert "import.meta.url" not in base, (
                    f"{js.name} still inlines main-thread WASM as a data URL "
                    f"(base {base!r}); it must be extracted to a co-located file"
                )

    def test_wasm_referenced_relative_to_bundle(self, lib_dist: Path):
        """WASM is fetched via `new URL('./x.wasm', import.meta.url)`."""
        found = False
        for js in lib_dist.glob("*.js"):
            if re.search(
                r"""new URL\(\s*["'`]\./[^"'`]*\.wasm["'`]\s*,\s*import\.meta\.url""",
                js.read_text(),
            ):
                found = True
                break
        assert found, "no co-located `new URL('./*.wasm', import.meta.url)` reference"

    def test_no_bundled_react_or_mantine(self, lib_dist: Path):
        # The engine entry drops the React/Mantine chrome; the CDN bundle must
        # not carry it (ADR 0004 §11). A cheap proxy: React's dev-warning prefix.
        code = (lib_dist / "mjswan.js").read_text()
        assert "Warning: React" not in code and "@mantine" not in code

    def test_no_unfolded_process_env_node_env(self, lib_dist: Path):
        """The bundle must be browser-self-contained: no unfolded `process`.

        mjswan Cloud loads this bundle straight from a CDN with `@vite-ignore`,
        so the consuming bundler never substitutes globals away. An eager,
        unguarded `process.env.NODE_ENV` (shipped by React/Mantine dev checks)
        therefore throws `ReferenceError: process is not defined` at mount time
        in the browser. `vite.lib.config.ts` statically folds it to
        "production"; this asserts none survives. The engine — not the host —
        owns this invariant (the host must not need a `process` shim). See
        vite.lib.config.ts `define` and mjswan-cloud ADR 0001.

        Residual `process.*` references (e.g. setimmediate's `process.nextTick`)
        are allowed only because they sit behind runtime guards and never
        evaluate single-threaded in the browser.
        """
        offenders = [
            js.name
            for js in lib_dist.glob("*.js")
            if "process.env.NODE_ENV" in js.read_text()
        ]
        assert not offenders, (
            f"unfolded process.env.NODE_ENV in {offenders}; the Vite `define` "
            "in vite.lib.config.ts must fold it to a literal so the CDN-loaded "
            "engine needs no host-side `process` shim."
        )
