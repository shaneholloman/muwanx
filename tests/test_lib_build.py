"""Tests for the Vite library build (`mjswan.js` mount entry).

L3 slow (triggers a frontend build): TestLibBuild
Run with: pytest -m slow -k LibBuild

The library build produces a single self-contained ESM consumed by mjswan
Cloud from a CDN. These tests enforce the load-bearing invariants:
  - `dist/mjswan.js` exists and exports `mount` (and `unmount`/default),
  - every dependency is bundled (no bare imports left to resolve from a CDN),
  - the MuJoCo/ONNX WASM is emitted as co-located files (NOT inlined as base64
    data URLs) and referenced relative to the bundle.
See vite.lib.config.ts and mjswan-cloud ADR 0001.
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
    # The core imports these generated modules; ensure they exist (empty stubs
    # when no custom terms are registered).
    builder.generate_custom_observations()
    builder.generate_custom_commands()
    builder.generate_custom_events()
    builder.generate_custom_terminations()
    builder.generate_viewer_config_defaults()
    builder.install_dependencies()
    builder.run_build_script("build:lib")
    return TEMPLATE_DIR / "dist"


@pytest.mark.slow
class TestLibBuild:
    def test_mjswan_js_emitted(self, lib_dist: Path):
        assert (lib_dist / "mjswan.js").is_file()

    def test_exports_mount(self, lib_dist: Path):
        code = (lib_dist / "mjswan.js").read_text()
        assert re.search(r"\bas mount\b", code) or re.search(
            r"export\s*\{[^}]*\bmount\b", code
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
        wasm_files = list(lib_dist.glob("*.wasm"))
        assert wasm_files, "no co-located .wasm files emitted in dist/"
        # No multi-MB base64 WASM left inlined in any JS file.
        for js in lib_dist.glob("*.js"):
            assert "data:application/wasm" not in js.read_text(), (
                f"{js.name} still contains an inlined WASM data URL"
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

    def test_styles_inlined(self, lib_dist: Path):
        # The CSS is injected by JS so a single import brings its own styles.
        assert "mjswan-styles" in (lib_dist / "mjswan.js").read_text()
