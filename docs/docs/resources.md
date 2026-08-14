---
icon: octicons/link-16
---

# Resources

## mjswan — MyoConference 2026 Presentation

<iframe
  src="https://www.youtube.com/embed/0B4Ky0hf2Gg"
  title="mjswan MyoConference Presentation"
  loading="lazy"
  frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  referrerpolicy="strict-origin-when-cross-origin"
  allowfullscreen
  style="width: 100%; aspect-ratio: 16 / 9; border: 0; border-radius: 0.4rem;">
</iframe>

## Foundations

[**MuJoCo WASM**](https://github.com/google-deepmind/mujoco/tree/main/wasm) — The official MuJoCo WebAssembly build that mjswan uses at its core.

[**mjlab**](https://github.com/mujocolab/mjlab) — GPU-accelerated robot-learning framework. mjswan visualizes its tasks directly and traces its MDP functions to ONNX; see [Using mjlab](guides/mjlab.md).

[**ONNX Runtime Web**](https://github.com/microsoft/onnxruntime) — Runs both the policy network and the traced MDP term bodies in the browser.

[**three.js**](https://github.com/mrdoob/three.js) — Rendering, including reflections, shadows, Gaussian splat backgrounds, and WebXR.

[**Facet**](https://facet.pages.dev/#/) — Browser-based MuJoCo viewer from Tsinghua University that originally inspired mjswan.

[**zalo/mujoco_wasm**](https://github.com/zalo/mujoco_wasm) — One of the earliest efforts to run MuJoCo in a browser, and the groundwork mjswan builds on.

## Similar tools

[**Viser**](https://viser.studio/main/) — Python 3D visualisation library with a web-based viewer. Server-based rather than static; useful when you need a live Python backend.

[**Mels.ai**](https://research.mels.ai/ide?mels=UnitreeG1.3gwz1) — Web IDE for MuJoCo Playground environments.

[**Real-time Web Viewer for Brax**](https://github.com/pal-robotics/brax_training_viewer) — FastAPI-based approach to streaming Brax simulations to the browser (Google Summer of Code 2025).

## Citation

```bibtex
@software{mjswan,
  author = {Tsujimoto, Tatsuki},
  title = {{mjswan: MuJoCo Simulation on Web Assembly with Neural networks}},
  url = {https://github.com/ttktjmt/mjswan},
  license = {Apache-2.0}
}
```

[CITATION.cff](https://github.com/ttktjmt/mjswan/blob/main/CITATION.cff) is the
authoritative metadata.
