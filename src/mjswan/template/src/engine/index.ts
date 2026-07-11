/**
 * Public entry for the headless mjswan engine. See docs/adr/0004-headless-engine-core.md.
 *
 * Phase 0 (issue #76) establishes the type contract below; `createEngine` is
 * implemented in Phase 1, when the runtime (currently React-driven via
 * MjswanViewer/MountApp) is lifted into this pure-TS engine.
 */
export * from './types';
import type { CreateEngineOptions, MjswanEngine } from './types';

/**
 * Prepare an engine (MuJoCo WASM + WebGL) in `element`, then `loadScene(...)`.
 *
 * ponytail: stub — implemented in Phase 1 (issue #76). Kept so the public
 * signature and exports are fixed for downstream layers to build against.
 */
export async function createEngine(
  _element: HTMLElement,
  _options?: CreateEngineOptions,
): Promise<MjswanEngine> {
  throw new Error('mjswan.createEngine: not implemented yet (Phase 1 — issue #76).');
}

export default createEngine;
