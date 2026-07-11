/**
 * Public entry for the headless mjswan engine. See docs/adr/0004-headless-engine-core.md.
 *
 * `createEngine(element, options?)` prepares a pure-TS, instance-scoped engine
 * (MuJoCo WASM + WebGL, no React) and exposes bytes-in / snapshot-out verbs.
 */
export * from './types';
export { createEngine } from './createEngine';
export { createEngine as default } from './createEngine';
