/**
 * Types for the `mjswan/manifest` build (`dist/manifest.js`).
 *
 * `parseManifest(manifest, byteSource) -> Catalog` turns a build's `manifest.json`
 * plus a byte source into the typed catalog of `SceneInput`/`PolicyInput` the
 * engine consumes. Framework-free; shared by the in-repo app and mjswan Cloud.
 * See src/manifest/ and docs/adr/0004-headless-engine-core.md §1/§9.
 */
export * from './src/manifest/index';
