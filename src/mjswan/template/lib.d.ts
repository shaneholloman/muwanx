/**
 * Type declarations for the mjswan library build (`dist/mjswan.js`).
 *
 * The default and named `mount` export embeds a published mjswan simulation
 * into a host element, from either a config.json URL (production watch/embed) or
 * an in-memory file resolver (mjswan Cloud's upload preview). See src/mount.tsx
 * and mjswan-cloud ADR 0005.
 */

/** Resolve a scene-relative path (e.g. 'config.json', 'main/scene.mjz') to bytes, or null if absent. */
export type MjswanFileResolver = (path: string) => Promise<ArrayBuffer | null>;

/** Where `mount` reads scene data from: a config.json URL, or an in-memory resolver. */
export type MjswanSource = string | { resolve: MjswanFileResolver };

/** A mounted simulation instance. */
export interface MjswanInstance {
  /** Capture the current frame as a JPEG Blob (renders + reads back synchronously). */
  captureThumbnail: (options?: { maxDim?: number; quality?: number }) => Promise<Blob>;
  /** Tear down the simulation, free resources, and detach any local file source. */
  dispose: () => void;
}

/**
 * Render a published mjswan simulation into `element`.
 *
 * @param element  Host element to render the viewer (and controls) into.
 * @param source  A config.json URL whose directory all assets resolve against
 *   (works cross-origin), or a `{ resolve(path) }` file resolver that returns
 *   locally-selected bytes for each scene-relative path.
 * @returns Resolves, once the first scene is running, to an {@link MjswanInstance};
 *   rejects on load failure.
 */
export function mount(element: HTMLElement, source: MjswanSource): Promise<MjswanInstance>;

/** Tear down a previously mounted simulation and free its resources. */
export function unmount(element: HTMLElement): void;

export default mount;
