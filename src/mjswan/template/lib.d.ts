/**
 * Type declarations for the mjswan library build (`dist/mjswan.js`).
 *
 * The default and named `mount` export embeds a published mjswan simulation
 * (rendered from its `config.json`) into a host element. See src/mount.tsx.
 */

/**
 * Render a published mjswan simulation into `element`.
 *
 * @param element  Host element to render the viewer (and controls) into.
 * @param configUrl  Absolute (or page-relative) URL of the simulation's
 *   `config.json`. Every other asset is resolved relative to its directory, so
 *   it works cross-origin.
 * @returns Resolves once the first scene is running; rejects on load failure.
 */
export function mount(element: HTMLElement, configUrl: string): Promise<void>;

/** Tear down a previously mounted simulation and free its resources. */
export function unmount(element: HTMLElement): void;

export default mount;
