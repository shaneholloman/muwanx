/**
 * Normalize a scene path for consistent caching
 *
 * This function ensures consistent path normalization across the codebase:
 * - Trims whitespace
 * - Removes leading "./" patterns
 * - Collapses multiple consecutive slashes
 *
 * @param scenePath The path to normalize
 * @returns Normalized path
 */
export function normalizeScenePath(scenePath: string): string {
  return scenePath
    .trim()
    .replace(/^(\.\/)+/, '')
    .replace(/\/+/g, '/');
}

/**
 * Collapse runs of consecutive slashes without mangling a URL scheme.
 *
 * `https://cdn.mjswan.com//a//b` → `https://cdn.mjswan.com/a/b`, while a
 * relative/path base such as `/mjswan//a` collapses to `/mjswan/a`. The `://`
 * after a scheme is preserved because the doubled slash there follows a colon.
 *
 * This matters for the cross-origin `mount()` library build, where the asset
 * base is a fully-qualified URL (data on `cdn.mjswan.com`, page on
 * `mjswan.com`) rather than the same-origin path the SPA build uses.
 */
export function collapseSlashes(input: string): string {
  return input.replace(/([^:])\/{2,}/g, '$1/').replace(/^\/{2,}/, '/');
}

/** Join a base (path or absolute URL) with a relative path, slash-safe. */
export function joinUrl(base: string, path: string): string {
  return collapseSlashes(`${base}/${path}`);
}
