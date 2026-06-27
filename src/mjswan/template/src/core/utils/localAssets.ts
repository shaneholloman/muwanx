/**
 * In-browser asset source for `mount()`: lets the engine render a simulation
 * from locally-selected files (mjswan Cloud's upload preview) instead of a CDN
 * URL, without touching the scattered `fetch(baseUrl + path)` asset-loading code.
 *
 * A resolver is registered under a unique synthetic origin
 * (`https://mjswan.local/<token>/`), which `mount()` hands down as the `baseUrl`.
 * Every existing asset fetch then produces an absolute URL under that origin; a
 * scoped `window.fetch` wrapper intercepts exactly those URLs and serves bytes
 * from the resolver, delegating everything else untouched. The wrapper is
 * installed only while at least one local source is mounted and restored once
 * none remain (ref-counted), so the host page's `fetch` is otherwise unaffected.
 * All scene assets (config.json, .mjz, .onnx, .npz, policy/manifest JSON) load
 * through `fetch`, so this single choke point covers them. See mjswan-cloud ADR 0005.
 */

/** Resolve a scene-relative path (e.g. 'config.json', 'main/scene.mjz') to bytes, or null if absent. */
export type MjswanFileResolver = (path: string) => Promise<ArrayBuffer | null>;

const LOCAL_ORIGIN = 'https://mjswan.local';
const resolvers = new Map<string, MjswanFileResolver>();
let originalFetch: typeof window.fetch | null = null;
let tokenSeq = 0;

function makeToken(): string {
  tokenSeq += 1;
  // crypto.randomUUID isn't guaranteed on http: origins; a per-page counter is
  // enough to keep concurrent local mounts from colliding.
  return typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `t${tokenSeq}`;
}

/** Extract the registration token from a synthetic URL (base or asset), or null. */
function tokenOf(url: string): string | null {
  const match = url.match(/^https:\/\/mjswan\.local\/([^/]+)\//);
  return match ? match[1] : null;
}

function installInterceptor(): void {
  if (originalFetch) return;
  originalFetch = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const token = tokenOf(url);
    if (token === null) return originalFetch!(input, init);

    const resolve = resolvers.get(token);
    if (!resolve) return new Response(null, { status: 404, statusText: 'Not Found' });

    const path = decodeURIComponent(url.slice(`${LOCAL_ORIGIN}/${token}/`.length).split(/[?#]/)[0]);

    let bytes: ArrayBuffer | null;
    try {
      bytes = await resolve(path);
    } catch {
      return new Response(null, { status: 500, statusText: 'Resolver Error' });
    }
    if (bytes === null) return new Response(null, { status: 404, statusText: 'Not Found' });

    const method = (init?.method ?? 'GET').toUpperCase();
    // HEAD existence checks (the asset collector) only need an ok status.
    return new Response(method === 'HEAD' ? null : bytes, {
      status: 200,
      headers: { 'Content-Type': 'application/octet-stream' },
    });
  };
}

function uninstallInterceptor(): void {
  if (originalFetch && resolvers.size === 0) {
    window.fetch = originalFetch;
    originalFetch = null;
  }
}

/** Register a local file resolver. Returns the synthetic baseUrl to hand to `mount()`. */
export function registerLocalAssets(resolve: MjswanFileResolver): string {
  const token = makeToken();
  resolvers.set(token, resolve);
  installInterceptor();
  return `${LOCAL_ORIGIN}/${token}/`;
}

/** Unregister a base returned by `registerLocalAssets`; restores `window.fetch` when none remain. */
export function unregisterLocalAssets(baseUrl: string): void {
  const token = tokenOf(baseUrl);
  if (token !== null) resolvers.delete(token);
  uninstallInterceptor();
}
