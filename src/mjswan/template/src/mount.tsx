/**
 * Library entry point for embedding a mjswan simulation in a host page.
 *
 * Built by Vite library mode (`vite.lib.config.ts`) into a single self-contained
 * ESM (`dist/mjswan.js`) with every dependency bundled and the MuJoCo / ONNX
 * WASM co-located in `dist/`. mjswan Cloud loads it from a pinned CDN URL:
 *
 *   const { mount } = await import(
 *     'https://cdn.jsdelivr.net/npm/mjswan@<version>/dist/mjswan.js'
 *   )
 *   const sim = await mount(container, 'https://cdn.mjswan.com/.../scenes/<id>/config.json')
 *
 * `mount` accepts either:
 *  - a **config.json URL** (production watch/embed): every asset (scene.mjz,
 *    policy.onnx, policy.json, motion.npz, splats) resolves relative to its
 *    directory, so it works cross-origin (data on `cdn.mjswan.com`, page on
 *    `mjswan.com`); or
 *  - a **file resolver** `{ resolve(path) }` (mjswan Cloud's upload preview):
 *    the engine asks for each scene-relative path and the host returns the
 *    locally-selected bytes — no upload/CDN round-trip. See ADR 0005.
 *
 * It returns an instance exposing `captureThumbnail()` (used by the upload
 * preview's "Scan thumbnail") and `dispose()`. It runs single-threaded by
 * default and is independent of COOP/COEP. See mjswan-cloud ADR 0001 / 0005.
 */
import { createRoot, type Root } from 'react-dom/client';
import { MountApp } from './MountApp';
import type { AppConfig } from './core/appConfig';
import type { mjswanRuntime } from './core/engine/runtime';
import {
  registerLocalAssets,
  unregisterLocalAssets,
  type MjswanFileResolver,
} from './core/utils/localAssets';
import '@mantine/core/styles.css';
import './index.css';

const roots = new WeakMap<HTMLElement, Root>();

/** Where `mount` reads scene data from: a config.json URL, or an in-memory resolver. */
export type MjswanSource = string | { resolve: MjswanFileResolver };

/** A mounted simulation instance. */
export interface MjswanInstance {
  /** Capture the current frame as a JPEG Blob (renders + reads back synchronously). */
  captureThumbnail: (options?: { maxDim?: number; quality?: number }) => Promise<Blob>;
  /** Tear down the simulation, free resources, and detach any local file source. */
  dispose: () => void;
}

async function fetchConfig(configUrl: string): Promise<AppConfig> {
  let response: Response;
  try {
    response = await fetch(configUrl, { cache: 'no-store' });
  } catch (error) {
    throw new Error(
      `mjswan.mount: failed to fetch config.json from ${configUrl}: ` +
        (error instanceof Error ? error.message : String(error))
    );
  }
  if (!response.ok) {
    throw new Error(`mjswan.mount: failed to fetch ${configUrl}: HTTP ${response.status}`);
  }
  let config: AppConfig;
  try {
    config = (await response.json()) as AppConfig;
  } catch (error) {
    throw new Error(
      `mjswan.mount: invalid config.json at ${configUrl}: ` +
        (error instanceof Error ? error.message : String(error))
    );
  }
  if (config.uses_custom_js === true) {
    // The shared engine interprets declarative data only; a custom-JS build's
    // terms have no implementation here. Fail clearly instead of rendering a
    // broken scene. See mjswan ADR 0003 / mjswan-cloud ADR 0001 §3.3.
    throw new Error(
      'mjswan.mount: this build uses custom-JS MDP terms (uses_custom_js: true) ' +
        'and cannot be rendered by the shared engine.'
    );
  }
  if (!Array.isArray(config.projects) || config.projects.length === 0) {
    throw new Error(`mjswan.mount: config.json at ${configUrl} has no projects.`);
  }
  return config;
}

/**
 * Render a published mjswan simulation into `element`.
 *
 * @param element  Host element to render the viewer (and controls) into.
 * @param source  A config.json URL, or a `{ resolve(path) }` file resolver.
 * @returns Resolves, once the first scene is running, to a {@link MjswanInstance};
 *   rejects on load failure.
 */
export async function mount(element: HTMLElement, source: MjswanSource): Promise<MjswanInstance> {
  if (!(element instanceof HTMLElement)) {
    throw new Error('mjswan.mount: first argument must be an HTMLElement.');
  }

  let configUrl: string;
  let cleanupSource: (() => void) | null = null;

  if (typeof source === 'string') {
    if (!source) {
      throw new Error('mjswan.mount: second argument must be a config.json URL or a { resolve } file resolver.');
    }
    configUrl = new URL(source, window.location.href).toString();
  } else if (source && typeof source.resolve === 'function') {
    const base = registerLocalAssets(source.resolve);
    configUrl = `${base}config.json`;
    cleanupSource = () => unregisterLocalAssets(base);
  } else {
    throw new Error('mjswan.mount: second argument must be a config.json URL or a { resolve } file resolver.');
  }

  // Asset base = the directory containing config.json.
  const baseUrl = new URL('.', configUrl).toString();

  let config: AppConfig;
  try {
    config = await fetchConfig(configUrl);
  } catch (error) {
    cleanupSource?.();
    throw error;
  }

  let root = roots.get(element);
  if (!root) {
    root = createRoot(element);
    roots.set(element, root);
  }

  let runtime: mjswanRuntime | null = null;

  try {
    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const onReady = () => {
        if (!settled) {
          settled = true;
          resolve();
        }
      };
      const onError = (error: Error) => {
        if (!settled) {
          settled = true;
          reject(error);
        }
      };
      root!.render(
        <MountApp
          config={config}
          baseUrl={baseUrl}
          onReady={onReady}
          onError={onError}
          onRuntimeReady={(rt) => {
            runtime = rt;
          }}
        />
      );
    });
  } catch (error) {
    cleanupSource?.();
    throw error;
  }

  return {
    captureThumbnail: (options) => {
      if (!runtime) {
        return Promise.reject(new Error('mjswan: simulation not ready; cannot capture a thumbnail.'));
      }
      return runtime.captureThumbnail(options);
    },
    dispose: () => {
      unmount(element);
      cleanupSource?.();
    },
  };
}

/** Tear down a previously mounted simulation and free its resources. */
export function unmount(element: HTMLElement): void {
  const root = roots.get(element);
  if (root) {
    root.unmount();
    roots.delete(element);
  }
}

export default mount;
