/**
 * Library entry point for embedding a mjswan simulation in a host page.
 *
 * Built by Vite library mode (`vite.lib.config.ts`) into a single self-contained
 * ESM (`dist/mjswan.js`) with every dependency bundled and the MuJoCo / ONNX
 * WASM co-located in `dist/`. mjswan Cloud (v2) loads it from a pinned CDN URL:
 *
 *   const { mount } = await import(
 *     'https://cdn.jsdelivr.net/npm/mjswan@<version>/dist/mjswan.js'
 *   )
 *   await mount(container, 'https://cdn.mjswan.com/mjswan/scenes/<id>/config.json')
 *
 * `mount` fetches `configUrl` and resolves every asset (scene.mjz, policy.onnx,
 * policy.json, motion.npz, splats) against `configUrl`'s directory, so it works
 * cross-origin (data on `cdn.mjswan.com`, page on `mjswan.com`). It runs
 * single-threaded by default and is independent of COOP/COEP. See mjswan-cloud
 * ADR 0001.
 */
import { createRoot, type Root } from 'react-dom/client';
import { MountApp } from './MountApp';
import type { AppConfig } from './core/appConfig';
import '@mantine/core/styles.css';
import './index.css';

const roots = new WeakMap<HTMLElement, Root>();

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
 * @param configUrl  Absolute (or page-relative) URL of the simulation's
 *   `config.json`. All other assets are resolved relative to its directory.
 * @returns Resolves once the first scene is running; rejects on load failure.
 */
export async function mount(element: HTMLElement, configUrl: string): Promise<void> {
  if (!(element instanceof HTMLElement)) {
    throw new Error('mjswan.mount: first argument must be an HTMLElement.');
  }
  if (!configUrl || typeof configUrl !== 'string') {
    throw new Error('mjswan.mount: second argument must be a config.json URL.');
  }

  const absoluteConfigUrl = new URL(configUrl, window.location.href).toString();
  // Asset base = the directory containing config.json.
  const baseUrl = new URL('.', absoluteConfigUrl).toString();

  const config = await fetchConfig(absoluteConfigUrl);

  let root = roots.get(element);
  if (!root) {
    root = createRoot(element);
    roots.set(element, root);
  }

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
      <MountApp config={config} baseUrl={baseUrl} onReady={onReady} onError={onError} />
    );
  });
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
