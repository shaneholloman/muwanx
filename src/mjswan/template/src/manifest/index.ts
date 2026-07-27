/**
 * `mjswan/manifest` — parse a Builder `config.json` + a byte source into a typed
 * catalog of loadable scenes/policies/splats (ADR 0004 §1, §9). Framework-free;
 * shared by the in-repo React app and mjswan Cloud. The engine itself knows
 * nothing about config.json — the app owns the catalog and calls engine verbs.
 *
 * A build may hold multiple projects; the catalog exposes all of them and the
 * app picks which one is active (the `id: null` "main" project is the default).
 */
import type { Bytes } from '../core/utils/bytes';
import type { ViewerConfig } from '../core/engine/viewer_config';
import type { EventConfig, TerrainData } from '../core/event/EventBase';
import type { SplatConfig, SplatTransform } from '../core/scene/splat';
import { eventGraphRefs, policyGraphRefs } from '../core/onnx/graphRefs';
import type { PolicyInput, SceneInput, SplatInput } from '../engine/types';

/** Maps a build-relative asset path (e.g. `main/assets/humanoid/scene.mjz`) to bytes. */
export type ByteSource = (relPath: string) => Bytes;

// ── config.json shape (Python-Builder ↔ consumer contract) ──────────────────
interface ConfigPolicyRef {
  name: string;
  config?: string; // relative path to policy.json
  default?: boolean;
  motions?: Array<{ name: string; default?: boolean }>;
}
interface ConfigScene {
  name: string;
  path?: string; // relative model path (.mjz)
  policies: ConfigPolicyRef[];
  splats?: SplatConfig[];
  splatSection?: boolean;
  camera?: ViewerConfig;
  events?: EventConfig[];
  terrainData?: TerrainData;
}
interface ConfigProject {
  name: string;
  id: string | null;
  scenes: ConfigScene[];
}
export interface AppConfig {
  version: string;
  uses_custom_js?: boolean;
  /** Build-relative path to the runtime custom-MDP plugin ESM (custom-JS builds). */
  plugins?: string;
  projects: ConfigProject[];
}

// ── catalog (what the app holds) ─────────────────────────────────────────────
export interface PolicyEntry {
  name: string;
  default: boolean;
  motions: Array<{ name: string; default: boolean }>;
  /** Fetch + assemble the engine PolicyInput (policy.json, onnx, motion bytes). */
  build(): Promise<PolicyInput>;
}
export interface SplatEntry {
  name: string;
  /** Whether the UI should show calibration controls. */
  control: boolean;
  /** Initial placement, for seeding a calibration UI (engine.calibrateSplat). */
  transform: SplatTransform;
  build(): Promise<SplatInput>;
}
export interface SceneEntry {
  name: string;
  camera?: ViewerConfig;
  splatSection: boolean;
  policies: PolicyEntry[];
  splats: SplatEntry[];
  /** Assemble a full SceneInput for a chosen policy/splat (names; defaults if omitted). */
  buildScene(opts?: { policy?: string | null; splat?: string | null }): Promise<SceneInput>;
}
export interface ProjectCatalog {
  name: string;
  id: string | null;
  scenes: SceneEntry[];
}
export interface Catalog {
  /** All projects in the build; the app chooses the active one. First is default. */
  projects: ProjectCatalog[];
  /**
   * Build-relative path to the author custom-MDP plugin ESM, when present. The
   * app (trusted contexts only) dynamically imports it and passes the exports
   * as {@link EnginePlugins}; mjswan Cloud ignores it (ADR 0004 §10).
   */
  pluginsPath?: string;
}

/** Lowercase-underscore slug, mirroring the Python `name2id` helper. */
export function sanitizeName(name: string): string {
  return name.toLowerCase().replace(/ /g, '_').replace(/-/g, '_');
}

function dirName(project: ConfigProject): string {
  return project.id ? project.id : 'main';
}

/** `humanoid/scene.mjz` (project-relative) → `main/assets/humanoid/scene.mjz`. */
function scenePath(project: ConfigProject, scene: ConfigScene): string {
  const rel = scene.path ? scene.path : `scene/${sanitizeName(scene.name)}/scene.mjz`;
  return `${dirName(project)}/assets/${rel}`.replace(/\/+/g, '/');
}

/** Resolve a project-relative asset (policy.json, splat) to a build-relative path. */
function projectAsset(project: ConfigProject, rel: string): string {
  return `${dirName(project)}/assets/${rel}`.replace(/\/+/g, '/');
}

/**
 * Join an asset path referenced *inside* another file, relative to that file's
 * directory: policy.json's onnx/motion/graph paths, and a scene's event graphs
 * (which sit beside the model).
 */
function siblingOf(filePath: string, assetPath: string): string {
  const lastSlash = filePath.lastIndexOf('/');
  const dir = lastSlash >= 0 ? filePath.slice(0, lastSlash + 1) : '';
  return `${dir}${assetPath}`.replace(/\/+/g, '/');
}

/** Lazy-fetch an absolute/external URL as bytes (for splats hosted off-build). */
function urlBytes(url: string): Bytes {
  return async () => {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`mjswan/manifest: failed to fetch ${url}: ${res.status}`);
    }
    return res.arrayBuffer();
  };
}

async function fetchJson(bytes: Bytes): Promise<Record<string, unknown>> {
  const buffer = typeof bytes === 'function' ? await bytes() : bytes;
  return JSON.parse(new TextDecoder().decode(buffer)) as Record<string, unknown>;
}

function splatTransform(splat: SplatConfig): SplatTransform {
  return {
    scale: splat.scale,
    xOffset: splat.xOffset,
    yOffset: splat.yOffset,
    zOffset: splat.zOffset,
    roll: splat.roll,
    pitch: splat.pitch,
    yaw: splat.yaw,
  };
}

function buildSplat(project: ConfigProject, splat: SplatConfig, source: ByteSource): SplatInput {
  const data = splat.path ? source(projectAsset(project, splat.path)) : urlBytes(splat.url!);
  const collider = splat.colliderUrl
    ? (/^[a-z]+:\/\//i.test(splat.colliderUrl)
        ? urlBytes(splat.colliderUrl)
        : source(projectAsset(project, splat.colliderUrl)))
    : undefined;
  return { data, collider, transform: splatTransform(splat) };
}

async function buildPolicy(
  project: ConfigProject,
  policy: ConfigPolicyRef,
  source: ByteSource,
): Promise<PolicyInput> {
  if (!policy.config) {
    throw new Error(`mjswan/manifest: policy "${policy.name}" has no config path.`);
  }
  const configPath = projectAsset(project, policy.config);
  const config = await fetchJson(source(configPath));

  const onnxRel = (config.onnx as { path?: string } | undefined)?.path;
  if (!onnxRel) {
    throw new Error(`mjswan/manifest: policy "${policy.name}" config missing onnx.path.`);
  }
  const onnx = source(siblingOf(configPath, onnxRel));

  const motions = Array.isArray(config.motions)
    ? (config.motions as Array<{ name: string; path: string; default?: boolean }>).map((m) => ({
        name: m.name,
        data: source(siblingOf(configPath, m.path)),
        default: m.default,
      }))
    : [];

  return { config, onnx, graphs: graphBytes(policyGraphRefs(config), configPath, source), motions };
}

/**
 * Byte sources for a config's traced term graphs, keyed by the path the config
 * refers to them by (ADR 0005 §4).
 *
 * The keys stay config-relative — that is what the runtime looks a session up by
 * — while the source is asked for the build-relative path, same as any other
 * policy-adjacent asset.
 */
function graphBytes(
  refs: string[],
  configPath: string,
  source: ByteSource,
): Record<string, Bytes> {
  const graphs: Record<string, Bytes> = {};
  for (const ref of refs) graphs[ref] = source(siblingOf(configPath, ref));
  return graphs;
}

function toSceneEntry(project: ConfigProject, scene: ConfigScene, source: ByteSource): SceneEntry {
  const policies: PolicyEntry[] = scene.policies.map((p) => ({
    name: p.name,
    default: p.default ?? false,
    motions: (p.motions ?? []).map((m) => ({ name: m.name, default: m.default ?? false })),
    build: () => buildPolicy(project, p, source),
  }));

  const splats: SplatEntry[] = (scene.splats ?? []).map((s) => ({
    name: s.name,
    control: s.control ?? false,
    transform: splatTransform(s),
    build: async () => buildSplat(project, s, source),
  }));

  return {
    name: scene.name,
    camera: scene.camera,
    splatSection: scene.splatSection ?? false,
    policies,
    splats,
    buildScene: async (opts) => {
      const policyName = opts?.policy;
      const policy =
        policyName == null
          ? scene.policies.find((p) => p.default) ?? scene.policies[0]
          : scene.policies.find((p) => p.name === policyName);
      const splatName = opts?.splat;
      const splat =
        splatName == null ? undefined : scene.splats?.find((s) => s.name === splatName);
      // Event graphs are written beside the model, so they resolve relative to it
      // rather than to a policy.json.
      const modelPath = scenePath(project, scene);
      return {
        model: source(modelPath),
        policy: policy ? await buildPolicy(project, policy, source) : null,
        splat: splat ? buildSplat(project, splat, source) : null,
        viewer: scene.camera,
        events: scene.events,
        terrainData: scene.terrainData,
        graphs: graphBytes(eventGraphRefs(scene.events), modelPath, source),
      };
    },
  };
}

/** Parse a Builder `config.json` (object or JSON string) into a {@link Catalog}. */
export function parseManifest(config: AppConfig | string, source: ByteSource): Catalog {
  const parsed: AppConfig = typeof config === 'string' ? JSON.parse(config) : config;
  if (!parsed.projects?.length) {
    throw new Error('mjswan/manifest: config.json has no projects.');
  }
  // "main" (id: null) first so it stays the default active project.
  const ordered = [...parsed.projects.filter((p) => p.id === null), ...parsed.projects.filter((p) => p.id !== null)];
  return {
    projects: ordered.map((project) => ({
      name: project.name,
      id: project.id,
      scenes: project.scenes.map((scene) => toSceneEntry(project, scene, source)),
    })),
    pluginsPath: parsed.plugins,
  };
}
