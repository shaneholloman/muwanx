/**
 * `mjswan/manifest` — parse a Builder `config.json` + a byte source into a typed
 * catalog of loadable scenes/policies/splats (ADR 0004 §1, §9). Framework-free;
 * shared by the in-repo React app and mjswan Cloud. The engine itself knows
 * nothing about config.json — the app owns the catalog and calls engine verbs.
 *
 * A build may hold multiple projects; the catalog exposes all of them and the
 * app picks which one is active. The one marked `default` comes first, else the
 * first in document order (ADR 0006 §4).
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
  /** Seconds per control step, from the task (mjlab's `timestep * decimation`). */
  controlDt?: number;
}
interface ConfigProject {
  name: string;
  /** `name2id(name)`, unique in the document: the project's directory and `?project=` value. */
  id: string;
  /** At most one project sets it; none set means the first in document order. */
  default?: boolean;
  scenes: ConfigScene[];
}
export interface AppConfig {
  /**
   * Document format (ADR 0006 §7): the structure of the build, bumped only for a break an
   * older reader would misread. Absent on builds that predate it. Distinct from `version`,
   * which names the mjswan release that wrote the document.
   */
  format?: number;
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
  id: string;
  /** The project the app opens on. Exactly one entry in a catalog has it set. */
  default: boolean;
  scenes: SceneEntry[];
}
export interface Catalog {
  /** All projects in the build; the app chooses the active one. First is default. */
  projects: ProjectCatalog[];
  /**
   * Path to the author custom-MDP plugin ESM, if any. A trusted app imports it and passes
   * the exports as {@link EnginePlugins}; mjswan Cloud ignores it.
   */
  pluginsPath?: string;
}

/**
 * The id a name sanitizes to — the same function as Python's `name2id`, character for
 * character: lowercase, every run of anything outside `[a-z0-9]` collapsed to one `_`,
 * leading and trailing `_` stripped. The pair is pinned by `name2id_cases.json`, which
 * both test suites read, because the two used to disagree on apostrophes, parentheses
 * and accents and only a raw-name fallback hid it (ADR 0006 §4).
 */
export function sanitizeName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

/** `humanoid/scene.mjz` (project-relative) → `<project-id>/assets/humanoid/scene.mjz`. */
function scenePath(project: ConfigProject, scene: ConfigScene): string {
  const rel = scene.path ? scene.path : `scene/${sanitizeName(scene.name)}/scene.mjz`;
  return `${project.id}/assets/${rel}`.replace(/\/+/g, '/');
}

/** Resolve a project-relative asset (policy.json, splat) to a build-relative path. */
function projectAsset(project: ConfigProject, rel: string): string {
  return `${project.id}/assets/${rel}`.replace(/\/+/g, '/');
}

/**
 * Join a path referenced *inside* another file, relative to that file's directory:
 * policy.json's onnx/motion/graph paths, and a scene's event graphs.
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
      // Event graphs sit beside the model, so they resolve relative to it, not policy.json.
      const modelPath = scenePath(project, scene);
      return {
        model: source(modelPath),
        policy: policy ? await buildPolicy(project, policy, source) : null,
        splat: splat ? buildSplat(project, splat, source) : null,
        viewer: scene.camera,
        events: scene.events,
        terrainData: scene.terrainData,
        controlDt: scene.controlDt,
        graphs: graphBytes(eventGraphRefs(scene.events), modelPath, source),
      };
    },
  };
}

/** The newest document format this reader understands (ADR 0006 §7). */
export const MAX_DOCUMENT_FORMAT = 1;

/**
 * Refuse a document written in a format newer than this reader. The check reads
 * `format` only: `version` is provenance, not a gate — a host may pick an engine by it,
 * but the engine itself must not turn a version mismatch into an error.
 */
function checkFormat(parsed: AppConfig): void {
  const format = parsed.format;
  if (format === undefined) return;
  if (!Number.isInteger(format) || format < 0) {
    throw new Error(`mjswan/manifest: document format ${JSON.stringify(format)} is not a format number.`);
  }
  if (format > MAX_DOCUMENT_FORMAT) {
    throw new Error(
      `mjswan/manifest: this document is format ${format} (written by mjswan ${parsed.version}), ` +
        `but this engine reads up to format ${MAX_DOCUMENT_FORMAT}. Update the engine.`,
    );
  }
}

/** Parse a Builder `config.json` (object or JSON string) into a {@link Catalog}. */
export function parseManifest(config: AppConfig | string, source: ByteSource): Catalog {
  const parsed: AppConfig = typeof config === 'string' ? JSON.parse(config) : config;
  checkFormat(parsed);
  if (!parsed.projects?.length) {
    throw new Error('mjswan/manifest: config.json has no projects.');
  }
  // The default project first, so `projects[0]` is always the one to open on. The build
  // refuses two defaults; with none, document order already puts the right one first.
  const flagged = parsed.projects.find((p) => p.default) ?? parsed.projects[0];
  const ordered = [flagged, ...parsed.projects.filter((p) => p !== flagged)];
  return {
    projects: ordered.map((project) => ({
      name: project.name,
      id: project.id,
      default: project === flagged,
      scenes: project.scenes.map((scene) => toSceneEntry(project, scene, source)),
    })),
    pluginsPath: parsed.plugins,
  };
}
