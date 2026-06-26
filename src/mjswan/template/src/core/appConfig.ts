/**
 * Shared `config.json` shape and selection helpers.
 *
 * Used by both entry points: the standalone SPA (`App.tsx`, which discovers the
 * config from the page URL) and the embeddable library build (`MountApp.tsx`,
 * which is handed a single `config` + cross-origin `baseUrl` by `mount()`).
 */
import type { SplatConfig } from './scene/splat';
import type { EventConfig, TerrainData } from './event/EventBase';

export interface PolicyConfig {
  name: string;
  metadata?: Record<string, unknown>;
  config?: string;
  default?: boolean;
  motions?: Array<{
    name: string;
    default?: boolean;
  }>;
}

export interface CameraConfig {
  lookat?: [number, number, number];
  distance?: number;
  fovy?: number;
  elevation?: number;
  azimuth?: number;
  originType?: 'AUTO' | 'WORLD' | 'ASSET_ROOT' | 'ASSET_BODY';
  entityName?: string;
  bodyName?: string;
  enableReflections?: boolean;
  enableShadows?: boolean;
  height?: number;
  width?: number;
}

export interface SceneConfig {
  name: string;
  metadata?: Record<string, unknown>;
  policies: PolicyConfig[];
  path?: string;
  splats?: SplatConfig[];
  splatSection?: boolean;
  camera?: CameraConfig;
  events?: EventConfig[];
  terrainData?: TerrainData;
}

export interface ProjectConfig {
  name: string;
  id: string | null;
  metadata?: Record<string, unknown>;
  scenes: SceneConfig[];
}

export interface AppConfig {
  version: string;
  /** True when the build embeds author-supplied TypeScript MDP terms. */
  uses_custom_js?: boolean;
  projects: ProjectConfig[];
}

/** Lowercase-underscore slug, mirroring the Python `name2id` helper. */
export function sanitizeName(name: string): string {
  return name.toLowerCase().replace(/ /g, '_').replace(/-/g, '_');
}

/** Directory name a project's assets live under in the build output. */
export function projectDirName(project: ProjectConfig): string {
  return project.id ? project.id : 'main';
}

export function pickScene(project: ProjectConfig, sceneQuery: string | null): SceneConfig | null {
  if (!project.scenes.length) {
    return null;
  }
  if (!sceneQuery) {
    return project.scenes[0];
  }
  const normalized = sceneQuery.trim().toLowerCase();
  return (
    project.scenes.find((scene) => scene.name.toLowerCase() === normalized) ||
    project.scenes.find((scene) => sanitizeName(scene.name) === normalized) ||
    project.scenes[0]
  );
}

export function pickPolicy(scene: SceneConfig, policyQuery: string | null): string | null {
  if (!scene.policies.length) {
    return null;
  }
  const fallback = scene.policies.find((policy) => policy.default) ?? scene.policies[0];
  if (!policyQuery) {
    return fallback.name;
  }
  const normalized = policyQuery.trim().toLowerCase();
  const found =
    scene.policies.find((policy) => policy.name.toLowerCase() === normalized) ||
    scene.policies.find((policy) => sanitizeName(policy.name) === normalized);
  return found?.name ?? fallback.name;
}

export function pickMotion(policy: PolicyConfig | null, motionQuery: string | null): string | null {
  if (!policy?.motions?.length) {
    return null;
  }
  const fallback = policy.motions.find((motion) => motion.default) ?? policy.motions[0];
  if (!motionQuery) {
    return fallback.name;
  }
  const normalized = motionQuery.trim().toLowerCase();
  const found =
    policy.motions.find((motion) => motion.name.toLowerCase() === normalized) ||
    policy.motions.find((motion) => sanitizeName(motion.name) === normalized);
  return found?.name ?? fallback.name;
}

/**
 * Resolve a scene's model path relative to the asset base.
 * `humanoid/scene.mjz` (project-relative) → `main/assets/humanoid/scene.mjz`.
 */
export function resolveScenePath(project: ProjectConfig, scene: SceneConfig): string {
  const dir = projectDirName(project);
  const rel = scene.path ? scene.path : `scene/${sanitizeName(scene.name)}/scene.xml`;
  return `${dir}/assets/${rel}`.replace(/\/+/g, '/');
}

/** Resolve a project-relative asset (policy config, splat) to a base-relative path. */
export function resolveProjectAsset(project: ProjectConfig, rel: string): string {
  const dir = projectDirName(project);
  return `${dir}/assets/${rel}`.replace(/\/+/g, '/');
}
