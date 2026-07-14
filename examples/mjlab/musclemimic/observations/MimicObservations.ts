import { ObservationBase } from 'mjswan/observation';
import type { ObservationConfig } from 'mjswan/observation';
import type { PolicyRunner } from 'mjswan/types';
import { loadNpz } from 'mjswan/npz';

// ---------------------------------------------------------------------------
// Shared clip cache — keyed by URL; loaded once, shared across all observers
// ---------------------------------------------------------------------------

type MimicClipRaw = {
  qpos: Float32Array;
  qvel: Float32Array;
  siteXpos: Float32Array;
  clipSiteNames: string[];
  nFrames: number;
  nq: number;
  nv: number;
  nClipSites: number;
};

// The app feeds the clip bytes as the 'mimic_clip' motion; read that slot via
// the runner (no fetch — the engine no longer serves paths). ADR 0004 §4/§10.
// Cache the parsed clip per runner so all mimic observers share one load.
const _mimicClipCache = new WeakMap<PolicyRunner, Promise<MimicClipRaw | null>>();

function _loadMimicClipRaw(runner: PolicyRunner): Promise<MimicClipRaw | null> {
  let cached = _mimicClipCache.get(runner);
  if (!cached) {
    cached = _parseMimicClip(runner);
    _mimicClipCache.set(runner, cached);
  }
  return cached;
}

async function _parseMimicClip(runner: PolicyRunner): Promise<MimicClipRaw | null> {
  try {
    const bytes = await runner.getMotionData('mimic_clip');
    if (!bytes) {
      console.warn('[MimicObservations] mimic_clip motion bytes not supplied');
      return null;
    }
    const npz = await loadNpz(bytes);
    const qposEntry = npz['qpos'];
    const qvelEntry = npz['qvel'];
    const siteEntry = npz['site_xpos'];
    const siteNamesEntry = npz['site_names'];
    if (!qposEntry || !qvelEntry || !siteEntry) {
      console.warn('[MimicObservations] NPZ missing qpos/qvel/site_xpos');
      return null;
    }
    const nFrames = qposEntry.shape[0] ?? 0;
    const nq = qposEntry.shape[1] ?? 0;
    const nv = qvelEntry.shape[1] ?? 0;
    const nClipSites = siteEntry.shape[1] ?? 0;
    return {
      qpos: qposEntry.data,
      qvel: qvelEntry.data,
      siteXpos: siteEntry.data,
      clipSiteNames: siteNamesEntry?.strings ?? [],
      nFrames,
      nq,
      nv,
      nClipSites,
    };
  } catch (e) {
    console.warn('[MimicObservations] Failed to load mimic clip:', e);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _resolveBodyIds(runner: PolicyRunner, bodyNames: string[]): number[] {
  if (bodyNames.length === 0) return [];
  const ctx = runner.getContext();
  if (!ctx?.mjModel) return bodyNames.map(() => -1);
  const bodyObjType = (ctx.mujoco.mjtObj?.mjOBJ_BODY?.value) ?? 1;
  const ids = bodyNames.map((n) => ctx.mujoco.mj_name2id(ctx.mjModel!, bodyObjType, n));
  const missing = bodyNames.filter((_, i) => ids[i] < 0);
  if (missing.length > 0) {
    console.warn('[MimicObservations] _resolveBodyIds: body not found:', missing);
  }
  return ids;
}

function _frameIndex(simTime: number, fps: number, nFrames: number): number {
  if (nFrames <= 0 || fps <= 0) return 0;
  return Math.floor(simTime * fps) % nFrames;
}

// ---------------------------------------------------------------------------
// MimicQpos — full qpos vector (nq = 89)
// ---------------------------------------------------------------------------

export class MimicQpos extends ObservationBase {
  private nq: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.nq = runner.getContext()?.mjModel?.nq ?? 89;
  }

  get size(): number {
    return this.nq;
  }

  compute(): Float32Array {
    const qpos = this.runner.getContext()?.mjData?.qpos;
    if (!qpos) return new Float32Array(this.nq);
    const out = new Float32Array(this.nq);
    for (let i = 0; i < this.nq; i++) out[i] = qpos[i] ?? 0;
    return out;
  }
}

// ---------------------------------------------------------------------------
// MimicQvel — qvel scaled by 1/fps (nv = 88)
// ---------------------------------------------------------------------------

export class MimicQvel extends ObservationBase {
  private nv: number;
  private fps: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.nv = runner.getContext()?.mjModel?.nv ?? 88;
    this.fps = (config.fps as number | undefined) ?? 100;
  }

  get size(): number {
    return this.nv;
  }

  compute(): Float32Array {
    const qvel = this.runner.getContext()?.mjData?.qvel;
    if (!qvel) return new Float32Array(this.nv);
    const out = new Float32Array(this.nv);
    const ctrlDt = 1.0 / this.fps;
    for (let i = 0; i < this.nv; i++) out[i] = (qvel[i] ?? 0) * ctrlDt;
    return out;
  }
}

// ---------------------------------------------------------------------------
// MimicAct — muscle activation state (na actuators)
// ---------------------------------------------------------------------------

export class MimicAct extends ObservationBase {
  private na: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.na = runner.getContext()?.mjModel?.na ?? 0;
  }

  get size(): number {
    return this.na;
  }

  compute(): Float32Array {
    const act = this.runner.getContext()?.mjData?.act;
    if (!act || this.na === 0) return new Float32Array(this.na);
    const out = new Float32Array(this.na);
    for (let i = 0; i < this.na; i++) out[i] = act[i] ?? 0;
    return out;
  }
}

// ---------------------------------------------------------------------------
// MimicSitePos — current tracked body positions (n_sites * 3)
// Uses body xpos as proxy for mimic tracking sites (sites are absent in the
// compiled browser model; they were defined at the body origin in training).
// ---------------------------------------------------------------------------

export class MimicSitePos extends ObservationBase {
  private bodyIds: number[];

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const bodyNames = (config.body_names as string[] | undefined) ?? [];
    this.bodyIds = _resolveBodyIds(runner, bodyNames);
  }

  get size(): number {
    return this.bodyIds.length * 3;
  }

  compute(): Float32Array {
    const out = new Float32Array(this.size);
    if (this.bodyIds.length === 0) return out;
    const xpos = this.runner.getContext()?.mjData?.xpos;
    if (!xpos) return out;
    for (let i = 0; i < this.bodyIds.length; i++) {
      const id = this.bodyIds[i];
      if (id < 0) continue;
      out[i * 3] = xpos[id * 3] ?? 0;
      out[i * 3 + 1] = xpos[id * 3 + 1] ?? 0;
      out[i * 3 + 2] = xpos[id * 3 + 2] ?? 0;
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Base class for clip-based observations (lazy async load)
// ---------------------------------------------------------------------------

abstract class MimicClipObsBase extends ObservationBase {
  protected clip: MimicClipRaw | null = null;
  protected fps: number;
  private _loadPromise: Promise<void> | null = null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.fps = (config.fps as number | undefined) ?? 100;
    this._loadPromise = _loadMimicClipRaw(runner).then((data) => {
      this.clip = data;
    }).catch(() => {});
  }

  preload(): Promise<void> {
    return this._loadPromise ?? Promise.resolve();
  }

  protected get simTime(): number {
    return this.runner.getContext()?.mjData?.time ?? 0;
  }

  protected get rootPos(): [number, number, number] {
    const qpos = this.runner.getContext()?.mjData?.qpos;
    return [qpos?.[0] ?? 0, qpos?.[1] ?? 0, qpos?.[2] ?? 0];
  }

  protected frameIdx(nFrames: number): number {
    return _frameIndex(this.simTime, this.fps, nFrames);
  }
}

// ---------------------------------------------------------------------------
// MimicSiteTarget — clip site_xpos at current frame, ordered by model sites
// ---------------------------------------------------------------------------

export class MimicSiteTarget extends MimicClipObsBase {
  private siteNames: string[];
  private clipSiteIds: number[] | null = null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.siteNames = (config.site_names as string[] | undefined) ?? [];
  }

  get size(): number {
    return this.siteNames.length * 3;
  }

  private getClipSiteIds(): number[] {
    if (this.clipSiteIds !== null) return this.clipSiteIds;
    if (!this.clip) return this.siteNames.map(() => -1);
    this.clipSiteIds = this.siteNames.map((n) => this.clip!.clipSiteNames.indexOf(n));
    return this.clipSiteIds;
  }

  compute(): Float32Array {
    const out = new Float32Array(this.size);
    if (!this.clip) return out;
    const idx = this.frameIdx(this.clip.nFrames);
    const clipIds = this.getClipSiteIds();
    const stride = this.clip.nClipSites * 3;
    for (let i = 0; i < clipIds.length; i++) {
      const ci = clipIds[i];
      if (ci < 0) continue;
      const base = idx * stride + ci * 3;
      out[i * 3] = this.clip.siteXpos[base] ?? 0;
      out[i * 3 + 1] = this.clip.siteXpos[base + 1] ?? 0;
      out[i * 3 + 2] = this.clip.siteXpos[base + 2] ?? 0;
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// MimicSiteErr — clip target minus current site positions
// ---------------------------------------------------------------------------

export class MimicSiteErr extends MimicClipObsBase {
  private siteNames: string[];
  private bodyIds: number[];
  private clipSiteIds: number[] | null = null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.siteNames = (config.site_names as string[] | undefined) ?? [];
    const bodyNames = (config.body_names as string[] | undefined) ?? [];
    this.bodyIds = _resolveBodyIds(runner, bodyNames);
  }

  get size(): number {
    return this.siteNames.length * 3;
  }

  private getClipSiteIds(): number[] {
    if (this.clipSiteIds !== null) return this.clipSiteIds;
    if (!this.clip) return this.siteNames.map(() => -1);
    this.clipSiteIds = this.siteNames.map((n) => this.clip!.clipSiteNames.indexOf(n));
    return this.clipSiteIds;
  }

  compute(): Float32Array {
    const out = new Float32Array(this.size);
    const xpos = this.runner.getContext()?.mjData?.xpos;
    if (!this.clip || !xpos) return out;
    const idx = this.frameIdx(this.clip.nFrames);
    const clipIds = this.getClipSiteIds();
    const stride = this.clip.nClipSites * 3;
    for (let i = 0; i < clipIds.length; i++) {
      const ci = clipIds[i];
      const bi = this.bodyIds[i];
      if (ci < 0 || bi < 0) continue;
      const base = idx * stride + ci * 3;
      out[i * 3] = (this.clip.siteXpos[base] ?? 0) - (xpos[bi * 3] ?? 0);
      out[i * 3 + 1] = (this.clip.siteXpos[base + 1] ?? 0) - (xpos[bi * 3 + 1] ?? 0);
      out[i * 3 + 2] = (this.clip.siteXpos[base + 2] ?? 0) - (xpos[bi * 3 + 2] ?? 0);
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// MimicClipRefQpos — reference qpos at current clip frame
// ---------------------------------------------------------------------------

export class MimicClipRefQpos extends MimicClipObsBase {
  private nq: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.nq = runner.getContext()?.mjModel?.nq ?? 89;
  }

  get size(): number {
    return this.nq;
  }

  compute(): Float32Array {
    const out = new Float32Array(this.nq);
    if (!this.clip) return out;
    const idx = this.frameIdx(this.clip.nFrames);
    const base = idx * this.clip.nq;
    for (let i = 0; i < this.nq; i++) out[i] = this.clip.qpos[base + i] ?? 0;
    return out;
  }
}

// ---------------------------------------------------------------------------
// MimicClipRefQvel — reference qvel at current clip frame (no scaling)
// ---------------------------------------------------------------------------

export class MimicClipRefQvel extends MimicClipObsBase {
  private nv: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.nv = runner.getContext()?.mjModel?.nv ?? 88;
  }

  get size(): number {
    return this.nv;
  }

  compute(): Float32Array {
    const out = new Float32Array(this.nv);
    if (!this.clip) return out;
    const idx = this.frameIdx(this.clip.nFrames);
    const base = idx * this.clip.nv;
    for (let i = 0; i < this.nv; i++) out[i] = this.clip.qvel[base + i] ?? 0;
    return out;
  }
}

// ---------------------------------------------------------------------------
// MimicClipPhase — normalised position in [0, 1] along the clip
// ---------------------------------------------------------------------------

export class MimicClipPhase extends MimicClipObsBase {
  get size(): number {
    return 1;
  }

  compute(): Float32Array {
    if (!this.clip || this.clip.nFrames <= 0) return new Float32Array(1);
    const idx = this.frameIdx(this.clip.nFrames);
    return new Float32Array([idx / this.clip.nFrames]);
  }
}

// ---------------------------------------------------------------------------
// MimicLookahead — k-step lookahead over clip site positions + root kinematics
//
// Per step (i = 1..k, stride frames apart):
//   [0..nSites*3-1]        relative site positions (clip_site_xpos - cur_root_pos)
//   [nSites*3..nSites*3+2] delta root pos (clip.qpos[future,:3] - cur_root_pos)
//   [nSites*3+3..+5]       future root vel (clip.qvel[future,:3])
//   [nSites*3+6]           phase (future_frame / max(nFrames-1, 1))
// Total per step: nSites*3 + 7; default nSites=17 -> 58 per step, k=5 -> 290
// ---------------------------------------------------------------------------

export class MimicLookahead extends MimicClipObsBase {
  private k: number;
  private stride: number;
  private nClipSitesHint: number;
  private siteNames: string[];
  private clipSiteIds: number[] | null = null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);  // sets this.fps
    this.k = (config.k as number | undefined) ?? 5;
    this.stride = (config.stride as number | undefined) ?? 20;
    this.nClipSitesHint = (config.n_clip_sites as number | undefined) ?? 17;
    this.siteNames = (config.site_names as string[] | undefined) ?? [];
  }

  private getClipSiteIds(): number[] {
    if (this.clipSiteIds !== null) return this.clipSiteIds;
    if (!this.clip) return this.siteNames.map(() => -1);
    this.clipSiteIds = this.siteNames.map((n) => this.clip!.clipSiteNames.indexOf(n));
    return this.clipSiteIds;
  }

  get size(): number {
    const nSites = this.siteNames.length > 0
      ? this.siteNames.length
      : (this.clip?.nClipSites ?? this.nClipSitesHint);
    return this.k * (nSites * 3 + 7);
  }

  compute(): Float32Array {
    const out = new Float32Array(this.size);
    if (!this.clip) return out;
    const { nFrames, nClipSites, siteXpos, qpos: clipQpos, qvel: clipQvel, nq, nv } = this.clip;
    if (nFrames <= 0) return out;

    const [curRootX, curRootY, curRootZ] = this.rootPos;
    const curIdx = this.frameIdx(nFrames);

    const clipIds = this.getClipSiteIds();
    const nSites = clipIds.length;
    const perStep = nSites * 3 + 7;

    for (let si = 0; si < this.k; si++) {
      const futureIdx = (curIdx + (si + 1) * this.stride) % nFrames;
      let offset = si * perStep;

      const siteBase = futureIdx * nClipSites * 3;
      for (let j = 0; j < nSites; j++) {
        const ci = clipIds[j];
        if (ci < 0) continue;
        out[offset + j * 3] = (siteXpos[siteBase + ci * 3] ?? 0) - curRootX;
        out[offset + j * 3 + 1] = (siteXpos[siteBase + ci * 3 + 1] ?? 0) - curRootY;
        out[offset + j * 3 + 2] = (siteXpos[siteBase + ci * 3 + 2] ?? 0) - curRootZ;
      }
      offset += nSites * 3;

      const qposBase = futureIdx * nq;
      out[offset] = (clipQpos[qposBase] ?? 0) - curRootX;
      out[offset + 1] = (clipQpos[qposBase + 1] ?? 0) - curRootY;
      out[offset + 2] = (clipQpos[qposBase + 2] ?? 0) - curRootZ;
      offset += 3;

      const qvelBase = futureIdx * nv;
      out[offset] = clipQvel[qvelBase] ?? 0;
      out[offset + 1] = clipQvel[qvelBase + 1] ?? 0;
      out[offset + 2] = clipQvel[qvelBase + 2] ?? 0;
      offset += 3;

      out[offset] = futureIdx / Math.max(nFrames - 1, 1);
    }

    return out;
  }
}
