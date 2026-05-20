import type { MjModel } from 'mujoco';
import { ObservationBase } from './ObservationBase';
import type { ObservationConfig } from './ObservationBase';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';
import { loadNpz } from '../scene/npz';

// ---------------------------------------------------------------------------
// Shared clip cache — keyed by URL; loaded once, shared across all observers
// ---------------------------------------------------------------------------

type MimicClipRaw = {
  qpos: Float32Array;       // flat (T * nq)
  qvel: Float32Array;       // flat (T * nv)
  siteXpos: Float32Array;   // flat (T * nClipSites * 3)
  clipSiteNames: string[];  // clip's own site ordering
  nFrames: number;
  nq: number;
  nv: number;
  nClipSites: number;
};

const _mimicClipCache = new Map<string, Promise<MimicClipRaw | null>>();

function _loadMimicClipRaw(url: string): Promise<MimicClipRaw | null> {
  if (!_mimicClipCache.has(url)) {
    _mimicClipCache.set(url, _fetchMimicClip(url));
  }
  return _mimicClipCache.get(url)!;
}

async function _fetchMimicClip(url: string): Promise<MimicClipRaw | null> {
  try {
    const npz = await loadNpz(url);
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

function _getModelSiteNames(mjModel: MjModel): string[] {
  const namesArray = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let i = 0; i < mjModel.nsite; i++) {
    let start = mjModel.name_siteadr[i];
    let end = start;
    while (end < namesArray.length && namesArray[end] !== 0) end++;
    names.push(decoder.decode(namesArray.subarray(start, end)));
  }
  return names;
}

function _resolveSiteIds(mjModel: MjModel | null, siteNames: string[]): number[] {
  if (!mjModel || siteNames.length === 0) return [];
  const allNames = _getModelSiteNames(mjModel);
  return siteNames.map((n) => allNames.indexOf(n));
}

function _findClipUrl(runner: PolicyRunner): string | null {
  const motions = runner.getConfig().motions ?? [];
  const clipMotion = motions.find((m) => m.name === 'mimic_clip');
  return clipMotion?.path ?? null;
}

function _frameIndex(simTime: number, ctrlDt: number, nFrames: number): number {
  if (nFrames <= 0 || ctrlDt <= 0) return 0;
  return Math.floor(simTime / ctrlDt) % nFrames;
}

// ---------------------------------------------------------------------------
// MimicQpos — full qpos vector (nq = 89)
// ---------------------------------------------------------------------------

class MimicQpos extends ObservationBase {
  private nq: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const mjModel = runner.getContext()?.mjModel ?? null;
    this.nq = mjModel?.nq ?? 89;
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
// MimicQvel — qvel scaled by ctrl_dt (nv = 88)
// ---------------------------------------------------------------------------

class MimicQvel extends ObservationBase {
  private nv: number;
  private ctrlDt: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const mjModel = runner.getContext()?.mjModel ?? null;
    this.nv = mjModel?.nv ?? 88;
    this.ctrlDt = (config.ctrl_dt as number | undefined) ?? 0.01;
  }

  get size(): number {
    return this.nv;
  }

  compute(): Float32Array {
    const qvel = this.runner.getContext()?.mjData?.qvel;
    if (!qvel) return new Float32Array(this.nv);
    const out = new Float32Array(this.nv);
    for (let i = 0; i < this.nv; i++) out[i] = (qvel[i] ?? 0) * this.ctrlDt;
    return out;
  }
}

// ---------------------------------------------------------------------------
// MimicAct — muscle activation state (na actuators)
// ---------------------------------------------------------------------------

class MimicAct extends ObservationBase {
  private na: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const mjModel = runner.getContext()?.mjModel ?? null;
    this.na = mjModel?.na ?? 0;
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
// MimicSitePos — current tracked site positions (n_sites * 3)
// ---------------------------------------------------------------------------

class MimicSitePos extends ObservationBase {
  private siteIds: number[];

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const siteNames = (config.site_names as string[] | undefined) ?? [];
    const mjModel = runner.getContext()?.mjModel ?? null;
    this.siteIds = _resolveSiteIds(mjModel, siteNames);
  }

  get size(): number {
    return this.siteIds.length * 3;
  }

  compute(): Float32Array {
    const out = new Float32Array(this.size);
    if (this.siteIds.length === 0) return out;
    const siteXpos = this.runner.getContext()?.mjData?.site_xpos;
    if (!siteXpos) return out;
    for (let i = 0; i < this.siteIds.length; i++) {
      const id = this.siteIds[i];
      out[i * 3] = siteXpos[id * 3] ?? 0;
      out[i * 3 + 1] = siteXpos[id * 3 + 1] ?? 0;
      out[i * 3 + 2] = siteXpos[id * 3 + 2] ?? 0;
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Base class for clip-based observations (lazy async load)
// ---------------------------------------------------------------------------

abstract class MimicClipObsBase extends ObservationBase {
  protected clip: MimicClipRaw | null = null;
  protected ctrlDt: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.ctrlDt = (config.ctrl_dt as number | undefined) ?? 0.01;
    const url = _findClipUrl(runner);
    if (url) {
      _loadMimicClipRaw(url).then((data) => {
        this.clip = data;
      }).catch(() => {});
    }
  }

  protected get simTime(): number {
    return this.runner.getContext()?.mjData?.time ?? 0;
  }

  protected get rootPos(): [number, number, number] {
    const qpos = this.runner.getContext()?.mjData?.qpos;
    return [qpos?.[0] ?? 0, qpos?.[1] ?? 0, qpos?.[2] ?? 0];
  }

  protected frameIdx(nFrames: number): number {
    return _frameIndex(this.simTime, this.ctrlDt, nFrames);
  }
}

// ---------------------------------------------------------------------------
// MimicSiteTarget — clip site_xpos at current frame, ordered by model sites
// ---------------------------------------------------------------------------

class MimicSiteTarget extends MimicClipObsBase {
  private siteNames: string[];
  private siteIds: number[];
  private clipSiteIds: number[] | null = null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.siteNames = (config.site_names as string[] | undefined) ?? [];
    const mjModel = runner.getContext()?.mjModel ?? null;
    this.siteIds = _resolveSiteIds(mjModel, this.siteNames);
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

class MimicSiteErr extends MimicClipObsBase {
  private siteNames: string[];
  private siteIds: number[];
  private clipSiteIds: number[] | null = null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.siteNames = (config.site_names as string[] | undefined) ?? [];
    const mjModel = runner.getContext()?.mjModel ?? null;
    this.siteIds = _resolveSiteIds(mjModel, this.siteNames);
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
    const siteXpos = this.runner.getContext()?.mjData?.site_xpos;
    if (!this.clip || !siteXpos) return out;
    const idx = this.frameIdx(this.clip.nFrames);
    const clipIds = this.getClipSiteIds();
    const stride = this.clip.nClipSites * 3;
    for (let i = 0; i < clipIds.length; i++) {
      const ci = clipIds[i];
      const si = this.siteIds[i];
      if (ci < 0 || si < 0) continue;
      const base = idx * stride + ci * 3;
      out[i * 3] = (this.clip.siteXpos[base] ?? 0) - (siteXpos[si * 3] ?? 0);
      out[i * 3 + 1] = (this.clip.siteXpos[base + 1] ?? 0) - (siteXpos[si * 3 + 1] ?? 0);
      out[i * 3 + 2] = (this.clip.siteXpos[base + 2] ?? 0) - (siteXpos[si * 3 + 2] ?? 0);
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// MimicClipRefQpos — reference qpos at current clip frame
// ---------------------------------------------------------------------------

class MimicClipRefQpos extends MimicClipObsBase {
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
// MimicClipRefQvel — reference qvel at current clip frame (no ctrl_dt scaling)
// ---------------------------------------------------------------------------

class MimicClipRefQvel extends MimicClipObsBase {
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

class MimicClipPhase extends MimicClipObsBase {
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
//   [0..nSites*3-1]       relative site positions (clip_site_xpos - cur_root_pos)
//   [nSites*3..nSites*3+2] delta root pos (clip.qpos[future,:3] - cur_root_pos)
//   [nSites*3+3..nSites*3+5] future root vel (clip.qvel[future,:3])
//   [nSites*3+6]          phase (future_frame / max(nFrames-1, 1))
// Total per step: nClipSites*3 + 7; default = 17*3+7 = 58 -> k=5 -> 290
// ---------------------------------------------------------------------------

class MimicLookahead extends MimicClipObsBase {
  private k: number;
  private stride: number;
  private nClipSitesHint: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.k = (config.k as number | undefined) ?? 5;
    this.stride = (config.stride as number | undefined) ?? 20;
    this.nClipSitesHint = (config.n_clip_sites as number | undefined) ?? 17;
  }

  get size(): number {
    const nSites = this.clip?.nClipSites ?? this.nClipSitesHint;
    return this.k * (nSites * 3 + 7);
  }

  compute(): Float32Array {
    const out = new Float32Array(this.size);
    if (!this.clip) return out;
    const { nFrames, nClipSites, siteXpos, qpos: clipQpos, qvel: clipQvel, nq, nv } = this.clip;
    if (nFrames <= 0) return out;

    const [curRootX, curRootY, curRootZ] = this.rootPos;

    const curIdx = this.frameIdx(nFrames);
    const perStep = nClipSites * 3 + 7;

    for (let si = 0; si < this.k; si++) {
      const futureIdx = (curIdx + (si + 1) * this.stride) % nFrames;
      let offset = si * perStep;

      // Relative site positions (future clip sites minus current root pos)
      const siteBase = futureIdx * nClipSites * 3;
      for (let j = 0; j < nClipSites; j++) {
        out[offset + j * 3] = (siteXpos[siteBase + j * 3] ?? 0) - curRootX;
        out[offset + j * 3 + 1] = (siteXpos[siteBase + j * 3 + 1] ?? 0) - curRootY;
        out[offset + j * 3 + 2] = (siteXpos[siteBase + j * 3 + 2] ?? 0) - curRootZ;
      }
      offset += nClipSites * 3;

      // Delta root position (future clip qpos[0:3] minus current root pos)
      const qposBase = futureIdx * nq;
      out[offset] = (clipQpos[qposBase] ?? 0) - curRootX;
      out[offset + 1] = (clipQpos[qposBase + 1] ?? 0) - curRootY;
      out[offset + 2] = (clipQpos[qposBase + 2] ?? 0) - curRootZ;
      offset += 3;

      // Future root velocity (clip qvel[0:3], no subtraction)
      const qvelBase = futureIdx * nv;
      out[offset] = clipQvel[qvelBase] ?? 0;
      out[offset + 1] = clipQvel[qvelBase + 1] ?? 0;
      out[offset + 2] = clipQvel[qvelBase + 2] ?? 0;
      offset += 3;

      // Phase
      out[offset] = futureIdx / Math.max(nFrames - 1, 1);
    }

    return out;
  }
}
