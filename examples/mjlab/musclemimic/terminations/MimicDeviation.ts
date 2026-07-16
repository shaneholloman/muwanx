import { TerminationBase, type TerminationConfig } from 'mjswan/termination';
import type { PolicyState, PolicyRunner } from 'mjswan/types';
import { loadNpz } from 'mjswan/npz';

// ---------------------------------------------------------------------------
// Clip (minimal: only site_xpos + site_names needed)
// ---------------------------------------------------------------------------

type MimicClipBrief = {
  siteXpos: Float32Array;
  clipSiteNames: string[];
  nFrames: number;
  nClipSites: number;
};

// ---------------------------------------------------------------------------
// MimicDeviation termination
//
// Mirrors _mimic_early_termination in mimic_mjlab_env.py:
//   terminate when mean(||clip_site_pos - body_pos||) > site_err_threshold
//   OR ||qpos[:3] - centroid(clip_sites)||  > root_err_threshold
// ---------------------------------------------------------------------------

export class MimicDeviation extends TerminationBase {
  private clip: MimicClipBrief | null = null;
  private bodyIds: number[] | null = null;
  private clipSiteIds: number[] | null = null;
  private readonly siteNames: string[];
  private readonly bodyNames: string[];
  private readonly fps: number;
  private readonly siteErrThreshold: number;
  private readonly rootErrThreshold: number;

  constructor(runner: PolicyRunner, config: TerminationConfig) {
    super(runner, config);
    const p = config.params ?? {};
    this.siteNames = (p.site_names as string[] | undefined) ?? [];
    this.bodyNames = (p.body_names as string[] | undefined) ?? [];
    this.fps = (p.fps as number | undefined) ?? 100;
    this.siteErrThreshold = (p.site_err_threshold as number | undefined) ?? 1.0;
    this.rootErrThreshold = (p.root_err_threshold as number | undefined) ?? 0.3;

    // The app feeds the clip bytes as the 'mimic_clip' motion; read that slot
    // via the runner (no fetch — the engine no longer serves paths). ADR 0004 §10.
    void this.loadClip();
  }

  private async loadClip(): Promise<void> {
    try {
      const bytes = await this.runner.getMotionData('mimic_clip');
      if (!bytes) return;
      const npz = await loadNpz(bytes);
      const siteEntry = npz['site_xpos'];
      const siteNamesEntry = npz['site_names'];
      if (!siteEntry) {
        console.warn('[MimicDeviation] NPZ missing site_xpos');
        return;
      }
      this.clip = {
        siteXpos: siteEntry.data,
        clipSiteNames: siteNamesEntry?.strings ?? [],
        nFrames: siteEntry.shape[0] ?? 0,
        nClipSites: siteEntry.shape[1] ?? 0,
      };
    } catch (e) {
      console.warn('[MimicDeviation] Failed to load clip:', e);
    }
  }

  reset(): void {
    // nothing to reset — clip and body IDs persist across episodes
  }

  private resolveBodyIds(): number[] {
    if (this.bodyIds !== null) return this.bodyIds;
    const ctx = this.runner.getContext();
    if (!ctx?.mjModel) return this.bodyNames.map(() => -1);
    const bodyObjType = (ctx.mujoco.mjtObj?.mjOBJ_BODY?.value) ?? 1;
    this.bodyIds = this.bodyNames.map(
      (n) => ctx.mujoco.mj_name2id(ctx.mjModel!, bodyObjType, n),
    );
    return this.bodyIds;
  }

  private resolveClipSiteIds(): number[] {
    if (this.clipSiteIds !== null) return this.clipSiteIds;
    if (!this.clip) return this.siteNames.map(() => -1);
    this.clipSiteIds = this.siteNames.map((n) => this.clip!.clipSiteNames.indexOf(n));
    return this.clipSiteIds;
  }

  evaluate(_state: PolicyState): boolean {
    if (!this.clip || this.clip.nFrames === 0) return false;
    const ctx = this.runner.getContext();
    if (!ctx?.mjData) return false;
    const { mjData } = ctx;

    const simTime = mjData.time ?? 0;
    const frameIdx = this.fps > 0
      ? Math.floor(simTime * this.fps) % this.clip.nFrames
      : 0;

    const bodyIds = this.resolveBodyIds();
    const clipSiteIds = this.resolveClipSiteIds();
    const xpos = mjData.xpos;
    const qpos = mjData.qpos;
    if (!xpos || !qpos || bodyIds.length === 0) return false;

    const stride = this.clip.nClipSites * 3;
    const siteBase = frameIdx * stride;

    let totalDist = 0;
    let count = 0;
    let centX = 0, centY = 0, centZ = 0;

    for (let i = 0; i < clipSiteIds.length; i++) {
      const ci = clipSiteIds[i];
      const bi = bodyIds[i];
      if (ci < 0 || bi < 0) continue;

      const base = siteBase + ci * 3;
      const tx = this.clip.siteXpos[base] ?? 0;
      const ty = this.clip.siteXpos[base + 1] ?? 0;
      const tz = this.clip.siteXpos[base + 2] ?? 0;
      const dx = tx - (xpos[bi * 3] ?? 0);
      const dy = ty - (xpos[bi * 3 + 1] ?? 0);
      const dz = tz - (xpos[bi * 3 + 2] ?? 0);

      totalDist += Math.sqrt(dx * dx + dy * dy + dz * dz);
      centX += tx; centY += ty; centZ += tz;
      count++;
    }

    if (count === 0) return false;

    if (totalDist / count > this.siteErrThreshold) return true;

    // Root position vs centroid of target sites (mirrors tgt.mean(dim=1) in Python)
    centX /= count; centY /= count; centZ /= count;
    const rx = (qpos[0] ?? 0) - centX;
    const ry = (qpos[1] ?? 0) - centY;
    const rz = (qpos[2] ?? 0) - centZ;
    return Math.sqrt(rx * rx + ry * ry + rz * rz) > this.rootErrThreshold;
  }
}
