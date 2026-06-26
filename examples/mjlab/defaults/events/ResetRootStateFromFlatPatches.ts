import { EventBase, type EventConfig, type EventContext } from 'mjswan/event';

/**
 * Reset root state by placing the robot on a random flat terrain patch.
 *
 * This is a mjswan browser enhancement (not an mjlab term): mjlab trains with
 * many parallel envs spread across terrain tiles, so per-env spawn jitter is
 * small.  The browser has one env, so we spawn it on a random flat patch to
 * cover the whole terrain.  It reads ``terrainData.flat_patches`` from the
 * scene config — an engine capability outside bounded linear algebra, hence a
 * ``ts_src`` term (not declarative).  See ADR 0003.
 *
 * Params:
 *   patch_name: string (default: "spawn")
 *   pose_range: optional { yaw?: [min, max], x?/y?/z?: [min, max] }
 *
 * Falls back to a uniform pose-range offset when no patch data is available.
 */
export class ResetRootStateFromFlatPatches extends EventBase {
  private readonly patchName: string;
  private readonly poseRange: Record<string, [number, number]>;

  constructor(config: EventConfig) {
    super(config);
    this.patchName = (config.params?.patch_name as string) ?? 'spawn';
    this.poseRange =
      (config.params?.pose_range as Record<string, [number, number]>) ?? {};
  }

  onReset(context: EventContext): void {
    const { mjModel, mjData, terrainData } = context;
    if (!mjModel || !mjData) return;

    const freeJointIdx = this._findFreeJoint(mjModel);
    if (freeJointIdx === -1) return;
    const qposAdr = mjModel.jnt_qposadr[freeJointIdx];

    const patches = terrainData?.flat_patches?.[this.patchName];
    if (!patches || patches.length === 0) {
      // No patch data: fall back to uniform pose-range offsets.
      mjData.qpos[qposAdr + 0] += this._sample('x');
      mjData.qpos[qposAdr + 1] += this._sample('y');
      mjData.qpos[qposAdr + 2] += this._sample('z');
    } else {
      const patch = patches[Math.floor(Math.random() * patches.length)];
      const defaultZ = mjData.qpos[qposAdr + 2];
      mjData.qpos[qposAdr + 0] = patch[0];
      mjData.qpos[qposAdr + 1] = patch[1];
      mjData.qpos[qposAdr + 2] = patch[2] + defaultZ;
    }

    const yawRange = this.poseRange['yaw'] ?? [-Math.PI, Math.PI];
    const yaw = yawRange[0] + Math.random() * (yawRange[1] - yawRange[0]);
    this._applyYawRotation(mjData.qpos, qposAdr + 3, yaw);
  }

  private _sample(key: string): number {
    const range = this.poseRange[key];
    if (!range) return 0;
    return range[0] + Math.random() * (range[1] - range[0]);
  }

  private _findFreeJoint(mjModel: import('mujoco').MjModel): number {
    for (let i = 0; i < mjModel.njnt; i++) {
      if (mjModel.jnt_type[i] === 0) return i; // mjJNT_FREE = 0
    }
    return -1;
  }

  private _applyYawRotation(qpos: Float64Array, quatAdr: number, yaw: number): void {
    const hw = Math.cos(yaw / 2);
    const hz = Math.sin(yaw / 2);
    const w = qpos[quatAdr + 0];
    const x = qpos[quatAdr + 1];
    const y = qpos[quatAdr + 2];
    const z = qpos[quatAdr + 3];
    qpos[quatAdr + 0] = hw * w - hz * z;
    qpos[quatAdr + 1] = hw * x + hz * y;
    qpos[quatAdr + 2] = hw * y - hz * x;
    qpos[quatAdr + 3] = hw * z + hz * w;
  }
}
