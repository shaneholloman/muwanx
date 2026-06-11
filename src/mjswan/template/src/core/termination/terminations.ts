import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { PolicyState } from '../policy/types';
import { CustomTerminations } from './custom_terminations';
import { getCommandManager, isTrackingSource } from '../command';
import { quatApplyInv } from '../observation/math';

function getBodyIdByName(mjModel: import('mujoco').MjModel, bodyName: string): number {
  for (let i = 0; i < mjModel.nbody; i++) {
    const name = mjModel.body(i).name;
    if (name === bodyName || name.endsWith(`/${bodyName}`)) return i;
  }
  return -1;
}

/**
 * Terminate when the episode step count exceeds max_episode_length.
 *
 * mjlab: env.episode_length_buf >= env.max_episode_length
 */
export class TimeOut extends TerminationBase {
  private stepCount = 0;
  private maxSteps: number;

  constructor(config: TerminationConfig) {
    super(config);
    const params = config.params ?? {};
    this.maxSteps = (params.max_episode_length as number) ?? Infinity;
  }

  evaluate(): boolean {
    this.stepCount++;
    return this.stepCount >= this.maxSteps;
  }

  reset(): void {
    this.stepCount = 0;
  }
}

/**
 * Terminate when the robot's orientation exceeds a limit angle.
 *
 * Uses the projected gravity vector from PolicyState to compute the
 * angle between the robot's up axis and world up.
 *
 * mjlab: torch.acos(-projected_gravity[:, 2]).abs() > limit_angle
 */
export class BadOrientation extends TerminationBase {
  private limitAngle: number;

  constructor(config: TerminationConfig) {
    super(config);
    const params = config.params ?? {};
    this.limitAngle = (params.limit_angle as number) ?? 1.0;
  }

  evaluate(state: PolicyState): boolean {
    const rootQuat = state.rootQuat;
    if (!rootQuat || rootQuat.length < 4) return false;

    // Compute tilt angle from quaternion.
    const x = rootQuat[1];
    const y = rootQuat[2];
    const gz = 1.0 - 2.0 * (x * x + y * y);
    const angle = Math.acos(Math.max(-1.0, Math.min(1.0, gz)));

    return Math.abs(angle) > this.limitAngle;
  }
}

/**
 * Terminate when the robot's root height drops below a minimum.
 *
 * mjlab: asset.data.root_link_pos_w[:, 2] < minimum_height
 */
export class RootHeightBelowMinimum extends TerminationBase {
  private minimumHeight: number;

  constructor(config: TerminationConfig) {
    super(config);
    const params = config.params ?? {};
    this.minimumHeight = (params.minimum_height as number) ?? 0.0;
  }

  evaluate(state: PolicyState): boolean {
    const rootPos = state.rootPos;
    if (!rootPos || rootPos.length < 3) return false;
    return rootPos[2] < this.minimumHeight;
  }
}

/**
 * Terminate when the robot leaves a fixed terrain footprint.
 *
 * mjlab: tasks/velocity/mdp/terminations.out_of_terrain_bounds
 * Params: limit_x, limit_y (computed at Python build time from the
 * TerrainGeneratorCfg).  Absent params disable the check.
 */
export class OutOfTerrainBounds extends TerminationBase {
  private readonly limitX: number;
  private readonly limitY: number;

  constructor(config: TerminationConfig) {
    super(config);
    this.limitX = (config.params?.limit_x as number | undefined) ?? Infinity;
    this.limitY = (config.params?.limit_y as number | undefined) ?? Infinity;
  }

  evaluate(state: PolicyState): boolean {
    const pos = state.rootPos;
    if (!pos || pos.length < 3) return false;
    return Math.abs(pos[0]) > this.limitX || Math.abs(pos[1]) > this.limitY;
  }
}

/**
 * Terminate when the robot's displacement from its spawn origin exceeds the
 * sub-terrain boundary.  Captures the spawn from the first post-reset root
 * position.  Skips the first two steps to avoid stale-position triggers.
 *
 * mjlab: tasks/velocity/mdp/terminations.terrain_edge_reached
 * Params: half_x, half_y (from TerrainGeneratorCfg.size), threshold_fraction (0.95).
 */
export class TerrainEdgeReached extends TerminationBase {
  private readonly halfX: number;
  private readonly halfY: number;
  private readonly thresholdFraction: number;
  private spawnX: number | null = null;
  private spawnY: number | null = null;
  private stepCount = 0;

  constructor(config: TerminationConfig) {
    super(config);
    this.halfX = (config.params?.half_x as number | undefined) ?? Infinity;
    this.halfY = (config.params?.half_y as number | undefined) ?? Infinity;
    this.thresholdFraction = (config.params?.threshold_fraction as number | undefined) ?? 0.95;
  }

  evaluate(state: PolicyState): boolean {
    this.stepCount++;
    const pos = state.rootPos;
    if (!pos || pos.length < 3) return false;
    if (this.spawnX === null || this.spawnY === null) {
      this.spawnX = pos[0];
      this.spawnY = pos[1];
    }
    if (this.stepCount <= 2) return false;
    return (
      Math.abs(pos[0] - this.spawnX) > this.halfX * this.thresholdFraction
      || Math.abs(pos[1] - this.spawnY) > this.halfY * this.thresholdFraction
    );
  }

  reset(): void {
    this.stepCount = 0;
    this.spawnX = null;
    this.spawnY = null;
  }
}

/**
 * Terminate when the tracking-reference anchor body's z position diverges
 * from the current robot anchor body's z position by more than `threshold`.
 */
export class BadAnchorPosZOnly extends TerminationBase {
  private readonly threshold: number;

  constructor(config: TerminationConfig) {
    super(config);
    this.threshold = (config.params?.threshold as number | undefined) ?? Infinity;
  }

  evaluate(_state: PolicyState): boolean {
    const tracking = getCommandManager().getTerm('motion');
    const context = getCommandManager().getContext();
    const mjModel = context?.mjModel ?? null;
    const mjData = context?.mjData ?? null;
    if (!isTrackingSource(tracking) || !tracking.isReady() || !mjModel || !mjData) {
      return false;
    }
    const anchorPos = tracking.getAnchorPos();
    const anchorName = tracking.getAnchorBodyName();
    if (!anchorPos || anchorPos.length < 3 || !anchorName) return false;
    const anchorId = getBodyIdByName(mjModel, anchorName);
    if (anchorId < 0) return false;
    const currentAnchorZ = mjData.xpos[anchorId * 3 + 2] ?? 0.0;
    return Math.abs(anchorPos[2] - currentAnchorZ) > this.threshold;
  }
}

/**
 * Terminate when the projected-gravity z component of the tracking reference
 * anchor diverges from the current robot's by more than `threshold`.
 */
export class BadAnchorOri extends TerminationBase {
  private readonly threshold: number;

  constructor(config: TerminationConfig) {
    super(config);
    this.threshold = (config.params?.threshold as number | undefined) ?? Infinity;
  }

  evaluate(_state: PolicyState): boolean {
    const tracking = getCommandManager().getTerm('motion');
    const context = getCommandManager().getContext();
    const mjModel = context?.mjModel ?? null;
    const mjData = context?.mjData ?? null;
    if (!isTrackingSource(tracking) || !tracking.isReady() || !mjModel || !mjData) {
      return false;
    }
    const anchorQuat = tracking.getAnchorQuat();
    const anchorName = tracking.getAnchorBodyName();
    if (!anchorQuat || anchorQuat.length < 4 || !anchorName) return false;
    const anchorId = getBodyIdByName(mjModel, anchorName);
    if (anchorId < 0) return false;
    const currentAnchorQuat = mjData.xquat.slice(anchorId * 4, anchorId * 4 + 4);
    const gravity: [number, number, number] = [0.0, 0.0, -1.0];
    const motionGravity = quatApplyInv(anchorQuat, gravity);
    const robotGravity = quatApplyInv(currentAnchorQuat, gravity);
    return Math.abs(motionGravity[2] - robotGravity[2]) > this.threshold;
  }
}

/**
 * Terminate when any tracked body's z position diverges from the reference
 * by more than `threshold`.  Defaults to all tracked bodies; restrict via
 * the `body_names` param.
 */
export class BadMotionBodyPosZOnly extends TerminationBase {
  private readonly threshold: number;
  private readonly bodyNames: string[] | null;

  constructor(config: TerminationConfig) {
    super(config);
    this.threshold = (config.params?.threshold as number | undefined) ?? Infinity;
    this.bodyNames = Array.isArray(config.params?.body_names)
      ? config.params!.body_names.filter((v): v is string => typeof v === 'string')
      : null;
  }

  evaluate(_state: PolicyState): boolean {
    const tracking = getCommandManager().getTerm('motion');
    const context = getCommandManager().getContext();
    const mjModel = context?.mjModel ?? null;
    const mjData = context?.mjData ?? null;
    if (!isTrackingSource(tracking) || !tracking.isReady() || !mjModel || !mjData) {
      return false;
    }
    const bodyNames = this.bodyNames && this.bodyNames.length > 0
      ? this.bodyNames
      : tracking.getBodyNames();
    if (bodyNames.length === 0) return false;
    const refBodyPosW = tracking.getBodyPosW();
    if (!refBodyPosW) return false;
    const allTrackingBodies = tracking.getBodyNames();
    for (const bodyName of bodyNames) {
      const bodySlot = allTrackingBodies.indexOf(bodyName);
      const bodyId = getBodyIdByName(mjModel, bodyName);
      if (bodySlot < 0 || bodyId < 0) continue;
      const refZ = refBodyPosW[bodySlot * 3 + 2] ?? 0.0;
      const currentZ = mjData.xpos[bodyId * 3 + 2] ?? 0.0;
      if (Math.abs(refZ - currentZ) > this.threshold) return true;
    }
    return false;
  }
}

/**
 * Terminate when the base angular velocity exceeds `threshold` on any axis.
 */
export class BaseAngVelExceed extends TerminationBase {
  private readonly threshold: number;

  constructor(config: TerminationConfig) {
    super(config);
    this.threshold = (config.params?.threshold as number | undefined) ?? Infinity;
  }

  evaluate(state: PolicyState): boolean {
    const rootAngVel = state.rootAngVel;
    if (!rootAngVel || rootAngVel.length < 3) return false;
    return (
      Math.abs(rootAngVel[0]) > this.threshold
      || Math.abs(rootAngVel[1]) > this.threshold
      || Math.abs(rootAngVel[2]) > this.threshold
    );
  }
}

export type TerminationConstructor = new (config: TerminationConfig) => TerminationBase;

const BuiltinTerminations: Record<string, TerminationConstructor> = {
  TimeOut,
  BadOrientation,
  RootHeightBelowMinimum,
  OutOfTerrainBounds,
  TerrainEdgeReached,
  BadAnchorPosZOnly,
  BadAnchorOri,
  BadMotionBodyPosZOnly,
  BaseAngVelExceed,
};

/**
 * Registry mapping termination class names to constructors.
 */
export const Terminations: Record<string, TerminationConstructor> = {
  ...BuiltinTerminations,
  ...CustomTerminations,
};
