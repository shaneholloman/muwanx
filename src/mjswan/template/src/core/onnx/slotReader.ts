/**
 * `createSlotReader`: the native side of every traced term (ADR 0005 §6).
 *
 * A traced graph is a pure function of the state it declared as `input_slots`.
 * Filling those slots is explicitly *not* traced — state collection stays native
 * (§6's `collectRawState`) — so this module is the one place that reproduces
 * mjlab's `EntityData` semantics against `mjModel`/`mjData`. Get a field wrong
 * here and the graph computes the right function of the wrong numbers, which no
 * amount of Python-side parity checking can catch.
 *
 * Three slot shapes, matching `mjswan.compile.tracer.slot_to_json`:
 *
 * | slot                | source                                            |
 * |---------------------|---------------------------------------------------|
 * | `{entity, field}`     | an `Entity.data.<field>` tensor (table below)    |
 * | `{sensor}`            | `mjData.sensordata[adr … adr+dim]`              |
 * | `{sensor, field}`     | a structured sensor's field, cast here (`raycast.ts`) |
 * | `{command, field}`    | a live `OnnxCommand`'s state field               |
 *
 * The `{sensor, field}` shape is mjlab's `RayCastSensor` (a height scan): its data
 * is ray hits, not a `sensordata` window, so it is recomputed rather than read —
 * see `raycast.ts`.
 *
 * **The whole field, not a slice.** The graph carries the term's own indexing
 * (mjlab's managers resolve `SceneEntityCfg` name patterns to ids at build time,
 * and the tracer bakes those ids in), so a slot must supply the entity's
 * *complete* field in mjlab's element order — `site_pos_w` is every site of the
 * entity, flattened, not the one site the term happens to read.
 *
 * **Element order** is mjlab's, which is MJCF spec order within an entity
 * (`EntityIndexing` is built from `spec.joints` / `spec.sites`). Model ids are
 * assigned in attach order, so ascending model id within an entity reproduces it.
 * Joints exclude the free joint: mjlab keeps it in `free_joint_q_adr`, and
 * `joint_pos`/`joint_vel` cover the non-free joints only.
 *
 * **Entity scoping.** `MjSpec.attach(prefix=f"{name}/")` means a scene assembled
 * by mjlab prefixes every element with its entity name. A model exported from a
 * plain MJCF (the `set_trace_env` path, where the browser model and the trace
 * env are built from the same spec but only the latter goes through mjlab's
 * scene) has no prefix, so resolution falls back to the whole model — correct
 * because such a scene is single-entity by construction.
 *
 * The one entity mjlab itself attaches prefix-free is `terrain`, so in an
 * mjlab-assembled scene a `terrain` slot resolves to nothing and reads as
 * unavailable. Deliberate: the alternative is guessing that unprefixed elements
 * mean the terrain, which would also silently swallow a misspelled entity name.
 * No traced term reads terrain state as an *entity*: mjlab's own MDP terms take a
 * robot or object `SceneEntityCfg`, and the height scan reaches the terrain through
 * a `RayCastSensor` slot rather than `terrain.data`. So this costs nothing today
 * and fails visibly (a named warning from the caller) if that ever changes.
 *
 * **float64 → float32** happens here, at the read site: mjData is float64 and
 * ORT-Web wants Float32Array.
 *
 * Fields are implemented explicitly and an unknown one returns null (the caller
 * then warns and holds its previous value) rather than being approximated. The
 * fields below are the ones the traced terms actually declare across mjlab's
 * default tasks plus the migrated examples; the derived siblings that share a
 * code path come along for free.
 *
 * | field                   | mjlab definition                                  |
 * |-------------------------|---------------------------------------------------|
 * | `joint_pos`             | `qpos[joint_q_adr]`                               |
 * | `joint_pos_biased`      | `joint_pos + encoder_bias`                        |
 * | `joint_vel`             | `qvel[joint_v_adr]`                               |
 * | `root_link_pos_w`       | `xpos[root_body]`                                 |
 * | `root_link_quat_w`      | `xquat[root_body]`                                |
 * | `root_link_pose_w`      | pos ++ quat (7)                                   |
 * | `root_link_lin_vel_w`   | `cvel` linear part, de-offset from the subtree COM |
 * | `root_link_ang_vel_w`   | `cvel[root_body][0:3]`                            |
 * | `root_link_vel_w`       | lin ++ ang (6)                                    |
 * | `root_link_lin_vel_b`   | `quat⁻¹ · lin_vel_w`                              |
 * | `root_link_ang_vel_b`   | `quat⁻¹ · ang_vel_w`                              |
 * | `gravity_vec_w`         | the constant `(0, 0, -1)`                         |
 * | `projected_gravity_b`   | `quat⁻¹ · (0, 0, -1)`                             |
 * | `heading_w`             | `atan2((quat · x̂)ᵧ, (quat · x̂)ₓ)`                 |
 * | `site_pos_w`            | `site_xpos[site_ids]`, flattened                  |
 */

import { quatApply, quatApplyInv } from '../observation/math';
import { RaycastSensor, isRaycastField, type RaycastSensorDescriptor } from './raycast';
import type { OnnxInputSlot, SlotReader } from './session';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MainModule = import('mujoco').MainModule;

/** A command term that can hand back one of its traced state fields by name. */
export interface CommandStateSource {
  getStateField(field: string): Float32Array | null;
}

/**
 * Live simulation handles. Shaped to match `PolicyRunnerContext` so the runtime
 * can pass `() => runner.getContext()` straight through.
 */
export type SlotReaderContext = {
  mjModel: MjModel | null;
  mjData: MjData | null;
  /** Needed to cast a `RayCastSensor`'s rays (`mj_ray`); absent before load. */
  mujoco?: MainModule | null;
  commandManager?: { getTerm(name: string): unknown } | null;
};

export type SlotReaderOptions = {
  /**
   * Per-joint encoder bias by *unprefixed* joint name, for `joint_pos_biased`.
   *
   * mjlab randomizes `encoder_bias` per episode; a web bundle bakes whatever the
   * export captured into `policy.json`, in policy-action order — hence by name
   * rather than by index, since a slot needs it in entity-joint order. Omitted
   * means no bias, which is the common case and makes `joint_pos_biased`
   * identical to `joint_pos` exactly as mjlab's own zero-bias default does.
   */
  jointBias?: (jointName: string) => number;
  /**
   * Descriptors for the structured sensors the config's slots name, by sensor
   * name. A function because they arrive with the policy, while the reader is
   * built once with the runtime.
   */
  raycastSensors?: () => Record<string, RaycastSensorDescriptor>;
};

/** Everything about one entity that resolving its fields needs, computed once. */
type EntityIndex = {
  /** qpos addresses of the entity's non-free joints, in spec order. */
  qposAdr: number[];
  /** qvel (dof) addresses of the same joints, same order. */
  qvelAdr: number[];
  /** Encoder bias per joint, aligned to `qposAdr`. */
  jointBias: Float32Array;
  /** mjlab's `root_body_id`: the entity's first non-world body. */
  rootBodyId: number;
  /** Model site ids belonging to the entity, in spec order. */
  siteIds: number[];
};

/** Widths of a joint's qpos/qvel block by `mjtJoint` (free, ball, slide, hinge). */
const QPOS_WIDTH = [7, 4, 1, 1];
const DOF_WIDTH = [6, 3, 1, 1];
const MJ_JNT_FREE = 0;

function decodeNames(mjModel: MjModel, count: number, adr: ArrayLike<number>): string[] {
  const bytes = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let i = 0; i < count; i++) {
    const start = adr[i];
    let end = start;
    while (end < bytes.length && bytes[end] !== 0) end++;
    names.push(decoder.decode(bytes.subarray(start, end)));
  }
  return names;
}

/**
 * Indices of the elements belonging to `entity`, in ascending model id.
 *
 * Falling back to the whole model is gated on `prefixed` — whether this model was
 * assembled by mjlab at all — and not on the scoped match coming up empty. An
 * entity legitimately having none of some element kind is common (mjlab's `cube`
 * has no sites, `terrain` has no joints), and answering those with every *other*
 * entity's elements is exactly the silent-wrong-numbers failure this module
 * exists to avoid.
 */
function scopedIndices(
  names: string[],
  entity: string | null | undefined,
  prefixed: boolean,
): number[] {
  const all = names.map((_, i) => i);
  // No asset named in the term's params, or a plain-MJCF model: the whole model.
  if (!entity || !prefixed) return all;
  const prefix = `${entity}/`;
  return all.filter(i => names[i].startsWith(prefix));
}

/** The element's name with its `entity/` prefix removed, as mjlab reports it. */
function unprefixed(name: string): string {
  const slash = name.lastIndexOf('/');
  return slash < 0 ? name : name.slice(slash + 1);
}

function buildEntityIndex(
  mjModel: MjModel,
  entity: string | null | undefined,
  options: SlotReaderOptions,
): EntityIndex {
  const jointNames = decodeNames(mjModel, mjModel.njnt, mjModel.name_jntadr);
  const bodyNames = decodeNames(mjModel, mjModel.nbody, mjModel.name_bodyadr);
  const siteNames = decodeNames(mjModel, mjModel.nsite, mjModel.name_siteadr);
  // One verdict for the model, not per element kind: it either came through
  // mjlab's `attach(prefix=f"{name}/")` or it didn't.
  const prefixed = [...jointNames, ...bodyNames, ...siteNames].some(n => n.includes('/'));

  const qposAdr: number[] = [];
  const qvelAdr: number[] = [];
  const bias: number[] = [];
  for (const j of scopedIndices(jointNames, entity, prefixed)) {
    const type = mjModel.jnt_type[j];
    if (type === MJ_JNT_FREE) continue; // mjlab keeps the free joint separate
    const qWidth = QPOS_WIDTH[type] ?? 1;
    const vWidth = DOF_WIDTH[type] ?? 1;
    const jointBias = options.jointBias?.(unprefixed(jointNames[j])) ?? 0;
    for (let k = 0; k < qWidth; k++) {
      qposAdr.push(mjModel.jnt_qposadr[j] + k);
      bias.push(jointBias);
    }
    for (let k = 0; k < vWidth; k++) qvelAdr.push(mjModel.jnt_dofadr[j] + k);
  }

  // Skip the worldbody (id 0): mjlab's `bodies` tuple is `spec.bodies[1:]`.
  const bodyIds = scopedIndices(bodyNames, entity, prefixed).filter(i => i !== 0);

  return {
    qposAdr,
    qvelAdr,
    jointBias: Float32Array.from(bias),
    rootBodyId: bodyIds.length > 0 ? bodyIds[0] : -1,
    siteIds: scopedIndices(siteNames, entity, prefixed),
  };
}

function gather(source: ArrayLike<number>, addresses: readonly number[]): Float32Array {
  const out = new Float32Array(addresses.length);
  for (let i = 0; i < addresses.length; i++) out[i] = source[addresses[i]] ?? 0;
  return out;
}

function vec3At(source: ArrayLike<number>, index: number): Float32Array {
  const base = index * 3;
  return new Float32Array([source[base] ?? 0, source[base + 1] ?? 0, source[base + 2] ?? 0]);
}

function quatAt(source: ArrayLike<number>, index: number): Float32Array {
  const base = index * 4;
  return new Float32Array([
    source[base] ?? 1,
    source[base + 1] ?? 0,
    source[base + 2] ?? 0,
    source[base + 3] ?? 0,
  ]);
}

function concat(...parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

/**
 * mjlab's `root_link_vel_w`: `cvel` is expressed about the body's subtree COM, so
 * the linear part needs the rotational offset to the body origin removed
 * (mjlab's `compute_velocity_from_cvel`). Returns lin ++ ang, 6 values.
 */
function rootLinkVelW(rootBodyId: number, mjData: MjData): Float32Array {
  const pos = vec3At(mjData.xpos, rootBodyId);
  const com = vec3At(mjData.subtree_com, rootBodyId);
  const base = rootBodyId * 6;
  // MuJoCo's cvel packs angular first, then linear.
  const ang = angVelW(mjData, rootBodyId);
  const linC = new Float32Array([
    mjData.cvel[base + 3] ?? 0,
    mjData.cvel[base + 4] ?? 0,
    mjData.cvel[base + 5] ?? 0,
  ]);
  const ox = com[0] - pos[0];
  const oy = com[1] - pos[1];
  const oz = com[2] - pos[2];
  const lin = new Float32Array([
    linC[0] - (ang[1] * oz - ang[2] * oy),
    linC[1] - (ang[2] * ox - ang[0] * oz),
    linC[2] - (ang[0] * oy - ang[1] * ox),
  ]);
  return concat(lin, ang);
}

type FieldReader = (index: EntityIndex, mjData: MjData) => Float32Array | null;

/**
 * Wrap a reader that needs the entity's root body.
 *
 * An entity with no body of its own has no root pose to report; returning null
 * makes the caller hold its previous value and warn, where indexing at -1 would
 * quietly hand the graph a plausible-looking vector of zeros.
 */
function rootField(read: (root: number, mjData: MjData) => Float32Array): FieldReader {
  return (index, mjData) => (index.rootBodyId < 0 ? null : read(index.rootBodyId, mjData));
}

/** `cvel`'s angular part for a body — packed before the linear part in MuJoCo. */
function angVelW(mjData: MjData, root: number): Float32Array {
  const base = root * 6;
  return new Float32Array([
    mjData.cvel[base] ?? 0,
    mjData.cvel[base + 1] ?? 0,
    mjData.cvel[base + 2] ?? 0,
  ]);
}

const FIELD_READERS: Record<string, FieldReader> = {
  joint_pos: (index, mjData) => gather(mjData.qpos, index.qposAdr),
  joint_pos_biased: (index, mjData) => {
    const out = gather(mjData.qpos, index.qposAdr);
    for (let i = 0; i < out.length; i++) out[i] += index.jointBias[i] ?? 0;
    return out;
  },
  joint_vel: (index, mjData) => gather(mjData.qvel, index.qvelAdr),

  root_link_pos_w: rootField((root, mjData) => vec3At(mjData.xpos, root)),
  root_link_quat_w: rootField((root, mjData) => quatAt(mjData.xquat, root)),
  root_link_pose_w: rootField((root, mjData) =>
    concat(vec3At(mjData.xpos, root), quatAt(mjData.xquat, root)),
  ),

  root_link_vel_w: rootField(rootLinkVelW),
  root_link_lin_vel_w: rootField((root, mjData) => rootLinkVelW(root, mjData).slice(0, 3)),
  // mjlab reads the angular part straight off cvel rather than through
  // compute_velocity_from_cvel — the COM offset only shifts the linear part.
  root_link_ang_vel_w: rootField((root, mjData) => angVelW(mjData, root)),
  root_link_lin_vel_b: rootField((root, mjData) =>
    Float32Array.from(
      quatApplyInv(quatAt(mjData.xquat, root), rootLinkVelW(root, mjData).subarray(0, 3)),
    ),
  ),
  root_link_ang_vel_b: rootField((root, mjData) =>
    Float32Array.from(quatApplyInv(quatAt(mjData.xquat, root), angVelW(mjData, root))),
  ),

  // mjlab's is a constant, not `mjModel.opt.gravity`: `entity.py` fills it with
  // (0, 0, -1) and terms use it as the world's down direction.
  gravity_vec_w: () => new Float32Array([0, 0, -1]),
  projected_gravity_b: rootField((root, mjData) =>
    Float32Array.from(quatApplyInv(quatAt(mjData.xquat, root), [0, 0, -1])),
  ),
  heading_w: rootField((root, mjData) => {
    const forward = quatApply(quatAt(mjData.xquat, root), [1, 0, 0]);
    return new Float32Array([Math.atan2(forward[1], forward[0])]);
  }),

  site_pos_w: (index, mjData) => {
    const out = new Float32Array(index.siteIds.length * 3);
    for (let i = 0; i < index.siteIds.length; i++) {
      out.set(vec3At(mjData.site_xpos, index.siteIds[i]), i * 3);
    }
    return out;
  },
};

/** Whether an `Entity.data` field can be served — useful for a build-time check. */
export function isReadableEntityField(field: string): boolean {
  return field in FIELD_READERS;
}

/**
 * Resolve a sensor's `sensordata` window, prefix-tolerantly.
 *
 * The build records mjlab's prefixed name (`robot/imu_lin_vel`); a model exported
 * from plain MJCF has the bare name. Try both before giving up.
 */
function sensorWindow(mjModel: MjModel, sensor: string): { adr: number; dim: number } | null {
  const names = decodeNames(mjModel, mjModel.nsensor, mjModel.name_sensoradr);
  let idx = names.indexOf(sensor);
  if (idx < 0) {
    const bare = unprefixed(sensor);
    idx = names.findIndex(name => name === bare || unprefixed(name) === bare);
  }
  if (idx < 0) return null;
  return { adr: mjModel.sensor_adr[idx], dim: mjModel.sensor_dim[idx] };
}

function isCommandStateSource(value: unknown): value is CommandStateSource {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as CommandStateSource).getStateField === 'function'
  );
}

/**
 * Build the `SlotReader` every ONNX-backed term (observation, termination,
 * command, event) reads its graph inputs through.
 *
 * `getContext` is called per read so a scene reload — which replaces `mjModel`
 * and `mjData` — is picked up without rebuilding the reader; the per-entity index
 * is cached against the model it was derived from and recomputed when that
 * changes.
 */
export function createSlotReader(
  getContext: () => SlotReaderContext | null,
  options: SlotReaderOptions = {},
): SlotReader {
  let cachedModel: MjModel | null = null;
  const indices = new Map<string, EntityIndex>();
  // One caster per sensor, kept because it holds the per-model frame resolution
  // and the ray buffers; a height scan is ~200 rays every control step.
  const casters = new Map<string, RaycastSensor | null>();

  const readRaycast = (
    sensor: string,
    field: string,
    context: SlotReaderContext,
  ): Float32Array | null => {
    const { mjModel, mjData, mujoco } = context;
    if (!mjModel || !mjData || !mujoco) return null;
    if (!casters.has(sensor)) {
      const descriptor = options.raycastSensors?.()[sensor];
      if (!descriptor) {
        console.warn(
          `[slotReader] no raycast descriptor for sensor "${sensor}"; the build ` +
            'did not emit one.',
        );
      }
      casters.set(sensor, descriptor ? new RaycastSensor(mujoco, descriptor) : null);
    }
    const caster = casters.get(sensor);
    if (!caster) return null;
    if (!isRaycastField(field)) {
      console.warn(`[slotReader] raycast sensor "${sensor}" cannot serve "${field}".`);
      return null;
    }
    return caster.read(field, mjModel, mjData);
  };

  const indexFor = (mjModel: MjModel, entity: string | null | undefined): EntityIndex => {
    if (mjModel !== cachedModel) {
      cachedModel = mjModel;
      indices.clear();
    }
    const key = entity ?? '';
    let index = indices.get(key);
    if (!index) {
      index = buildEntityIndex(mjModel, entity, options);
      indices.set(key, index);
    }
    return index;
  };

  return (slot: OnnxInputSlot): Float32Array | null => {
    const context = getContext();
    if (!context) return null;

    if (slot.command) {
      const term = context.commandManager?.getTerm(slot.command);
      if (!isCommandStateSource(term)) return null;
      return term.getStateField(slot.field ?? '');
    }

    const { mjModel, mjData } = context;
    if (!mjModel || !mjData) return null;

    if (slot.sensor) {
      if (slot.field) {
        // A *field* of a structured sensor — mjlab's `RayCastSensor`, whose data is
        // ray hits rather than a `sensordata` window. Never fall through to the
        // builtin path: this sensor has no window there to find.
        return readRaycast(slot.sensor, slot.field, context);
      }
      const window = sensorWindow(mjModel, slot.sensor);
      if (!window) return null;
      const out = new Float32Array(window.dim);
      for (let i = 0; i < window.dim; i++) out[i] = mjData.sensordata[window.adr + i] ?? 0;
      return out;
    }

    const read = slot.field ? FIELD_READERS[slot.field] : undefined;
    if (!read) return null;
    return read(indexFor(mjModel, slot.entity), mjData);
  };
}
