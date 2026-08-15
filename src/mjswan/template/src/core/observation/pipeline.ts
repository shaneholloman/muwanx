/**
 * The native half of mjlab's compute → noise → clip → scale → delay → history: noise and
 * delay are training-only, history belongs to the group. Clip → scale runs identically for
 * graph-backed and native terms, so the two cannot drift.
 */

/** Per-element or scalar scaling, as emitted by the build. */
export type ObservationScale = number | number[];

export type ObservationClip = [number, number];

export interface ObservationPipelineConfig {
  scale?: ObservationScale;
  clip?: ObservationClip;
}

/** Apply clip then scale in place — mjlab's order, since limits are in term units. */
export function applyObservationPipeline(
  values: Float32Array,
  config: ObservationPipelineConfig,
): Float32Array {
  const { clip, scale } = config;
  if (clip) {
    const [min, max] = clip;
    for (let i = 0; i < values.length; i++) {
      values[i] = Math.min(max, Math.max(min, values[i]));
    }
  }
  if (typeof scale === 'number') {
    for (let i = 0; i < values.length; i++) values[i] *= scale;
  } else if (Array.isArray(scale)) {
    // The build emits one entry per element, so a short array means a stale config.
    for (let i = 0; i < values.length && i < scale.length; i++) values[i] *= scale[i];
  }
  return values;
}

/** Conform `values` to `size`, zero-padding or truncating as needed. */
export function conformToSize(values: Float32Array, size: number): Float32Array {
  if (values.length === size) return values;
  const out = new Float32Array(size);
  out.set(values.subarray(0, Math.min(values.length, size)));
  return out;
}
