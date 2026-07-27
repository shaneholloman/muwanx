/**
 * The native half of the observation pipeline (ADR 0005 §Decision table).
 *
 * mjlab's order is compute → noise → clip → scale → delay → history. Noise and
 * delay are training-only and dropped (ADR 0003 scope); history is owned by the
 * surrounding group in `PolicyRunner`. What is left here is clip → scale, applied
 * identically to a term whose value came from an ONNX graph and to one the
 * runtime supplies natively, so the two can never drift.
 */

/** Per-element or scalar scaling, as emitted by the build. */
export type ObservationScale = number | number[];

export type ObservationClip = [number, number];

export interface ObservationPipelineConfig {
  scale?: ObservationScale;
  clip?: ObservationClip;
}

/**
 * Apply clip then scale, in place, returning the same array.
 *
 * Clip-before-scale matches mjlab: the limits are expressed in the term's own
 * units, before any scaling is applied.
 */
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
    // A per-element scale shorter than the value is applied elementwise as far
    // as it goes; the build emits one entry per element, so a mismatch means a
    // stale config rather than something to silently pad.
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
