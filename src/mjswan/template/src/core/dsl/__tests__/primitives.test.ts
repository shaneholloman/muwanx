/**
 * Direct unit tests for the DSL primitives added for composition-graph
 * observations (ADR 0003): arithmetic (Div/Sqrt/Sum), Slice, and the
 * motion-reference sources (TrackingRefField/TrackingIsReady).  Each primitive
 * is exercised on its own, not through any task's observation graph.
 */
import { describe, expect, it } from 'vitest';

import type { EvalContext } from '../interpreter';
import { Primitives } from '../primitives';

function expectClose(a: Float32Array, b: ArrayLike<number>, tol = 1e-6): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tol);
}

// A minimal motion command exposing the reference-trajectory buffers the
// sources read (the generic MotionRefSource shape — not tied to any task).
interface RefTerm {
  isReady(): boolean;
  refRootPos: Float32Array[];
  refRootQuat: Float32Array[];
  refJointPos: Float32Array[];
  refIdx: number;
  refLen: number;
}

function ctx(term: RefTerm | null, nJoints = 3): EvalContext {
  const runner = {
    getContext: () => ({
      commandManager: { getTerm: (n: string) => (n === 'motion' ? term : null) },
    }),
    getNumActions: () => nJoints,
    getLastActions: () => new Float32Array(nJoints),
  };
  return { runner, state: {}, params: {} } as unknown as EvalContext;
}

function refTerm(refIdx: number, refLen: number, ready = true): RefTerm {
  const ramp = (base: number, w: number) =>
    new Float32Array(Array.from({ length: w }, (_v, i) => base + i));
  return {
    isReady: () => ready,
    refIdx,
    refLen,
    refRootPos: Array.from({ length: refLen }, (_v, t) => ramp(t * 10, 3)),
    refRootQuat: Array.from({ length: refLen }, () => new Float32Array([1, 0, 0, 0])),
    refJointPos: Array.from({ length: refLen }, (_v, t) => ramp(t * 100, 3)),
  };
}

describe('DSL arithmetic primitives', () => {
  it('Div broadcasts a scalar divisor over a vector', () => {
    expectClose(Primitives.Div([new Float32Array([3, 0, 4]), 2], {}, ctx(null), {}) as Float32Array, [1.5, 0, 2]);
  });
  it('Sqrt is elementwise', () => {
    expectClose(Primitives.Sqrt([new Float32Array([9, 16])], {}, ctx(null), {}) as Float32Array, [3, 4]);
  });
  it('Sum reduces a vector to its scalar sum', () => {
    expect(Primitives.Sum([new Float32Array([3, 0, 4])], {}, ctx(null), {})).toBe(7);
  });
  it('Div/Sqrt/Sum compose to L2-normalize (3,0,4 → 0.6,0,0.8)', () => {
    const v = new Float32Array([3, 0, 4]);
    const norm = Primitives.Sqrt([Primitives.Sum([Primitives.Mul([v, v], {}, ctx(null), {})], {}, ctx(null), {})], {}, ctx(null), {});
    expectClose(Primitives.Div([v, norm], {}, ctx(null), {}) as Float32Array, [0.6, 0, 0.8]);
  });
});

describe('Slice primitive', () => {
  it('extracts the contiguous sub-range [start, start+len)', () => {
    const v = new Float32Array([10, 11, 12, 13, 14, 15]);
    expectClose(Primitives.Slice([v], { start: 2, len: 3 }, ctx(null), {}) as Float32Array, [12, 13, 14]);
  });
  it('zero-pads past the end', () => {
    expectClose(Primitives.Slice([new Float32Array([1, 2])], { start: 1, len: 3 }, ctx(null), {}) as Float32Array, [2, 0, 0]);
  });
});

describe('motion-reference primitives', () => {
  it('TrackingRefField reads the field at refIdx+step, clamped to the clip', () => {
    const t = refTerm(2, 5);
    expectClose(Primitives.TrackingRefField([], { field: 'root_pos', step: 1 }, ctx(t), {}) as Float32Array, t.refRootPos[3]);
    expectClose(Primitives.TrackingRefField([], { field: 'joint_pos', step: -5 }, ctx(t), {}) as Float32Array, t.refJointPos[0]);
    expectClose(Primitives.TrackingRefField([], { field: 'root_pos', step: 99 }, ctx(t), {}) as Float32Array, t.refRootPos[4]);
  });
  it('TrackingRefField falls back to a finite value when not ready / absent', () => {
    const nr = ctx(refTerm(0, 4, false));
    expect(Array.from(Primitives.TrackingRefField([], { field: 'root_quat', step: 0 }, nr, {}) as Float32Array)).toEqual([1, 0, 0, 0]);
    expect((Primitives.TrackingRefField([], { field: 'root_pos', step: 0 }, nr, {}) as Float32Array).length).toBe(3);
    // joint width falls back to the policy action count when no clip is loaded.
    expect((Primitives.TrackingRefField([], { field: 'joint_pos', step: 0 }, ctx(null, 5), {}) as Float32Array).length).toBe(5);
  });
  it('TrackingIsReady reflects readiness / presence', () => {
    expect(Primitives.TrackingIsReady([], {}, ctx(refTerm(0, 4, true)), {})).toBe(1);
    expect(Primitives.TrackingIsReady([], {}, ctx(refTerm(0, 4, false)), {})).toBe(0);
    expect(Primitives.TrackingIsReady([], {}, ctx(null), {})).toBe(0);
  });
});
