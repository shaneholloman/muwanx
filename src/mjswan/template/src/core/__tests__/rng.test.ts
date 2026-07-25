/**
 * Orchestrator-owned seeded PRNG (ADR 0005 §2).
 *
 * The point of these tests is the *replay guarantee*: a recorded session must
 * reproduce bit-for-bit, so the sequence has to be a pure function of the seed
 * (and resumable from a state snapshot).
 */
import { describe, expect, it } from 'vitest';

import { SeededRng } from '../rng';

describe('SeededRng', () => {
  it('is deterministic for a given seed', () => {
    const a = new SeededRng(12345);
    const b = new SeededRng(12345);
    const drawsA = Array.from({ length: 64 }, () => a.next());
    const drawsB = Array.from({ length: 64 }, () => b.next());
    expect(drawsA).toEqual(drawsB);
  });

  it('produces different streams for different seeds', () => {
    const a = new SeededRng(1);
    const b = new SeededRng(2);
    const drawsA = Array.from({ length: 16 }, () => a.next());
    const drawsB = Array.from({ length: 16 }, () => b.next());
    expect(drawsA).not.toEqual(drawsB);
  });

  it('stays within [0, 1)', () => {
    const rng = new SeededRng(7);
    for (let i = 0; i < 2000; i++) {
      const v = rng.next();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('uniform() respects its range', () => {
    const rng = new SeededRng(99);
    for (let i = 0; i < 1000; i++) {
      const v = rng.uniform(-0.5, 2.5);
      expect(v).toBeGreaterThanOrEqual(-0.5);
      expect(v).toBeLessThan(2.5);
    }
  });

  it('resumes exactly from a state snapshot (session replay)', () => {
    const rng = new SeededRng(4242);
    for (let i = 0; i < 10; i++) rng.next();
    const state = rng.getState();
    const expected = Array.from({ length: 8 }, () => rng.next());

    const resumed = new SeededRng(0);
    resumed.setState(state);
    const actual = Array.from({ length: 8 }, () => resumed.next());
    expect(actual).toEqual(expected);
  });

  it('randVector fills an ONNX rand input, scaling per-element ranges', () => {
    const rng = new SeededRng(5);
    const ranges: Array<readonly [number, number]> = [
      [-0.5, 0.5],
      [-0.5, 0.5],
      [-0.4, 0.4],
      [-0.52, 0.52],
      [-0.52, 0.52],
      [-0.78, 0.78],
    ];
    const vec = rng.randVector(6, ranges);
    expect(vec).toBeInstanceOf(Float32Array);
    expect(vec.length).toBe(6);
    for (let i = 0; i < 6; i++) {
      expect(vec[i]).toBeGreaterThanOrEqual(ranges[i][0]);
      expect(vec[i]).toBeLessThan(ranges[i][1]);
    }
  });

  it('randVector without ranges yields plain [0,1) draws', () => {
    const vec = new SeededRng(11).randVector(4);
    for (const v of vec) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('never returns the same value twice in a row over a long run', () => {
    const rng = new SeededRng(2026);
    let previous = rng.next();
    let repeats = 0;
    for (let i = 0; i < 5000; i++) {
      const v = rng.next();
      if (v === previous) repeats++;
      previous = v;
    }
    expect(repeats).toBe(0);
  });
});
