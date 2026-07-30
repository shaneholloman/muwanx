/**
 * `motion` stays a native command (a clip lookup is data, not math), so the
 * tracking task's traced observations and terminations read their
 * `{command: "motion", field}` slots off `TrackingCommand` itself. This pins the
 * one part of that which can be silently wrong: the re-anchoring frame, mjlab's
 * `update_relative_body_poses`.
 */
import { describe, expect, it } from 'vitest';

import { reanchorBodyPositions } from '../TrackingCommand';

function close(actual: Float32Array, expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) expect(actual[i]).toBeCloseTo(expected[i], 6);
}

describe('reanchorBodyPositions', () => {
  it('takes x/y from the robot anchor, z from the reference, and rotates by the yaw between them', () => {
    const s = Math.SQRT1_2; // 90 deg about +z
    const relative = reanchorBodyPositions(
      // Two reference bodies: one 1 m ahead of the anchor, one 1 m above it.
      Float32Array.from([1, 0, 1, 0, 0, 2]),
      [0, 0, 1],
      [1, 0, 0, 0],
      [5, 7, 2],
      [s, 0, 0, s],
    );
    // Anchor lands at (5, 7, 1) — the robot's x/y, the reference's z, so a reader
    // that took the robot's z would put the whole skeleton 1 m too high.
    // The +x offset rotates onto +y; the +z offset is untouched by yaw.
    close(relative, [5, 8, 1, 5, 7, 2]);
  });

  it('is the identity when the robot sits exactly on the reference anchor', () => {
    const bodies = Float32Array.from([1, 2, 3, -1, 0, 0.5]);
    close(reanchorBodyPositions(bodies, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [1, 0, 0, 0]), [
      1, 2, 3, -1, 0, 0.5,
    ]);
  });
});
