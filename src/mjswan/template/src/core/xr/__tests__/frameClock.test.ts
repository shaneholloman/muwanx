/** The frame time every rate in an XR session is scaled by. */
import { describe, expect, it } from 'vitest';

import { FrameClock } from '../frameClock';

describe('FrameClock', () => {
  it('has no time on its first frame', () => {
    expect(new FrameClock().tick(1000)).toBe(0);
  });

  it('reports the gap between frames, in seconds', () => {
    const clock = new FrameClock();
    clock.tick(1000);
    expect(clock.tick(1016)).toBeCloseTo(0.016, 6);
    expect(clock.tick(1050)).toBeCloseTo(0.034, 6);
  });

  /** A tab away leaves a gap that would teleport whatever the frame time drives. */
  it('clamps a frame the browser slept through', () => {
    const clock = new FrameClock();
    clock.tick(0);
    expect(clock.tick(30_000)).toBeCloseTo(0.1, 6);
  });

  it('reads a clock that did not move forward as no time at all', () => {
    const clock = new FrameClock();
    clock.tick(1000);
    expect(clock.tick(1000)).toBe(0);
    expect(clock.tick(500)).toBe(0);
  });

  it('starts over after a reset, so a new session begins from a standstill', () => {
    const clock = new FrameClock();
    clock.tick(1000);
    clock.reset();
    expect(clock.tick(9000)).toBe(0);
  });
});
