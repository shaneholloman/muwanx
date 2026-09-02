/**
 * Seconds between rendered XR frames, for anything that moves the rig at a rate.
 *
 * Clamped, because a tab away leaves a huge gap and an unclamped first frame back would
 * teleport whatever it drives. Zero on the first frame, so a rate has something to
 * measure against before it is applied.
 */

const MAX_FRAME_SECONDS = 0.1;

export class FrameClock {
  private last: number | null = null;

  /** Zero on the first frame, and on a clock that did not move forward. */
  tick(now: number): number {
    const previous = this.last;
    this.last = now;
    if (previous === null) {
      return 0;
    }
    return Math.min(Math.max(now - previous, 0) / 1000, MAX_FRAME_SECONDS);
  }

  reset(): void {
    this.last = null;
  }
}
