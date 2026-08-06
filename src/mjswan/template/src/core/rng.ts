/**
 * The one PRNG behind every `rand` input the runtime feeds a term graph — never
 * `Math.random()`, and never ONNX's own `RandomUniform`/`RandomNormal`, whose PRNG the
 * spec leaves open. With the seed owned here, a recorded session replays bit-for-bit.
 *
 * xoshiro128** seeded by SplitMix32, spelled out rather than imported: a library's
 * internal change must never alter a recorded replay.
 */

export class SeededRng {
  private s0 = 0;
  private s1 = 0;
  private s2 = 0;
  private s3 = 0;
  private warnedRangeless = false;

  constructor(readonly seed: number) {
    // SplitMix32 expansion: derive four non-zero state words from one seed.
    let x = seed >>> 0;
    const next = (): number => {
      x = (x + 0x9e3779b9) >>> 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
      z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
      return (z ^ (z >>> 15)) >>> 0;
    };
    this.s0 = next();
    this.s1 = next();
    this.s2 = next();
    this.s3 = next();
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) this.s0 = 1;
  }

  /** Next raw 32-bit unsigned integer (xoshiro128**). */
  nextUint32(): number {
    const result = (Math.imul(rotl(Math.imul(this.s1, 5) >>> 0, 7), 9) >>> 0) >>> 0;
    const t = (this.s1 << 9) >>> 0;
    this.s2 = (this.s2 ^ this.s0) >>> 0;
    this.s3 = (this.s3 ^ this.s1) >>> 0;
    this.s1 = (this.s1 ^ this.s2) >>> 0;
    this.s0 = (this.s0 ^ this.s3) >>> 0;
    this.s2 = (this.s2 ^ t) >>> 0;
    this.s3 = rotl(this.s3, 11);
    return result;
  }

  /** Uniform in [0, 1). */
  next(): number {
    return this.nextUint32() / 4294967296;
  }

  /** Uniform in [low, high). */
  uniform(low: number, high: number): number {
    return low + this.next() * (high - low);
  }

  /**
   * Fill an ONNX `rand` input: `n` draws in the order the build recorded them, each scaled
   * by its `rand_ranges` entry. An element with no range warns and falls back to [0, 1).
   */
  randVector(n: number, ranges?: ReadonlyArray<readonly [number, number]>): Float32Array {
    if (n > 0 && (ranges?.length ?? 0) < n && !this.warnedRangeless) {
      this.warnedRangeless = true;
      console.warn(
        `[SeededRng] a term asked for ${n} draws but declared ${ranges?.length ?? 0} ` +
          'rand_ranges; the rest fall back to [0, 1). Rebuild the bundle.',
      );
    }
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const range = ranges?.[i];
      out[i] = range ? this.uniform(range[0], range[1]) : this.next();
    }
    return out;
  }

  /** Snapshot the internal state so a session can be resumed/replayed exactly. */
  getState(): [number, number, number, number] {
    return [this.s0, this.s1, this.s2, this.s3];
  }

  setState(state: readonly [number, number, number, number]): void {
    this.s0 = state[0] >>> 0;
    this.s1 = state[1] >>> 0;
    this.s2 = state[2] >>> 0;
    this.s3 = state[3] >>> 0;
  }
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}
