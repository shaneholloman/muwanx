/**
 * Termination terms under ADR 0005: traced-ONNX bodies, the native `time_out`,
 * and the manager's OR-reduce with truncation split out.
 *
 * The graph's math is validated Python-side by the parity harness; what matters
 * here is the native half — bool decoding, the skip-if-in-flight async boundary,
 * holding the previous verdict rather than reporting "not done" on missing state,
 * and terminated-vs-truncated bookkeeping.
 */
import { describe, expect, it, vi } from 'vitest';

import { OnnxTermination, type OnnxTerminationConfig } from '../OnnxTermination';
import { TerminationManager } from '../TerminationManager';
import { TimeOutTermination } from '../TimeOutTermination';
import type { OnnxSession, OnnxTensorLike } from '../../onnx/session';
import type { PolicyRunner } from '../../policy/PolicyRunner';

async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

/** Returns a scripted bool output; `manual` defers resolution. */
class FakeSession implements OnnxSession {
  readonly calls: Array<Record<string, OnnxTensorLike>> = [];
  private pending: Array<() => void> = [];

  constructor(
    private readonly respond: (call: number) => Uint8Array,
    private readonly manual = false,
  ) {}

  run(feeds: Record<string, OnnxTensorLike>): Promise<Record<string, OnnxTensorLike>> {
    const index = this.calls.length;
    this.calls.push(feeds);
    const result = { done: { data: this.respond(index), dims: [1] } };
    if (!this.manual) return Promise.resolve(result);
    return new Promise(resolve => this.pending.push(() => resolve(result)));
  }

  flush(): void {
    this.pending.shift()?.();
  }

  get inFlightCount(): number {
    return this.pending.length;
  }
}

const runner = {} as unknown as PolicyRunner;

const FELL_OVER_CFG: OnnxTerminationConfig = {
  name: 'fell_over',
  onnx: 'term/fell_over.onnx',
  input_slots: [
    { entity: 'robot', field: 'projected_gravity_b', input: 'robot__projected_gravity_b' },
  ],
};

describe('OnnxTermination', () => {
  it('decodes a bool output and feeds the declared slot', async () => {
    const session = new FakeSession(() => Uint8Array.from([1]));
    const term = new OnnxTermination(runner, FELL_OVER_CFG, {
      session,
      readSlot: () => new Float32Array([0, 0, 1]),
    });
    await term.step();
    expect(Object.keys(session.calls[0])).toEqual(['robot__projected_gravity_b']);
    expect(term.evaluate({} as never)).toBe(true);
  });

  it('reports false while the graph says not done', async () => {
    const session = new FakeSession(() => Uint8Array.from([0]));
    const term = new OnnxTermination(runner, FELL_OVER_CFG, {
      session,
      readSlot: () => new Float32Array([0, 0, -1]),
    });
    await term.step();
    expect(term.evaluate({} as never)).toBe(false);
  });

  it('does not block, and skips frames while inference is in flight', async () => {
    const session = new FakeSession(() => Uint8Array.from([1]), true);
    const term = new OnnxTermination(runner, FELL_OVER_CFG, {
      session,
      readSlot: () => new Float32Array([0, 0, 1]),
    });
    // First evaluate kicks inference off and returns the (not yet updated) verdict.
    expect(term.evaluate({} as never)).toBe(false);
    for (let i = 0; i < 10; i++) term.evaluate({} as never);
    expect(session.calls.length).toBe(1); // 9 frames skipped, not queued
    expect(session.inFlightCount).toBe(1);

    session.flush();
    await settle();
    expect(term.evaluate({} as never)).toBe(true);
  });

  it('holds the previous verdict when a slot is unreadable', async () => {
    const session = new FakeSession(() => Uint8Array.from([1]));
    let available = true;
    const term = new OnnxTermination(runner, FELL_OVER_CFG, {
      session,
      readSlot: () => (available ? new Float32Array([0, 0, 1]) : null),
    });
    await term.step();
    const callsWhileAvailable = session.calls.length;

    available = false;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await term.step();
    // The graph is not run at all on missing state...
    expect(session.calls.length).toBe(callsWhileAvailable);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
    // ...and the last verdict stands. Reporting "not done" instead would let a
    // real termination slip through.
    expect(term.evaluate({} as never)).toBe(true);
  });

  it('reset() clears the verdict', async () => {
    const session = new FakeSession(() => Uint8Array.from([1]));
    const term = new OnnxTermination(runner, FELL_OVER_CFG, {
      session,
      readSlot: () => new Float32Array([0, 0, 1]),
    });
    await term.step();
    expect(term.evaluate({} as never)).toBe(true);
    term.reset();
    expect(term.evaluate({} as never)).toBe(false);
  });
});

describe('TimeOutTermination', () => {
  it('fires once elapsed reaches the episode length', () => {
    let elapsed = 0;
    const term = new TimeOutTermination(
      runner,
      { name: 'time_out', episode_length_s: 2.0 },
      () => elapsed,
    );
    expect(term.evaluate({} as never)).toBe(false);
    elapsed = 1.9;
    expect(term.evaluate({} as never)).toBe(false);
    elapsed = 2.0;
    expect(term.evaluate({} as never)).toBe(true);
  });

  it('never fires without a finite episode length', () => {
    // mjlab's play configs set an effectively infinite horizon; a missing or
    // non-positive value must not mean "time out every frame".
    for (const episode_length_s of [undefined, 0]) {
      const term = new TimeOutTermination(
        runner,
        { name: 'time_out', episode_length_s },
        () => 1e9,
      );
      expect(term.evaluate({} as never)).toBe(false);
    }
  });
});

describe('TerminationManager', () => {
  it('accumulates dt for the native time_out and reports truncation', () => {
    const manager = new TerminationManager(
      {
        time_out: {
          name: 'time_out',
          native: 'elapsed_s >= episode_length_s',
          episode_length_s: 0.5,
          time_out: true,
        } as never,
      },
      {},
      runner,
    );
    expect(manager.evaluate({} as never, 0.2).done).toBe(false);
    const result = manager.evaluate({} as never, 0.4); // elapsed 0.6 >= 0.5
    expect(result.done).toBe(true);
    // A timeout is a truncation, not a failure.
    expect(result.truncated).toBe(true);
    expect(result.terminated).toBe(false);
    expect(result.reasons).toEqual(['time_out']);
  });

  it('reset() clears elapsed time', () => {
    const manager = new TerminationManager(
      {
        time_out: {
          name: 'time_out',
          native: 'elapsed_s >= episode_length_s',
          episode_length_s: 0.5,
          time_out: true,
        } as never,
      },
      {},
      runner,
    );
    manager.evaluate({} as never, 1.0);
    manager.reset();
    expect(manager.evaluate({} as never, 0.1).done).toBe(false);
  });

  it('builds an ONNX term and reports it as a termination, not a truncation', async () => {
    const session = new FakeSession(() => Uint8Array.from([1]));
    const manager = new TerminationManager(
      { fell_over: { ...FELL_OVER_CFG } as never },
      {},
      runner,
      {
        onnxSessions: { get: () => session } as never,
        readOnnxSlot: () => new Float32Array([0, 0, 1]),
      },
    );
    manager.evaluate({} as never, 0.02); // kicks inference off
    await settle();
    const result = manager.evaluate({} as never, 0.02);
    expect(result.done).toBe(true);
    expect(result.terminated).toBe(true);
    expect(result.truncated).toBe(false);
    expect(result.reasons).toEqual(['fell_over']);
  });

  it('warns and skips an ONNX term whose session is missing', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const manager = new TerminationManager(
      { fell_over: { ...FELL_OVER_CFG } as never },
      {},
      runner,
    );
    // Losing one reset condition beats taking down the whole scene.
    expect(manager.size).toBe(0);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});
