/**
 * Rollout parity: the whole browser MDP chain against mjlab, over N steps.
 *
 * ADR 0005's first acceptance criterion. Everything else in the verification
 * chain covers a *piece* — the Python harness proves each traced graph reproduces
 * its mjlab term, `slotReaderParity` proves the reader hands those graphs mjlab's
 * numbers, the manager suites prove the pipeline arithmetic. This is the only test
 * that runs them composed: fixture state → real `SlotReader` → real ORT session
 * over the graph bytes the Builder wrote → real `FusedObservation` → group vector,
 * and the same through `TerminationManager` to a verdict.
 *
 * What that composition can get wrong, and nothing narrower would catch: a layout
 * offset (right numbers, wrong order), a slot fed under a name the graph does not
 * declare, a lane read from the wrong column, `time_out` counted as a termination
 * rather than a truncation.
 *
 * **States are replayed, not co-simulated.** mjlab integrates with `mujoco_warp`
 * and the browser runs MuJoCo's own WASM build; two integrators do not agree
 * step-for-step, so a free-running comparison would measure MuJoCo against itself.
 * The fixture therefore carries mjlab's state at each step alongside mjlab's own
 * observation vector and termination verdicts *at that state*. Physics is MuJoCo's
 * code on both sides and is out of scope here.
 *
 * Regenerate: `MUJOCO_GL=disable .venv/bin/python scripts/dump_rollout_fixture.py`
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { beforeAll, describe, expect, it } from 'vitest';

import fixture from './fixtures/rollout/rollout.json';
import { FusedObservation, type FusedObservationConfig } from '../../observation/FusedObservation';
import { createOnnxSession, type OnnxSession, type SlotReader } from '../../onnx/session';
import { createSlotReader, type SlotReaderContext } from '../../onnx/slotReader';
import { TerminationManager } from '../../termination/TerminationManager';
import type { PolicyRunner } from '../../policy/PolicyRunner';
import type { PolicyState, TerminationConfigEntry } from '../../policy/types';

const FIXTURES = join(__dirname, 'fixtures/rollout');

interface Step {
  action: number[];
  data: Record<string, number[]>;
  native: Record<string, number[]>;
  obs: number[];
  terminations: Record<string, boolean>;
}

interface TaskFixture {
  group: FusedObservationConfig & { fused: string };
  terminations: Record<string, TerminationConfigEntry>;
  model: Record<string, number | number[]>;
  encoder_bias: Record<string, number>;
  num_actions: number;
  steps: Step[];
}

const TASKS = fixture as unknown as Record<string, TaskFixture>;

/** Load a graph the fixture dumper wrote, as the bytes a bundle would deliver. */
async function sessionFor(taskId: string, ref: string): Promise<OnnxSession> {
  const buf = readFileSync(join(FIXTURES, taskId, ref));
  return createOnnxSession(
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer,
  );
}

/**
 * The `mjModel`/`mjData` view the reader indexes, for one step.
 *
 * mjlab's arrays rather than a WASM model's: what is under test is the chain from
 * a state to an observation, and holding the state fixed is what makes mjlab's
 * vector the right answer to compare against.
 */
function contextFor(task: TaskFixture, step: Step): SlotReaderContext {
  const { names, ...rest } = task.model;
  const mjModel = { ...rest, names: Uint8Array.from(names as number[]).buffer };
  const mjData: Record<string, Float64Array> = {};
  for (const [key, values] of Object.entries(step.data)) {
    mjData[key] = Float64Array.from(values);
  }
  return { mjModel, mjData } as unknown as SlotReaderContext;
}

/**
 * Stands in for the orchestrator's env-level state: the policy's own last output
 * and a live command term. Both are native by design (no graph computes them), so
 * the fixture supplies mjlab's values and the comparison stays about the graph and
 * the pipeline. Their own computation is covered by the `PolicyRunner` and
 * `OnnxCommand` suites.
 */
function runnerFor(task: TaskFixture, step: () => Step): PolicyRunner {
  const nativeByKind = new Map<string, string>();
  for (const native of task.group.native_inputs ?? []) {
    nativeByKind.set(native.native, native.input);
  }
  const read = (kind: string): Float32Array => {
    const key = nativeByKind.get(kind);
    return Float32Array.from(key ? (step().native[key] ?? []) : []);
  };
  // The names the fixture's own group declares. `FusedObservation` binds each
  // `command` input against this set at construction, so deriving it from the
  // fixture (rather than answering every name) keeps the stub honest: a build that
  // emitted a command name the scene never defines would fail here too.
  const commandNames = (task.group.native_inputs ?? [])
    .filter(native => native.native === 'command')
    .map(native => native.command_name)
    .filter((name): name is string => Boolean(name));
  return {
    getLastActions: () => read('prev_action'),
    getContext: () => ({
      commandManager: {
        getCommand: () => read('command'),
        termNames: () => commandNames,
      },
    }),
  } as unknown as PolicyRunner;
}

/** float32 through the graph, so compare at float32 resolution. */
function expectClose(actual: ArrayLike<number>, expected: number[], label: string): void {
  expect(actual.length, `${label}: width`).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const tolerance = Math.max(1e-4, Math.abs(expected[i]) * 1e-4);
    expect(Math.abs(actual[i] - expected[i]), `${label}[${i}]`).toBeLessThan(tolerance);
  }
}

const EMPTY_STATE = {} as PolicyState;

describe.each(Object.keys(TASKS))('rollout parity vs mjlab — %s', taskId => {
  const task = TASKS[taskId];
  let current: Step = task.steps[0];
  let observation: FusedObservation;
  let terminations: TerminationManager;

  const readSlot: SlotReader = slot =>
    createSlotReader(() => contextFor(task, current), {
      jointBias: name => task.encoder_bias[name] ?? 0,
    })(slot);

  beforeAll(async () => {
    const runner = runnerFor(task, () => current);
    observation = new FusedObservation(runner, task.group, {
      session: await sessionFor(taskId, task.group.fused),
      readSlot,
    });

    // Every traced termination's graph, keyed the way the config references it —
    // so the manager resolves them exactly as the runtime does.
    const sessions = new Map<string, OnnxSession>();
    for (const entry of Object.values(task.terminations)) {
      for (const ref of [
        (entry as { onnx?: string }).onnx,
        (entry as { fused?: string }).fused,
      ]) {
        if (ref) sessions.set(ref, await sessionFor(taskId, ref));
      }
    }
    terminations = new TerminationManager(task.terminations, {}, runner, {
      onnxSessions: { get: (path: string) => sessions.get(path) } as never,
      readOnnxSlot: readSlot,
    });
  }, 120_000);

  it('reproduces mjlab’s observation vector at every step', async () => {
    for (const [index, step] of task.steps.entries()) {
      current = step;
      const vector = await observation.compute(EMPTY_STATE);
      expectClose(vector, step.obs, `step ${index}`);
    }
    // A group whose graph silently returned zeros would pass a vector of zeros
    // against a fixture of zeros; assert the states actually vary.
    const first = task.steps[0].obs;
    const last = task.steps[task.steps.length - 1].obs;
    expect(first.some((v, i) => Math.abs(v - last[i]) > 1e-6)).toBe(true);
  }, 120_000);

  it('reproduces every termination verdict at every step', async () => {
    for (const [index, step] of task.steps.entries()) {
      current = step;
      // `evaluate()` is synchronous and ORT is not, so a call kicks the graph and
      // reports the *previous* verdict — the runtime accepts that one-frame lag
      // (ADR 0005 §8). Reading this state's verdict therefore takes kick → drain →
      // read, and a trailing drain: the read is itself a kick, and leaving it in
      // flight would make the next iteration report this state's verdict instead
      // of its own, shifting the whole comparison by a step.
      terminations.evaluate(EMPTY_STATE, 0);
      await drainPending();
      const result = terminations.evaluate(EMPTY_STATE, 0);
      await drainPending();

      const expected = Object.entries(step.terminations)
        .filter(([, fired]) => fired)
        .map(([name]) => name);
      expect(result.reasons.slice().sort(), `step ${index} reasons`).toEqual(expected.sort());
      expect(result.done, `step ${index} done`).toBe(expected.length > 0);
    }
  }, 120_000);

  it('keeps a rollout that could tell a right verdict from a constant one', () => {
    // A verdict comparison over states where nothing ever fires proves only that
    // the graph raises no false positives — one hardwired to `false` would pass it.
    // The dumper tilts the root through an orientation limit for exactly this
    // reason; if a regenerated fixture loses those states, fail here rather than
    // quietly weakening the test above.
    const traced = Object.entries(task.terminations).filter(
      ([, entry]) => 'onnx' in entry || 'fused' in entry,
    );
    if (traced.length === 0) return; // Cartpole: its only termination is native.
    for (const [name] of traced) {
      const verdicts = task.steps.map(step => step.terminations[name]);
      expect(verdicts, `${name} never fires`).toContain(true);
      expect(verdicts, `${name} never clears`).toContain(false);
    }
  });

  it('covers the whole group, not a prefix of it', () => {
    // The layout is what splices term outputs into the vector; a truncated one
    // would still let the assertions above pass on the terms that survived.
    const layout = task.group.layout ?? [];
    const width = layout.reduce((n, term) => n + term.size, 0);
    expect(width).toBe(task.group.size);
    expect(task.steps[0].obs.length).toBe(task.group.size);
    expect(layout.length).toBeGreaterThan(0);
  });
});

/** Let every already-scheduled promise settle (the ORT run is one). */
async function drainPending(): Promise<void> {
  for (let i = 0; i < 8; i++) await new Promise(resolve => setTimeout(resolve, 0));
}
