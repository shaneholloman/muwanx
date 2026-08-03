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
import { CommandManager } from '../../command/CommandManager';
import type {
  CommandConfigEntry,
  CommandTerm,
  CommandTermConstructor,
  CommandTermContext,
  CommandsConfig,
} from '../../command/types';
import type { FusedObservationConfig } from '../../observation/FusedObservation';
import { createOnnxSession, type OnnxSession, type SlotReader } from '../../onnx/session';
import { createSlotReader, type SlotReaderContext } from '../../onnx/slotReader';
import { TerminationManager } from '../../termination/TerminationManager';
import { PolicyRunner } from '../../policy/PolicyRunner';
import type {
  PolicyConfig,
  PolicyRunnerContext,
  PolicyState,
  TerminationConfigEntry,
} from '../../policy/types';

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

/** Reads one native input's fixture value for the current step. */
function nativeReader(task: TaskFixture, step: () => Step, kind: string): () => Float32Array {
  const entry = (task.group.native_inputs ?? []).find(native => native.native === kind);
  return () => Float32Array.from(entry ? (step().native[entry.input] ?? []) : []);
}

/** A command term serving one fixture value, registered like any plugin term. */
function fixtureCommandClass(read: () => Float32Array): CommandTermConstructor {
  return class FixtureCommand implements CommandTerm {
    constructor(
      _termName: string,
      _config: CommandConfigEntry,
      _context: CommandTermContext,
    ) {}

    getCommand(): Float32Array {
      return read();
    }
  };
}

/**
 * The real `PolicyRunner` and `CommandManager`, wired as `runtime.ts` wires them.
 *
 * This used to be a hand-written stub answering `getLastActions`/`getCommand`
 * straight from the fixture — which left the two manager dependencies the
 * observation vector genuinely has (Action→Observation and Command→Observation) as
 * the only part of the chain this test did not run. mjlab's numbers still come from
 * the fixture, since nothing here recomputes a policy or a command term, but they now
 * travel the runtime's path to reach the graph: through `setLastActions` into the
 * runner's own buffer and back out of `getLastActions`, and through a term registered
 * under the name the build wrote into the config, found by `getCommand`'s lookup.
 *
 * The runner also builds the fused observation itself rather than the test
 * constructing one, so `buildObservationGroups` / `registerGroup` / `buildFrame` — the
 * layout and width bookkeeping around the graph — are under test too.
 */
async function harnessFor(
  taskId: string,
  task: TaskFixture,
  step: () => Step,
  readSlot: SlotReader,
): Promise<PolicyRunner> {
  const commandManager = new CommandManager();
  // Registered under the names the build emitted, through the real registry path, so
  // both `getCommand`'s lookup and `FusedObservation`'s construction-time name binding
  // run against a real manager instead of an object that answers everything.
  const commands: CommandsConfig = {};
  for (const native of task.group.native_inputs ?? []) {
    if (native.native === 'command' && native.command_name) {
      commands[native.command_name] = { name: 'FixtureCommand' };
    }
  }
  commandManager.initialize(commands, {} as unknown as CommandTermContext, {
    FixtureCommand: fixtureCommandClass(nativeReader(task, step, 'command')),
  });

  const sessions = new Map<string, OnnxSession>([
    [task.group.fused, await sessionFor(taskId, task.group.fused)],
  ]);
  const runner = new PolicyRunner(
    {
      policy_num_actions: task.num_actions,
      observations: { policy: task.group },
    } as unknown as PolicyConfig,
    {
      onnxSessions: { get: (path: string) => sessions.get(path) } as never,
      readOnnxSlot: readSlot,
    },
  );
  await runner.init({
    mujoco: null,
    mjModel: null,
    mjData: null,
    commandManager,
  } as unknown as PolicyRunnerContext);
  return runner;
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
  let runner: PolicyRunner;
  let terminations: TerminationManager;

  const readSlot: SlotReader = slot =>
    createSlotReader(() => contextFor(task, current), {
      jointBias: name => task.encoder_bias[name] ?? 0,
    })(slot);

  /** mjlab's `action_manager.action` for the current step, stored as the runtime does. */
  const readActions = nativeReader(task, () => current, 'prev_action');

  /** Advance to one fixture step, pushing its action through the real runner. */
  const seek = (step: Step): void => {
    current = step;
    runner.setLastActions(readActions());
  };

  beforeAll(async () => {
    runner = await harnessFor(taskId, task, () => current, readSlot);

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
      seek(step);
      // Through the runner, not a directly-built `FusedObservation`: the group's
      // layout, its declared width, and the two native reads all sit between the
      // graph and this vector.
      const vector = await runner.collectObservations(EMPTY_STATE);
      expectClose(vector, step.obs, `step ${index}`);
    }
    // A group whose graph silently returned zeros would pass a vector of zeros
    // against a fixture of zeros; assert the states actually vary.
    const first = task.steps[0].obs;
    const last = task.steps[task.steps.length - 1].obs;
    expect(first.some((v, i) => Math.abs(v - last[i]) > 1e-6)).toBe(true);
  }, 120_000);

  it('keeps the native inputs that make the manager wiring observable', () => {
    // A guard on the harness rather than on the runtime. The Action→Observation and
    // Command→Observation paths are only under test on a task whose group *has* those
    // native inputs — Velocity-Flat-G1 does, Cartpole's group has none — so a
    // regenerated fixture that lost them, or a `command` input that lost its
    // `command_name`, would silently stop exercising them while every other assertion
    // here still passed.
    const natives = task.group.native_inputs ?? [];
    const command = natives.find(native => native.native === 'command');
    if (command) {
      expect(command.command_name, 'a command input must name its term').toBeTruthy();
      // Found by name in the real manager, which is what `getCommand` resolves.
      expect(runner.getContext()?.commandManager?.termNames()).toContain(command.command_name);
    }
    if (natives.some(native => native.native === 'prev_action')) {
      expect(runner.getNumActions()).toBe(task.num_actions);
      expect(readActions().length).toBe(task.num_actions);
    }
    // Cartpole legitimately has neither; assert that is still why, so this reads as a
    // fixture property rather than a skipped check.
    if (natives.length === 0) expect(taskId).toBe('Mjlab-Cartpole-Balance');
  });

  it('reproduces every termination verdict at every step', async () => {
    for (const [index, step] of task.steps.entries()) {
      seek(step);
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
