/**
 * `CommandManager`: `OnnxCommand` registration (ADR 0005 §3).
 *
 * `OnnxCommand` bypasses the class registry the same way `OnnxEvent` bypasses
 * `EventManager`'s — it needs a session + rng the registry's
 * `new Term(name, config, context)` shape has no room for. These tests pin
 * that bypass and its failure mode (missing session/rng warns and skips
 * rather than throwing and taking down every other command).
 */
import { describe, expect, it, vi } from 'vitest';

import { SeededRng } from '../../rng';
import { OnnxSessionCache, type OnnxSession, type OnnxTensorLike } from '../../onnx/session';
import { CommandManager } from '../CommandManager';
import type { CommandTermContext, CommandsConfig } from '../types';
import type { OnnxCommandConfig } from '../OnnxCommand';

/** Tensor data as a plain number array (the data union needs one narrowing). */
function values(tensor: OnnxTensorLike): number[] {
  return Array.from(tensor.data as ArrayLike<number>, Number);
}

function fakeSession(run: (feeds: Record<string, OnnxTensorLike>) => Record<string, OnnxTensorLike>): OnnxSession {
  return { run: (feeds) => Promise.resolve(run(feeds)) };
}

async function contextWithSession(path: string, session: OnnxSession): Promise<CommandTermContext> {
  const sessions = new OnnxSessionCache(() => Promise.resolve(session));
  await sessions.load([{ name: path, data: new ArrayBuffer(0) }]);
  return {
    mujoco: {} as unknown as CommandTermContext['mujoco'],
    mjModel: null,
    mjData: null,
    scene: {} as unknown as CommandTermContext['scene'],
    rng: new SeededRng(1),
    onnxSessions: sessions,
  };
}

const VELOCITY_CFG: OnnxCommandConfig = {
  name: 'OnnxCommand',
  onnx: 'command/twist.onnx',
  command_field: 'vel_command_b',
  rand_dim: 6,
  state_fields: [{ name: 'vel_command_b', shape: [1, 3], dtype: 'float32' }],
};

describe('CommandManager: resetTerms', () => {
  /** A term that records when its reset starts and finishes. */
  function recordingTerm(name: string, order: string[], delayed: boolean) {
    return {
      getCommand: () => new Float32Array(0),
      reset: async () => {
        order.push(`${name}:start`);
        if (delayed) await new Promise(resolve => setTimeout(resolve, 0));
        order.push(`${name}:done`);
      },
    };
  }

  it('leaves the UI values alone when a reset redraws the command', async () => {
    // The panel used to mirror `getCommand()` by position, so the drawn forward velocity
    // landed in the `enabled` checkbox and flipped the Joystick toggle on every reset.
    const context = await contextWithSession(
      'command/twist.onnx',
      fakeSession(() => ({ next_vel_command_b: { data: new Float32Array([0.9, 0, 0]), dims: [1, 3] } })),
    );
    const mgr = new CommandManager();
    mgr.initialize(
      {
        twist: {
          ...VELOCITY_CFG,
          ui: {
            inputs: [
              { type: 'checkbox', name: 'enabled', label: 'Joystick', default: false },
              { type: 'slider', name: 'lin_vel_x', label: 'Forward', min: -1, max: 1, step: 0.05, default: 0.5 },
            ],
          },
        } as OnnxCommandConfig,
      },
      context,
    );

    await mgr.resetTerms();

    expect(mgr.getValues()['twist:enabled']).toBe(0);
    expect(mgr.getValues()['twist:lin_vel_x']).toBe(0.5);
  });

  it('awaits each term in config order, not concurrently', () => {
    // Each reset *is* the term's resample and may write to the sim, so overlaps resolve
    // last-writer-wins by config order. `Promise.all` would interleave the four events.
    const order: string[] = [];
    const manager = new CommandManager();
    const terms = (manager as unknown as { terms: Map<string, unknown> }).terms;
    terms.set('slow', recordingTerm('slow', order, true));
    terms.set('fast', recordingTerm('fast', order, false));

    return manager.resetTerms().then(() => {
      expect(order).toEqual(['slow:start', 'slow:done', 'fast:start', 'fast:done']);
    });
  });
});

describe('CommandManager: OnnxCommand registration', () => {
  it('bypasses the class registry and constructs an OnnxCommand', async () => {
    const context = await contextWithSession(
      'command/twist.onnx',
      fakeSession(() => ({ next_vel_command_b: { data: new Float32Array([1, 2, 3]), dims: [1, 3] } })),
    );
    const mgr = new CommandManager();
    const commands: CommandsConfig = { twist: VELOCITY_CFG };
    mgr.initialize(commands, context);

    const term = mgr.getTerm('twist');
    expect(term).toBeDefined();
    await (term as unknown as { step(resample: boolean): Promise<void> }).step(true);
    expect(Array.from(mgr.getCommand('twist'))).toEqual([1, 2, 3]);
  });

  it('does not throw when a registry name is unrelated to OnnxCommand', () => {
    const mgr = new CommandManager();
    const context: CommandTermContext = {
      mujoco: {} as unknown as CommandTermContext['mujoco'],
      mjModel: null,
      mjData: null,
      scene: {} as unknown as CommandTermContext['scene'],
    };
    expect(() => mgr.initialize({ ui: { name: 'UiCommand' } }, context)).not.toThrow();
    expect(mgr.getTerm('ui')).toBeDefined();
  });

  it('warns and skips when no onnxSessions/rng are supplied (does not throw)', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const mgr = new CommandManager();
    const context: CommandTermContext = {
      mujoco: {} as unknown as CommandTermContext['mujoco'],
      mjModel: null,
      mjData: null,
      scene: {} as unknown as CommandTermContext['scene'],
    };
    expect(() => mgr.initialize({ twist: VELOCITY_CFG }, context)).not.toThrow();
    expect(mgr.getTerm('twist')).toBeUndefined();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('warns and skips when the named onnx session was never loaded', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const context = await contextWithSession('command/other.onnx', fakeSession(() => ({})));
    const mgr = new CommandManager();
    expect(() => mgr.initialize({ twist: VELOCITY_CFG }, context)).not.toThrow();
    expect(mgr.getTerm('twist')).toBeUndefined();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('threads context.readOnnxSlot into the constructed OnnxCommand', async () => {
    const runFn = vi.fn((feeds: Record<string, OnnxTensorLike>) => {
      expect(values(feeds.robot__heading_w)).toEqual([1.5]);
      return { next_vel_command_b: { data: new Float32Array(3), dims: [1, 3] } };
    });
    const context = await contextWithSession('command/twist.onnx', fakeSession(runFn));
    context.readOnnxSlot = (slot) => (slot.field === 'heading_w' ? new Float32Array([1.5]) : null);
    const mgr = new CommandManager();
    mgr.initialize(
      { twist: { ...VELOCITY_CFG, input_slots: [{ entity: 'robot', field: 'heading_w' }] } },
      context,
    );
    const term = mgr.getTerm('twist');
    await (term as unknown as { step(resample: boolean): Promise<void> }).step(true);
    expect(runFn).toHaveBeenCalledTimes(1);
  });
});

/**
 * A `UiCommand`'s value as a traced graph's slot. A term that merely forwards a
 * command is native, but one that does arithmetic on it is a traced body whose
 * `{command, field: 'command'}` slot has to resolve against the UI command itself —
 * this command has no Python side to read it from.
 */
describe('CommandManager: UiCommand state field', () => {
  const context = {
    mujoco: {} as unknown as CommandTermContext['mujoco'],
    mjModel: null,
    mjData: null,
    scene: {} as unknown as CommandTermContext['scene'],
  };
  const config: CommandsConfig = {
    compliance: {
      name: 'UiCommand',
      ui: {
        inputs: [
          { type: 'checkbox', name: 'enabled', label: 'Enabled', default: true },
          { type: 'slider', name: 'force', label: 'Force', min: 10, max: 20, default: 12, step: 0.5 },
        ],
      },
    },
  } as unknown as CommandsConfig;

  it('serves the UI values under the `command` field, and nothing else', () => {
    const mgr = new CommandManager();
    mgr.initialize(config, context as CommandTermContext);
    const term = mgr.getTerm('compliance') as unknown as {
      getStateField(field: string): Float32Array | null;
    };
    expect(Array.from(term.getStateField('command')!)).toEqual([1, 12]);
    expect(term.getStateField('vel_command_b')).toBeNull();
  });

  it('tracks a slider change, so the graph sees the live value', () => {
    const mgr = new CommandManager();
    mgr.initialize(config, context as CommandTermContext);
    mgr.setValue('compliance:force', 18);
    const term = mgr.getTerm('compliance') as unknown as {
      getStateField(field: string): Float32Array | null;
    };
    expect(Array.from(term.getStateField('command')!)).toEqual([1, 18]);
  });
});
