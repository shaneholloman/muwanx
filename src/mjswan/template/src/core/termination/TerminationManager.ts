import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { TerminationConstructor } from './terminations';
import { OnnxTermination, type OnnxTerminationConfig } from './OnnxTermination';
import { TimeOutTermination, type TimeOutTerminationConfig } from './TimeOutTermination';
import type { OnnxSessionCache, SlotReader } from '../onnx/session';
import type { PolicyState, TerminationConfigEntry } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';

export type TerminationResult = {
  done: boolean;
  terminated: boolean;
  truncated: boolean;
  reasons: string[];
};

/**
 * Deps for ONNX-backed termination terms (ADR 0005), mirroring the observation
 * ones. Absent for a policy whose terminations are all native or legacy.
 */
export type TerminationManagerDeps = {
  onnxSessions?: OnnxSessionCache;
  readOnnxSlot?: SlotReader;
};

/** Whether an entry names a traced-ONNX termination (ADR 0005). */
function isOnnxEntry(entry: TerminationConfigEntry): entry is OnnxTerminationConfig {
  return typeof (entry as { onnx?: unknown }).onnx === 'string';
}

/**
 * Whether an entry is the native `time_out` marker. Matched on `native` being
 * present rather than on its exact text: the string is a human-readable
 * description of the comparison, not a wire enum.
 */
function isNativeTimeOutEntry(
  entry: TerminationConfigEntry,
): entry is TimeOutTerminationConfig {
  return typeof (entry as { native?: unknown }).native === 'string';
}

export class TerminationManager {
  private terms: { name: string; term: TerminationBase; isTimeOut: boolean }[] = [];
  /** Episode time, accumulated from the control `dt` the caller passes. */
  private elapsedS = 0;

  constructor(
    config: Record<string, TerminationConfigEntry>,
    registry: Record<string, TerminationConstructor>,
    runner: PolicyRunner,
    deps: TerminationManagerDeps = {},
  ) {
    for (const [name, entry] of Object.entries(config)) {
      if (isOnnxEntry(entry)) {
        const term = this.buildOnnxTermination(name, entry, runner, deps);
        if (term) {
          this.terms.push({ name, term, isTimeOut: entry.time_out ?? false });
        }
        continue;
      }
      if (isNativeTimeOutEntry(entry)) {
        this.terms.push({
          name,
          term: new TimeOutTermination(runner, { ...entry, name }, () => this.elapsedS),
          isTimeOut: entry.time_out ?? false,
        });
        continue;
      }
      const TermClass = registry[entry.name];
      if (!TermClass) {
        console.warn(`[TerminationManager] Unknown termination type: ${entry.name}`);
        continue;
      }
      const termConfig: TerminationConfig = {
        name: entry.name,
        params: entry.params,
        time_out: entry.time_out,
      };
      this.terms.push({
        name,
        term: new TermClass(runner, termConfig),
        isTimeOut: entry.time_out ?? false,
      });
    }
  }

  /**
   * Build a traced-ONNX termination, or warn and skip it.
   *
   * Unlike an observation — which is part of the policy's input vector, so a
   * missing one would silently reshape it — dropping a termination only loses one
   * reset condition. Warning and continuing keeps the rest of the scene running.
   */
  private buildOnnxTermination(
    name: string,
    entry: OnnxTerminationConfig,
    runner: PolicyRunner,
    deps: TerminationManagerDeps,
  ): OnnxTermination | null {
    const session = deps.onnxSessions?.get(entry.onnx);
    const readSlot = deps.readOnnxSlot;
    if (!session || !readSlot) {
      console.warn(
        `[TerminationManager] "${name}" needs the ONNX session "${entry.onnx}" and a ` +
          'slot reader; skipping.',
      );
      return null;
    }
    return new OnnxTermination(runner, { ...entry, name }, { session, readSlot });
  }

  /**
   * Evaluate every term, OR-reducing to one verdict with `time_out` split out
   * (ADR 0005's manager table).
   *
   * `dt` is the control step, accumulated for the native `time_out` comparison.
   */
  evaluate(state: PolicyState, dt = 0): TerminationResult {
    this.elapsedS += dt;
    let terminated = false;
    let truncated = false;
    const reasons: string[] = [];

    for (const { name, term, isTimeOut } of this.terms) {
      if (term.evaluate(state)) {
        reasons.push(name);
        if (isTimeOut) {
          truncated = true;
        } else {
          terminated = true;
        }
      }
    }

    return {
      done: terminated || truncated,
      terminated,
      truncated,
      reasons,
    };
  }

  reset(): void {
    this.elapsedS = 0;
    for (const { term } of this.terms) {
      term.reset?.();
    }
  }

  get size(): number {
    return this.terms.length;
  }
}
