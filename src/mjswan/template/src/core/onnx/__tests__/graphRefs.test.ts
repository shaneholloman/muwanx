/**
 * Which traced term graphs a config refers to (ADR 0005 §4).
 *
 * Getting this wrong is quiet: a ref that is missed means no session, which means
 * the manager warns and drops that term — a policy that runs with a hole in its
 * observation vector, or a termination that never fires.
 */
import { describe, expect, it } from 'vitest';

import { eventGraphRefs, policyGraphRefs } from '../graphRefs';

describe('policyGraphRefs', () => {
  it('collects refs from observation groups, terminations and commands', () => {
    expect(
      policyGraphRefs({
        onnx: { path: 'walk.onnx', meta: { in_keys: ['obs'] } },
        observations: {
          policy: [
            { name: 'joint_pos', onnx: 'obs/joint_pos.onnx' },
            { name: 'actions', native: 'prev_action' },
          ],
          critic: [{ name: 'joint_vel', onnx: 'obs/joint_vel.onnx' }],
        },
        terminations: {
          fell_over: { name: 'fell_over', onnx: 'term/fell_over.onnx' },
          time_out: { name: 'time_out', native: 'elapsed_s >= episode_length_s' },
        },
        commands: { twist: { name: 'OnnxCommand', onnx: 'command/twist.onnx' } },
      }),
    ).toEqual([
      'command/twist.onnx',
      'obs/joint_pos.onnx',
      'obs/joint_vel.onnx',
      'term/fell_over.onnx',
    ]);
  });

  it('ignores the policy network, whose `onnx` is an object', () => {
    // The network arrives as PolicyInput.onnx; treating it as a term graph loads it twice.
    expect(policyGraphRefs({ onnx: { path: 'walk.onnx' } })).toEqual([]);
  });

  it('de-duplicates a graph two terms share, and sorts', () => {
    // Observation fusion (ADR 0005 §4) points several terms at one graph.
    expect(
      policyGraphRefs({
        observations: {
          policy: [
            { name: 'b', onnx: 'obs/fused.onnx' },
            { name: 'a', onnx: 'obs/fused.onnx' },
          ],
        },
        terminations: { z: { onnx: 'term/z.onnx' } },
      }),
    ).toEqual(['obs/fused.onnx', 'term/z.onnx']);
  });

  it('is empty for a config with no traced terms', () => {
    expect(policyGraphRefs({})).toEqual([]);
    expect(policyGraphRefs({ observations: {}, terminations: {}, commands: {} })).toEqual([]);
  });
});

describe('eventGraphRefs', () => {
  it('collects refs from an event list and ignores native events', () => {
    expect(
      eventGraphRefs([
        { name: 'push_robot', onnx: 'event/push_robot.onnx' },
        { name: 'randomize_terrain', kind: 'event', mutations: [] },
        { name: 'reset_joints', onnx: 'event/reset_joints.onnx' },
      ]),
    ).toEqual(['event/push_robot.onnx', 'event/reset_joints.onnx']);
  });

  it('is empty for a scene with no events', () => {
    expect(eventGraphRefs(undefined)).toEqual([]);
    expect(eventGraphRefs([])).toEqual([]);
  });
});

describe('policyGraphRefs — fused groups', () => {
  it('collects a fused group graph, named by `fused` rather than `onnx`', () => {
    // The whole group is one graph, so an `onnx`-only collector delivers no bytes.
    expect(
      policyGraphRefs({
        onnx: { path: 'walk.onnx' },
        observations: {
          policy: { fused: 'obs/policy.onnx', size: 99, layout: [] },
          critic: { fused: 'obs/critic.onnx', size: 12, layout: [] },
        },
        terminations: { fell_over: { onnx: 'term/fell_over.onnx' } },
      }),
    ).toEqual(['obs/critic.onnx', 'obs/policy.onnx', 'term/fell_over.onnx']);
  });

  it('collects a fused termination graph beside its native siblings', () => {
    // Terminations fuse too, under a reserved key: one graph, several named terms.
    expect(
      policyGraphRefs({
        terminations: {
          time_out: { native: 'elapsed_s >= episode_length_s', episode_length_s: 20 },
          __fused__: {
            fused: 'term/terminations.onnx',
            lanes: [{ name: 'anchor_pos' }, { name: 'anchor_ori' }],
          },
        },
      }),
    ).toEqual(['term/terminations.onnx']);
  });
});
