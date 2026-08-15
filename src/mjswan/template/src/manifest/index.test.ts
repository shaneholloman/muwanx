import { describe, it, expect } from 'vitest';
import { parseManifest, sanitizeName, type AppConfig, type ByteSource } from './index';

/** Records every requested path; returns the policy JSON for *.json, tagged bytes otherwise. */
function fakeSource(policyJson: unknown): { source: ByteSource; requested: string[] } {
  const requested: string[] = [];
  const source: ByteSource = (relPath) => {
    requested.push(relPath);
    return async () => {
      const body = relPath.endsWith('.json') ? JSON.stringify(policyJson) : `bytes:${relPath}`;
      return new TextEncoder().encode(body).buffer as ArrayBuffer;
    };
  };
  return { source, requested };
}

const CONFIG: AppConfig = {
  version: '0.0.0',
  projects: [
    {
      name: 'Demo',
      id: null,
      scenes: [
        {
          name: 'Humanoid',
          path: 'humanoid/scene.mjz',
          camera: { distance: 5 },
          splatSection: true,
          splats: [{ name: 'Room', path: 'humanoid/room.spz', control: true, scale: 2 }],
          policies: [
            { name: 'walk', config: 'humanoid/walk.json', default: true, motions: [{ name: 'clip', default: true }] },
            { name: 'run', config: 'humanoid/run.json' },
          ],
        },
      ],
    },
  ],
};

const POLICY_JSON = {
  onnx: { path: 'walk.onnx' },
  motions: [{ name: 'clip', path: 'walk_clip.npz', fps: 50 }],
};

describe('parseManifest', () => {
  it('builds a single-project catalog with scene/policy/splat entries', () => {
    const { source } = fakeSource(POLICY_JSON);
    const catalog = parseManifest(CONFIG, source);
    expect(catalog.projects.map((p) => p.name)).toEqual(['Demo']);
    expect(catalog.projects[0].scenes.map((s) => s.name)).toEqual(['Humanoid']);
    const scene = catalog.projects[0].scenes[0];
    expect(scene.splatSection).toBe(true);
    expect(scene.policies.map((p) => p.name)).toEqual(['walk', 'run']);
    expect(scene.policies[0].default).toBe(true);
    expect(scene.policies[0].motions).toEqual([{ name: 'clip', default: true }]);
    expect(scene.splats[0]).toMatchObject({ name: 'Room', control: true, transform: { scale: 2 } });
  });

  it('resolves asset paths under main/assets and relative to policy.json', async () => {
    const { source, requested } = fakeSource(POLICY_JSON);
    const catalog = parseManifest(CONFIG, source);
    const input = await catalog.projects[0].scenes[0].buildScene({ policy: 'walk', splat: 'Room' });
    expect(requested).toContain('main/assets/humanoid/scene.mjz'); // model
    expect(requested).toContain('main/assets/humanoid/walk.json'); // policy.json
    expect(requested).toContain('main/assets/humanoid/walk.onnx'); // onnx (rel to policy dir)
    expect(requested).toContain('main/assets/humanoid/walk_clip.npz'); // motion (rel to policy dir)
    expect(requested).toContain('main/assets/humanoid/room.spz'); // splat
    expect(input.viewer).toEqual({ distance: 5 });
    expect(input.policy?.motions?.map((m) => m.name)).toEqual(['clip']);
  });

  it('defaults the policy and omits the splat, and accepts a JSON string', async () => {
    const { source } = fakeSource(POLICY_JSON);
    const catalog = parseManifest(JSON.stringify(CONFIG), source);
    const input = await catalog.projects[0].scenes[0].buildScene();
    expect(input.policy).not.toBeNull(); // default policy 'walk'
    expect(input.splat).toBeNull(); // no splat requested
  });

  it('exposes all projects, ordering id:null first, and throws on an empty catalog', () => {
    const multi: AppConfig = {
      version: '0',
      projects: [
        { name: 'Extra', id: 'extra', scenes: [] },
        { name: 'Main', id: null, scenes: [] },
      ],
    };
    const catalog = parseManifest(multi, fakeSource({}).source);
    expect(catalog.projects.map((p) => p.name)).toEqual(['Main', 'Extra']);
    expect(() => parseManifest({ version: '0', projects: [] }, fakeSource({}).source)).toThrow();
  });

  it('delivers the traced term graphs a policy.json refers to (ADR 0005)', async () => {
    // One graph per traced term beside the network, keyed by the config-relative path
    // the runtime looks a session up by, and fetched relative to policy.json.
    const traced = {
      onnx: { path: 'walk.onnx' },
      observations: {
        policy: [
          { name: 'joint_pos', onnx: 'obs/joint_pos.onnx' },
          { name: 'actions', native: 'prev_action' },
        ],
      },
      terminations: { fell_over: { name: 'fell_over', onnx: 'term/fell_over.onnx' } },
      commands: { twist: { name: 'OnnxCommand', onnx: 'command/twist.onnx' } },
    };
    const { source, requested } = fakeSource(traced);
    const catalog = parseManifest(CONFIG, source);
    const input = await catalog.projects[0].scenes[0].buildScene({ policy: 'walk' });

    expect(Object.keys(input.policy?.graphs ?? {}).sort()).toEqual([
      'command/twist.onnx',
      'obs/joint_pos.onnx',
      'term/fell_over.onnx',
    ]);
    expect(requested).toContain('main/assets/humanoid/obs/joint_pos.onnx');
    expect(requested).toContain('main/assets/humanoid/term/fell_over.onnx');
    expect(requested).toContain('main/assets/humanoid/command/twist.onnx');
    // The policy network's own `onnx` is an object, not a term reference.
    expect(input.policy?.graphs).not.toHaveProperty('walk.onnx');
  });

  it('resolves event graphs relative to the model, not to policy.json', async () => {
    // Event graphs sit beside the scene model, and a scene may have them with no policy.
    const withEvents: AppConfig = {
      ...CONFIG,
      projects: [
        {
          ...CONFIG.projects[0],
          scenes: [
            {
              ...CONFIG.projects[0].scenes[0],
              events: [
                { name: 'push_robot', onnx: 'event/push_robot.onnx' },
                { name: 'randomize_terrain' },
              ] as never,
            },
          ],
        },
      ],
    };
    const { source, requested } = fakeSource(POLICY_JSON);
    const catalog = parseManifest(withEvents, source);
    const input = await catalog.projects[0].scenes[0].buildScene({ policy: 'walk' });

    expect(Object.keys(input.graphs ?? {})).toEqual(['event/push_robot.onnx']);
    expect(requested).toContain('main/assets/humanoid/event/push_robot.onnx');
  });

  it('surfaces the runtime plugin module path when present', () => {
    const withPlugins = { ...CONFIG, plugins: 'assets/plugins.js' };
    expect(parseManifest(withPlugins, fakeSource(POLICY_JSON).source).pluginsPath).toBe(
      'assets/plugins.js',
    );
    expect(parseManifest(CONFIG, fakeSource(POLICY_JSON).source).pluginsPath).toBeUndefined();
  });
});

describe('sanitizeName', () => {
  it('lowercases and underscores spaces and hyphens', () => {
    expect(sanitizeName('Foo Bar-Baz')).toBe('foo_bar_baz');
  });
});
