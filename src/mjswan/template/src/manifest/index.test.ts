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
    expect(catalog.name).toBe('Demo');
    expect(catalog.scenes.map((s) => s.name)).toEqual(['Humanoid']);
    const scene = catalog.scenes[0];
    expect(scene.splatSection).toBe(true);
    expect(scene.policies.map((p) => p.name)).toEqual(['walk', 'run']);
    expect(scene.policies[0].default).toBe(true);
    expect(scene.policies[0].motions).toEqual([{ name: 'clip', default: true }]);
    expect(scene.splats[0]).toMatchObject({ name: 'Room', control: true, transform: { scale: 2 } });
  });

  it('resolves asset paths under main/assets and relative to policy.json', async () => {
    const { source, requested } = fakeSource(POLICY_JSON);
    const catalog = parseManifest(CONFIG, source);
    const input = await catalog.scenes[0].buildScene({ policy: 'walk', splat: 'Room' });
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
    const input = await catalog.scenes[0].buildScene();
    expect(input.policy).not.toBeNull(); // default policy 'walk'
    expect(input.splat).toBeNull(); // no splat requested
  });

  it('prefers the id:null project and throws on an empty catalog', () => {
    expect(() => parseManifest({ version: '0', projects: [] }, fakeSource({}).source)).toThrow();
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
