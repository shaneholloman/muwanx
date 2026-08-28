/**
 * MuJoCo material -> three.js PBR mapping.
 *
 * The property worth pinning: a material that sets only `rgba` — which is every
 * material in Menagerie and mjlab, G1's and Microduck's included — is a dielectric.
 * MuJoCo's `specular` defaults to 0.5, and standing it in for `metalness` made
 * three.js hand half of the base colour to a metallic reflection with no
 * `scene.environment` to reflect, so a white robot rendered as grey metal.
 */
import { describe, expect, it } from 'vitest';

import type { MjModel } from 'mujoco';
import { reflectanceParams } from '../scene';

/** What MuJoCo's compiler writes for a `<material>` attribute the MJCF leaves out. */
const MJ_DEFAULTS = { specular: 0.5, shininess: 0.5, metallic: -1, roughness: -1 };

function modelWithMaterial(attrs: Partial<typeof MJ_DEFAULTS> = {}): MjModel {
  const { specular, shininess, metallic, roughness } = { ...MJ_DEFAULTS, ...attrs };
  return {
    mat_specular: Float32Array.of(specular),
    mat_shininess: Float32Array.of(shininess),
    mat_metallic: Float32Array.of(metallic),
    mat_roughness: Float32Array.of(roughness),
  } as unknown as MjModel;
}

describe('reflectanceParams', () => {
  it('renders a material that declares no PBR attributes as a dielectric', () => {
    const params = reflectanceParams(modelWithMaterial(), 0);

    expect(params.metalness).toBe(0);
    expect(params.roughness).toBeCloseTo(0.5);
    expect(params.specularIntensity).toBeCloseTo(0.5);
  });

  it('leaves reflectivity alone, so three.js keeps its dielectric F0', () => {
    expect(reflectanceParams(modelWithMaterial(), 0)).not.toHaveProperty('reflectivity');
  });

  it("passes MuJoCo's own PBR attributes through when the material declares them", () => {
    const params = reflectanceParams(modelWithMaterial({ metallic: 1, roughness: 0.2 }), 0);

    expect(params.metalness).toBe(1);
    expect(params.roughness).toBeCloseTo(0.2);
  });

  it('treats an explicit metallic="0" as declared rather than unset', () => {
    const params = reflectanceParams(modelWithMaterial({ metallic: 0, specular: 0.9 }), 0);

    expect(params.metalness).toBe(0);
    expect(params.specularIntensity).toBeCloseTo(0.9);
  });

  it('falls back to MuJoCo defaults for a geom carrying no material', () => {
    const params = reflectanceParams(modelWithMaterial({ metallic: 1 }), -1);

    expect(params.metalness).toBe(0);
    expect(params.roughness).toBeCloseTo(0.5);
  });
});
