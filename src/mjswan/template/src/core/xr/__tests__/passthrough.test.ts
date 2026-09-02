/** What a passthrough session takes out of the scene, and puts back. */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { Passthrough } from '../passthrough';

/** A scene shaped like a loaded one: a skybox, two plane geoms, and a robot. */
function loaded(): {
  scene: THREE.Scene;
  skybox: THREE.CubeTexture;
  floor: THREE.Mesh;
  backface: THREE.Mesh;
  robot: THREE.Mesh;
} {
  const scene = new THREE.Scene();
  const skybox = new THREE.CubeTexture();
  scene.background = skybox;

  const floor = new THREE.Mesh();
  floor.userData.isGroundPlane = true;
  const backface = new THREE.Mesh();
  backface.userData.isGroundPlane = true;
  const robot = new THREE.Mesh();

  // Nested, as the loader parents geoms under body groups under the MuJoCo root.
  const root = new THREE.Group();
  root.add(floor, backface, robot);
  scene.add(root);

  return { scene, skybox, floor, backface, robot };
}

describe('Passthrough', () => {
  it('clears the skybox and every ground plane, and nothing else', () => {
    const { scene, floor, backface, robot } = loaded();
    new Passthrough(scene).enter();

    expect(scene.background).toBeNull();
    expect(floor.visible).toBe(false);
    expect(backface.visible).toBe(false);
    expect(robot.visible).toBe(true);
  });

  it('puts both back when the session ends', () => {
    const { scene, skybox, floor } = loaded();
    const passthrough = new Passthrough(scene);

    passthrough.enter();
    passthrough.exit();

    expect(scene.background).toBe(skybox);
    expect(floor.visible).toBe(true);
  });

  it('keeps the skybox it saved when entered twice', () => {
    const { scene, skybox } = loaded();
    const passthrough = new Passthrough(scene);

    passthrough.enter();
    passthrough.enter();
    passthrough.exit();

    expect(scene.background).toBe(skybox);
  });

  it('reports whether it is holding the scene open', () => {
    const { scene } = loaded();
    const passthrough = new Passthrough(scene);

    expect(passthrough.isActive).toBe(false);
    passthrough.enter();
    expect(passthrough.isActive).toBe(true);
    passthrough.exit();
    expect(passthrough.isActive).toBe(false);
  });

  it('leaves a scene alone when a session it never entered ends', () => {
    const { scene, skybox, floor } = loaded();
    new Passthrough(scene).exit();

    expect(scene.background).toBe(skybox);
    expect(floor.visible).toBe(true);
  });

  // A scene switch mid-session rebuilds the graph, skybox and floor included.
  it('hides a scene loaded while the session runs, and restores that one', () => {
    const first = loaded();
    const passthrough = new Passthrough(first.scene);
    passthrough.enter();

    const rebuiltSkybox = new THREE.CubeTexture();
    first.scene.background = rebuiltSkybox;
    const rebuiltFloor = new THREE.Mesh();
    rebuiltFloor.userData.isGroundPlane = true;
    first.scene.add(rebuiltFloor);

    passthrough.refresh();
    expect(first.scene.background).toBeNull();
    expect(rebuiltFloor.visible).toBe(false);

    passthrough.exit();
    expect(first.scene.background).toBe(rebuiltSkybox);
    expect(rebuiltFloor.visible).toBe(true);
  });

  it('does not touch a scene on refresh outside a session', () => {
    const { scene, skybox, floor } = loaded();
    new Passthrough(scene).refresh();

    expect(scene.background).toBe(skybox);
    expect(floor.visible).toBe(true);
  });
});
