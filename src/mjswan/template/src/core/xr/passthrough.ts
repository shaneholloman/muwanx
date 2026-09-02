/**
 * What has to stop being drawn for a passthrough session to show the room.
 *
 * three.js already clears the framebuffer transparent once the session reports an
 * `alpha-blend` blend mode, so the empty parts of the view come for free. What covers the
 * room is what mjswan itself draws over it:
 *
 * - The skybox. A CubeTexture on `scene.background` is rendered as a background box mesh,
 *   so a transparent clear underneath it changes nothing.
 * - The ground planes. MuJoCo's infinite plane is a full-screen quad that paints every
 *   pixel whose view ray meets z = 0, which is most of the lower half of the view.
 *
 * Both are restored on the way out, and re-hidden after a scene rebuild, which brings its
 * own skybox and floor.
 */
import * as THREE from 'three';

import { isGroundPlane } from '../scene/scene';

export class Passthrough {
  private readonly scene: THREE.Scene;
  private background: THREE.Scene['background'] = null;
  private active = false;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  get isActive(): boolean {
    return this.active;
  }

  enter(): void {
    if (this.active) {
      return;
    }
    this.active = true;
    this.hide();
  }

  exit(): void {
    if (!this.active) {
      return;
    }
    this.active = false;
    this.scene.background = this.background;
    this.background = null;
    for (const plane of this.groundPlanes()) {
      plane.visible = true;
    }
  }

  /** After a scene rebuild mid-session, whose skybox and floor arrive visible. */
  refresh(): void {
    if (this.active) {
      this.hide();
    }
  }

  private hide(): void {
    // Guarded, or a second call would remember the null it just wrote.
    if (this.scene.background !== null) {
      this.background = this.scene.background;
    }
    this.scene.background = null;
    for (const plane of this.groundPlanes()) {
      plane.visible = false;
    }
  }

  private groundPlanes(): THREE.Object3D[] {
    const planes: THREE.Object3D[] = [];
    this.scene.traverse((object) => {
      if (isGroundPlane(object)) {
        planes.push(object);
      }
    });
    return planes;
  }
}
