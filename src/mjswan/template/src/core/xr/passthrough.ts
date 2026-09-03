/**
 * What has to stop being drawn for a passthrough session to show the room.
 *
 * three.js already clears the framebuffer transparent once the session reports an
 * `alpha-blend` blend mode, so only mjswan's own drawing covers the room: a `CubeTexture`
 * on `scene.background` is rendered as a background box mesh that a transparent clear
 * underneath does not touch, and MuJoCo's infinite plane is a full-screen quad that paints
 * every pixel whose view ray meets z = 0.
 */
import * as THREE from 'three';

export class Passthrough {
  private readonly scene: THREE.Scene;
  private background: THREE.Scene['background'] = null;
  private active = false;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  enter(): void {
    if (!this.active) {
      this.active = true;
      this.hide();
    }
  }

  exit(): void {
    if (!this.active) {
      return;
    }
    this.active = false;
    this.scene.background = this.background;
    this.background = null;
    this.setPlanes(true);
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
    this.setPlanes(false);
  }

  /** Tagged by the scene loader. */
  private setPlanes(visible: boolean): void {
    this.scene.traverse((object) => {
      if (object.userData.isGroundPlane === true) {
        object.visible = visible;
      }
    });
  }
}
