/**
 * A "START AR" button, beside three's `VRButton` rather than in place of it: a Quest
 * supports both modes.
 *
 * three ships an `ARButton`, but it pins the session to the `local` reference space, whose
 * origin is wherever the head started, which on a headset leaves the scene's floor at eye
 * height. This asks for `local-floor` instead, so MuJoCo's z = 0 lands on the real floor.
 */
import type * as THREE from 'three';

/** VRButton's own look, so the two buttons read as a pair. */
const STYLE =
  'position:absolute;bottom:20px;width:100px;padding:12px 6px;border:1px solid #fff;' +
  'border-radius:4px;background:rgba(0,0,0,0.1);color:#fff;font:normal 13px sans-serif;' +
  'text-align:center;opacity:0.5;outline:none;z-index:999;cursor:pointer';

/** Enters `immersive-ar`, or `null` where the device cannot. The caller positions it. */
export async function createArButton(
  renderer: THREE.WebGLRenderer,
  sessionInit: XRSessionInit = {}
): Promise<HTMLElement | null> {
  const xr = navigator.xr;
  if (!xr) {
    return null;
  }
  try {
    if (!(await xr.isSessionSupported('immersive-ar'))) {
      return null;
    }
  } catch (error: unknown) {
    console.warn('[AR] isSessionSupported failed:', error);
    return null;
  }

  // A reference space only resolves if the session asked for it up front.
  const options: XRSessionInit = {
    ...sessionInit,
    optionalFeatures: ['local-floor', ...(sessionInit.optionalFeatures ?? [])],
  };

  const button = document.createElement('button');
  button.style.cssText = STYLE;
  button.textContent = 'START AR';
  button.onmouseenter = () => (button.style.opacity = '1.0');
  button.onmouseleave = () => (button.style.opacity = '0.5');

  let current: XRSession | null = null;

  button.onclick = () => {
    if (current) {
      void current.end();
      return;
    }
    xr.requestSession('immersive-ar', options)
      .then(async (session) => {
        session.addEventListener('end', () => {
          current = null;
          button.textContent = 'START AR';
        });
        // Before `setSession`, which is where the space is actually requested.
        renderer.xr.setReferenceSpaceType('local-floor');
        await renderer.xr.setSession(session);
        current = session;
        button.textContent = 'STOP AR';
      })
      .catch((error: unknown) => {
        console.warn('[AR] requestSession failed:', error);
      });
  };

  return button;
}
