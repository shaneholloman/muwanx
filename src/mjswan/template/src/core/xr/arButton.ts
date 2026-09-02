/**
 * An "START AR" button, alongside three's `VRButton` rather than in place of it: a Quest
 * supports both modes, and VR stays what the bundled app opens in.
 *
 * three ships an `ARButton`, but it pins the session to the `local` reference space, whose
 * origin is wherever the head started — on a headset that leaves the scene's floor at eye
 * height. This asks for `local-floor` instead, so MuJoCo's z = 0 lands on the real floor.
 */
import type * as THREE from 'three';

/** VRButton's own look, so the two buttons read as a pair. */
function stylize(element: HTMLElement): void {
  element.style.position = 'absolute';
  element.style.bottom = '20px';
  element.style.padding = '12px 6px';
  element.style.border = '1px solid #fff';
  element.style.borderRadius = '4px';
  element.style.background = 'rgba(0,0,0,0.1)';
  element.style.color = '#fff';
  element.style.font = 'normal 13px sans-serif';
  element.style.textAlign = 'center';
  element.style.opacity = '0.5';
  element.style.outline = 'none';
  element.style.zIndex = '999';
  element.style.cursor = 'pointer';
  element.style.width = '100px';
}

/**
 * A button that enters `immersive-ar`, or `null` where the device cannot. Unpositioned:
 * the caller places it, since it shares the bottom of the screen with the VR button.
 */
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

  // `requestReferenceSpace` only resolves for a space asked for up front; `local` is the
  // one an immersive session always has.
  const options: XRSessionInit = {
    ...sessionInit,
    optionalFeatures: ['local-floor', ...(sessionInit.optionalFeatures ?? [])],
  };

  const button = document.createElement('button');
  stylize(button);
  button.textContent = 'START AR';
  button.onmouseenter = () => (button.style.opacity = '1.0');
  button.onmouseleave = () => (button.style.opacity = '0.5');

  let current: XRSession | null = null;

  const onEnded = (): void => {
    current?.removeEventListener('end', onEnded);
    current = null;
    button.textContent = 'START AR';
  };

  button.onclick = () => {
    if (current) {
      current.end();
      return;
    }
    xr.requestSession('immersive-ar', options)
      .then(async (session) => {
        session.addEventListener('end', onEnded);
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
