// Browser-global signal that the scene has loaded and the first stable frame
// rendered. External tooling (e.g. headless thumbnail capture) polls
// `window.__mjswanReady` and/or subscribes to the `mjswan:ready` event to know
// when it is safe to screenshot, without depending on mount()'s Promise.

declare global {
  interface Window {
    __mjswanReady?: boolean;
    __mjswanError?: boolean;
  }
}

let readyDispatched = false;

/** Mark the scene as ready. Idempotent: the event fires at most once per load. */
export function signalReady(): void {
  if (typeof window === 'undefined') return;
  window.__mjswanReady = true;
  if (!readyDispatched) {
    readyDispatched = true;
    window.dispatchEvent(new CustomEvent('mjswan:ready'));
  }
}

/** Mark the scene load as failed so pollers can fail early. Never sets ready. */
export function signalError(): void {
  if (typeof window === 'undefined') return;
  window.__mjswanError = true;
}
