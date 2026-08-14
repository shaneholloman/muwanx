/** Build options both the SPA and the library config need, so the two cannot drift. */

/** Set MJSWAN_DEBUG=1 to keep console/debugger statements. */
export const isDebug = process.env.MJSWAN_DEBUG === '1';

/**
 * Console stripping, as `Builder(debug=...)` promises. A minifier option, not
 * `esbuild: { drop: [...] }`: Vite 8 is rolldown-based and ignores the deprecated
 * `esbuild` option, which shipped every `console.*` while looking handled.
 */
export const minify = isDebug
  ? true
  : { compress: { dropConsole: true, dropDebugger: true } };
