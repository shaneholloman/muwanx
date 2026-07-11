import { defineConfig } from 'vitest/config';

// Node-only unit tests for the framework-free, pure-logic layers (manifest,
// byte/npz parsing). The engine runtime itself needs WebGL + WASM and is
// exercised by the browser harness (src/engine/harness.ts), not here.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
});
