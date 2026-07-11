import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { vanillaExtractPlugin } from '@vanilla-extract/vite-plugin';
import path from 'path';
import fs from 'fs';

// Library build: emits a single self-contained ESM (`dist/mjswan.js`) exposing
// `createEngine(element, options?)` (the headless engine; no React/Mantine),
// with every dependency bundled and the MuJoCo / ONNX WASM co-located flat in
// `dist/` so they resolve relative to the bundle on a public CDN (jsDelivr).
// See src/engine/ and docs/adr/0004-headless-engine-core.md.

function getOrtCdnBase(): string {
  // Bake the installed ort version into the bundle so OnnxModule.ts can redirect
  // ort's dynamic file fetches to its own CDN package (*.jsep.mjs, *.wasm, etc.).
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const ortPkg = require('./node_modules/onnxruntime-web/package.json');
  return `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortPkg.version}/dist/`;
}

function getVersionFromPython(): string {
  const initPath = path.resolve(__dirname, '../__init__.py');
  try {
    const content = fs.readFileSync(initPath, 'utf-8');
    const match = content.match(/__version__\s*=\s*["']([^"']+)["']/);
    if (match) {
      return match[1];
    }
  } catch {
    // Fall through to package.json.
  }
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pkg = require('./package.json');
  return pkg.version || '0.0.0';
}

// Vite library mode force-inlines `new URL('x.wasm', import.meta.url)` as base64
// `data:` URLs (ignoring assetsInlineLimit), bloating the bundle to ~70 MB. This
// extracts each back to a co-located dist/ file resolved via `import.meta.url` so
// it loads relative to the CDN bundle. See mjswan-cloud ADR 0001.
function extractInlinedWasmPlugin(): Plugin {
  const B64 = '([A-Za-z0-9+/=]+)';
  // Two inlined shapes, each with a base arg re-checked below: normally quoted
  // (`"`/`'`/backtick, backreferenced), and — inside a dormant pthread-worker
  // string — escaped `\"data:...\"` (keep the escaping so the literal stays valid).
  const QUOTED = new RegExp(
    `new URL\\(\\s*(["'\`])data:application/wasm;base64,${B64}\\1\\s*,\\s*([^)]*)\\)`,
    'g'
  );
  const ESCAPED = new RegExp(`new URL\\(\\s*\\\\"data:application/wasm;base64,${B64}\\\\"\\s*,\\s*[^)]*\\)`, 'g');

  return {
    name: 'mjswan-extract-wasm',
    apply: 'build',
    enforce: 'post',
    generateBundle(_options, bundle) {
      const emitted = new Map<string, string>(); // base64 → emitted fileName
      const fileFor = (b64: string): string => {
        let fileName = emitted.get(b64);
        if (!fileName) {
          const ref = this.emitFile({
            type: 'asset',
            name: 'mjswan-engine.wasm',
            source: Buffer.from(b64, 'base64'),
          });
          fileName = this.getFileName(ref);
          emitted.set(b64, fileName);
        }
        return fileName;
      };
      const deinline = (code: string): string =>
        code
          .replace(
            QUOTED,
            (m, _quote: string, b64: string, base: string) => {
              // Only de-inline main-thread wasm (base `import.meta.url`). A
              // `self.location.href` base means a classic Blob worker (Spark's
              // Splat sort) where `import.meta` is a syntax error — leave inline.
              if (!/import\.meta\.url/.test(base)) return m;
              return `new URL(${JSON.stringify('./' + fileFor(b64))}, import.meta.url)`;
            }
          )
          .replace(
            ESCAPED,
            (m, b64: string) => {
              // Keep small active-worker wasm (Spark) inline; only extract large dormant wasm.
              if (b64.length < 1_000_000) return m;
              return `new URL(\\"./${fileFor(b64)}\\", self.location.href)`;
            }
          );

      for (const [fileName, chunk] of Object.entries(bundle)) {
        if (chunk.type === 'chunk') {
          if (chunk.code.includes('data:application/wasm')) {
            chunk.code = deinline(chunk.code);
          }
        } else if (chunk.type === 'asset' && fileName.endsWith('.js')) {
          // Emitted worker modules arrive as JS assets, not chunks.
          const source =
            typeof chunk.source === 'string'
              ? chunk.source
              : Buffer.from(chunk.source).toString('utf-8');
          if (source.includes('data:application/wasm')) {
            chunk.source = deinline(source);
          }
        }
      }
    },
  };
}

// Inline all bundled CSS (including vanilla-extract output) into the JS entry so
// a single `import('mjswan.js')` brings its own styles — no separate stylesheet
// the host page must remember to load.
function cssInjectedByJsPlugin(): Plugin {
  return {
    name: 'mjswan-css-inject',
    apply: 'build',
    enforce: 'post',
    generateBundle(_options, bundle) {
      let css = '';
      for (const [fileName, chunk] of Object.entries(bundle)) {
        if (chunk.type === 'asset' && fileName.endsWith('.css')) {
          css +=
            typeof chunk.source === 'string'
              ? chunk.source
              : Buffer.from(chunk.source).toString('utf-8');
          delete bundle[fileName];
        }
      }
      if (!css) return;
      const entry = Object.values(bundle).find(
        (chunk) => chunk.type === 'chunk' && chunk.isEntry
      );
      if (entry && entry.type === 'chunk') {
        const injector =
          `(function(){try{` +
          `var d=document;if(!d||d.getElementById('mjswan-styles'))return;` +
          `var s=d.createElement('style');s.id='mjswan-styles';` +
          `s.textContent=${JSON.stringify(css)};` +
          `(d.head||d.documentElement).appendChild(s);` +
          `}catch(e){console.error('mjswan: failed to inject styles',e);}})();\n`;
        entry.code = injector + entry.code;
      }
    },
  };
}

export default defineConfig({
  plugins: [
    react(),
    vanillaExtractPlugin(),
    extractInlinedWasmPlugin(),
    cssInjectedByJsPlugin(),
  ],
  define: {
    __APP_VERSION__: JSON.stringify(getVersionFromPython()),
    // Library mode (unlike app mode) does NOT replace `process.env.NODE_ENV`, so
    // React/Mantine's bare `process` references would throw `ReferenceError` when
    // loaded from a CDN. Fold to "production"; other `process`/`Buffer` refs are
    // runtime-guarded (`typeof process < "u"`) Node paths that never run here.
    'process.env.NODE_ENV': JSON.stringify('production'),
    // Lib-build only: redirect ort's dynamic fetches to its own CDN package.
    __ORT_CDN_BASE__: JSON.stringify(getOrtCdnBase()),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  optimizeDeps: {
    exclude: ['mujoco', 'mujoco/mt'],
  },
  assetsInclude: ['**/*.wasm'],
  worker: {
    format: 'es',
    // Workers are dormant single-threaded (no SharedArrayBuffer); emit them flat
    // in dist/ so they sit alongside the extracted WASM (the extract plugin runs
    // over them as JS assets of the main bundle).
    rollupOptions: {
      output: {
        entryFileNames: '[name]-[hash].js',
        chunkFileNames: '[name]-[hash].js',
        assetFileNames: '[name]-[hash][extname]',
      },
    },
  },
  build: {
    outDir: 'dist',
    // The SPA build (vite build) runs first and empties dist/; the lib build
    // runs second and must preserve those files alongside mjswan.js.
    emptyOutDir: false,
    // Discourage inlining of small assets. NB: lib mode still force-inlines the
    // multi-MB WASM regardless of this value — extractInlinedWasmPlugin above is
    // what actually pulls them back out into co-located files.
    assetsInlineLimit: 0,
    sourcemap: false,
    chunkSizeWarningLimit: 11000,
    // Bundle all CSS into one asset so the inject plugin can hoist it into JS.
    cssCodeSplit: false,
    lib: {
      // The engine entry (createEngine) — no React/Mantine in the CDN bundle.
      entry: path.resolve(__dirname, 'src/engine/index.ts'),
      formats: ['es'],
      fileName: () => 'mjswan.js',
    },
    rollupOptions: {
      // Bundle everything — nothing is external. A bare import left in the
      // output would be unresolvable from jsDelivr.
      external: [],
      output: {
        // Flat layout in dist/: mjswan.js next to its chunks and *.wasm, so
        // `new URL('./x.wasm', import.meta.url)` resolves on the CDN.
        entryFileNames: 'mjswan.js',
        chunkFileNames: '[name]-[hash].js',
        assetFileNames: '[name]-[hash][extname]',
      },
      onwarn(warning, warn) {
        if (
          warning.message.includes('mujoco') &&
          warning.message.includes('externalized for browser compatibility')
        )
          return;
        warn(warning);
      },
    },
  },
});
