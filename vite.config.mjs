// Plugins
import Vue from '@vitejs/plugin-vue'
import Vuetify, { transformAssetUrls } from 'vite-plugin-vuetify'
import Fonts from 'unplugin-fonts/vite'

// Utilities
import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const isDemoMode = process.env.BUILD_MODE === 'demo';

  return {
    base: '/muwanx/',
    publicDir: isDemoMode ? 'examples' : false,
    build: isDemoMode ?
    {
      outDir: 'dist',
      rollupOptions: {
        input: fileURLToPath(new URL('./index.html', import.meta.url)),
      },
    } : {
      lib: {
        entry: fileURLToPath(new URL('./src/index.ts', import.meta.url)),
        name: 'Muwanx',
        fileName: (format) => `muwanx.${format}.js`,
      },
      rollupOptions: {
        external: ['vue', 'vuetify', 'vue-router', 'three', 'onnxruntime-web'],
        output: {
          globals: {
            vue: 'Vue',
            vuetify: 'Vuetify',
            'vue-router': 'VueRouter',
            three: 'THREE',
            'onnxruntime-web': 'ort',
          },
        },
      },
    },
    plugins: [
    Vue({
      template: { transformAssetUrls },
    }),
    // https://github.com/vuetifyjs/vuetify-loader/tree/master/packages/vite-plugin#readme
    Vuetify({
      styles: {
        configFile: './src/viewer/styles/settings.scss',
      },
    }),
    Fonts({
      fontsource: {
        families: [
          {
            name: 'Roboto',
            weights: [100, 300, 400, 500, 700, 900],
            styles: ['normal', 'italic'],
          },
        ],
      },
    }),
  ],
  optimizeDeps: {
    exclude: ['vuetify', 'onnxruntime-web'],
  },
  define: { 'process.env': {} },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@examples': fileURLToPath(new URL('./examples', import.meta.url)),
    },
    extensions: [
      '.js',
      '.json',
      '.jsx',
      '.mjs',
      '.ts',
      '.tsx',
      '.vue',
    ],
  },
  server: {
    port: 3000,
  },
    css: {
      preprocessorOptions: {
        sass: {
          api: 'modern-compiler',
        },
        scss: {
          api:'modern-compiler',
        },
      },
    },
  };
});
