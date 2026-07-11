import { defineConfig, devices } from '@playwright/test';

// Runtime-tier E2E: drives the React-free engine harness in a real browser.
// Headless Chromium renders WebGL via SwiftShader (software GL), so no GPU is
// required in CI. Unit tests (pure logic) live under vitest; run: npm run test:e2e.
const PORT = 5178;

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  fullyParallel: false,
  webServer: {
    command: `npm run dev -- --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/harness.html`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  use: {
    baseURL: `http://localhost:${PORT}`,
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
        },
      },
    },
  ],
});
