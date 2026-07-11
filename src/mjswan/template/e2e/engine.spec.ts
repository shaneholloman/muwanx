import { test, expect } from '@playwright/test';

// Runtime-tier acceptance for the headless engine: a real browser loads the
// React-free harness, `createEngine` builds a scene from `.mjz` bytes, and the
// captured frame must be a non-blank render. Complements the pure-logic vitest
// unit tests. See src/harness/e2e-entry.ts and ADR 0004.
test('createEngine renders a scene from bytes, React-free', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (err) => pageErrors.push(String(err)));

  await page.goto('/harness.html');
  await page.waitForFunction(() => window.__harness !== undefined, undefined, { timeout: 60_000 });

  const result = await page.evaluate(() => window.__harness);
  expect(result?.ok, result?.error).toBe(true);
  expect(result?.running).toBe(true);
  expect(result?.nonBlank, `luminance range ${JSON.stringify(result?.luminanceRange)}`).toBe(true);
  expect(pageErrors).toEqual([]);
});
