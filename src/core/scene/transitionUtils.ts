/**
 * Utilities for managing UI transitions
 */

/**
 * Executes an action with a minimum visible duration for smooth UX
 */
export async function withMinimumDuration(
    action: () => Promise<any> | any,
    minVisibleMs: number = 150
): Promise<any> {
    const start = (typeof performance !== 'undefined' && performance.now)
        ? performance.now()
        : Date.now();

    const result = await action();

    const end = (typeof performance !== 'undefined' && performance.now)
        ? performance.now()
        : Date.now();
    const elapsed = end - start;

    if (elapsed < minVisibleMs) {
        await new Promise<void>(r => setTimeout(r, minVisibleMs - elapsed));
    }

    return result;
}

/**
 * Waits for frames to ensure rendering before heavy operations
 */
export async function waitForFrames(count: number = 2): Promise<void> {
    for (let i = 0; i < count; i++) {
        await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
    }
    // Add a macrotask to guarantee a paint on busy UIs
    await new Promise<void>(resolve => setTimeout(resolve, 0));
}
