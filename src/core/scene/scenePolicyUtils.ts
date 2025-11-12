import type { TaskConfigItem, PolicyConfigItem } from '@/types/config';

/**
 * Resolves the scene configuration from task and policy items
 */
export function resolveSceneConfig(
    task: TaskConfigItem | null,
    policy: PolicyConfigItem | null
): { scenePath: string | null; metaPath: string | null } {
    if (!task) {
        return { scenePath: null, metaPath: null };
    }

    const scenePath = policy?.model_xml ?? task.model_xml;
    const metaRaw = policy?.asset_meta ?? task.asset_meta ?? null;
    const metaPath = metaRaw === 'null' || metaRaw === '' ? null : metaRaw;

    return { scenePath, metaPath };
}

/**
 * Checks if asset metadata path exists and is valid
 */
export function hasAssetMeta(metaPath: string | null | undefined): boolean {
    return Boolean(metaPath);
}

/**
 * Loads asset metadata from the given path
 */
export async function loadAssetMeta(metaPath: string | null): Promise<any | null> {
    if (!metaPath) return null;

    try {
        const resp = await fetch(metaPath);
        if (!resp.ok) return null;
        return await resp.json();
    } catch (e) {
        console.warn('Failed to load asset_meta for inspection:', e);
        return null;
    }
}
