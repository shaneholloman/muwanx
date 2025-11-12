import type { PolicyConfigItem, TaskConfigItem } from '@/types/config';

/**
 * Runtime utilities for engine configuration and initialization
 */

/**
 * Determines if Isaac action manager is needed based on asset metadata
 */
export async function needsIsaacActionManager(
    metaPath: string | null
): Promise<boolean> {
    if (!metaPath) return false;

    try {
        const resp = await fetch(metaPath);
        if (!resp.ok) return false;

        const meta = await resp.json();
        if (!meta || typeof meta !== 'object') return false;

        const hasIsaacJoints = Array.isArray((meta as any).joint_names_isaac) &&
            (meta as any).joint_names_isaac.length > 0;
        const hasActuators = (meta as any).actuators &&
            typeof (meta as any).actuators === 'object' &&
            Object.keys((meta as any).actuators).length > 0;
        const hasDefaultJpos = Array.isArray((meta as any).default_joint_pos) &&
            (meta as any).default_joint_pos.length > 0;

        return hasIsaacJoints || hasActuators || hasDefaultJpos;
    } catch (e) {
        console.warn('Failed to load asset_meta for action manager detection:', e);
        return false;
    }
}

/**
 * Validates WebAssembly support
 */
export function isWebAssemblySupported(): boolean {
    return typeof WebAssembly === 'object' &&
        typeof (WebAssembly as any).instantiate === 'function';
}
