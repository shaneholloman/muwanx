// TODO: Remove this module after mujoco wasm mj_loadXML() supports mjz format loading.

import JSZip from 'jszip';
import type { MainModule } from 'mujoco';

export async function loadMjzFile(
    mujoco: MainModule,
    mjzBytes: ArrayBuffer
): Promise<string> {
    // Unzip the .mjz archive supplied by the app into the Emscripten VFS.
    const zip = await JSZip.loadAsync(mjzBytes);

    // Find root XML
    let xmlPath: string | null = null;

    // Extract all files to Emscripten FS
    for (const [relativePath, file] of Object.entries(zip.files)) {
        if (file.dir) continue;

        // Sanitize ZIP entry path to prevent path traversal
        const sanitized = relativePath
            .replace(/\\/g, '/')
            .split('/')
            .filter((seg) => seg && seg !== '..' && seg !== '.')
            .join('/');
        if (!sanitized) continue;

        const content = await file.async('uint8array');
        const fsPath = `/working/${sanitized}`;

        // Create directories
        const dirPath = fsPath.substring(0, fsPath.lastIndexOf('/'));
        try {
            mujoco.FS.mkdirTree(dirPath, 0o777);
        } catch {
            // Directory exists
        }

        // Write file
        mujoco.FS.writeFile(fsPath, content);

        // Track root XML
        if (sanitized.endsWith('.xml') && !sanitized.includes('/')) {
            xmlPath = fsPath;
        }
    }

    if (!xmlPath) {
        throw new Error('No root XML found in .mjz archive');
    }

    return xmlPath;
}
