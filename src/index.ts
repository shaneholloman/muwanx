/**
 * Muwanx
 *
 * Main package entry point
 * This file exports the core components and utilities for use as an npm package
 */

// Export type definitions
export type * from './types/config';
export type * from './types/events';
export type * from './types/state';

// Export core functionality (MuJoCo physics, scene, agent, etc.)
export * from './core';

// Export viewer components and composables
export * from './viewer';
