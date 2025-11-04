/**
 * Muwanx
 *
 * Main package entry point
 * This file exports the core components and utilities for use as an npm package
 */

// Export the main viewer component
export { default as MuwanxViewer } from './viewer/MuwanxViewer.vue'

// Export viewer components for customization
export { default as ControlPanel } from './viewer/components/ControlPanel.vue'
export { default as StatusDialogs } from './viewer/components/StatusDialogs.vue'
export { default as StatusOverlay } from './viewer/components/StatusOverlay.vue'
export { default as HelpDialog } from './viewer/components/HelpDialog.vue'
export { default as Notice } from './viewer/components/Notice.vue'
export { default as CommandControls } from './viewer/components/CommandControls.vue'
export { default as ForceControls } from './viewer/components/ForceControls.vue'
export { default as PolicySelector } from './viewer/components/PolicySelector.vue'
export { default as ProjectSelector } from './viewer/components/ProjectSelector.vue'
export { default as SceneSelector } from './viewer/components/SceneSelector.vue'
export { default as StiffnessControls } from './viewer/components/StiffnessControls.vue'
export { default as TrajectoryControls } from './viewer/components/TrajectoryControls.vue'

// Export composables for advanced usage
export { useConfig } from './viewer/composables/useConfig'
export { useResponsive } from './viewer/composables/useResponsive'
export { useRuntime } from './viewer/composables/useRuntime'
export { useScenePolicy } from './viewer/composables/useScenePolicy'
export { useTransition } from './viewer/composables/useTransition'
export { useUrlSync } from './viewer/composables/useUrlSync'

// Export constants
export * from './viewer/constants'

// Export type definitions
export type * from './types/config'
export type * from './types/events'
export type * from './types/state'

// Export core MuJoCo utilities
export { MujocoRuntime } from './core/mujoco/runtime/MujocoRuntime'
