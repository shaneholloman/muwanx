/**
 * main.ts
 *
 * Demo app using the muwanx package with imperative API
 */

import { createApp, h, ref } from 'vue'
import { registerPlugins } from '@/viewer/plugins'
import { MwxViewer, MwxViewerComponent } from 'muwanx'
import type { LegacyAppConfig } from 'muwanx'
import 'unfonts.css'

// Define available projects
const projects = [
  { id: 'default', name: 'Muwanx Demo', config: './assets/config.json' },
  { id: 'menagerie', name: 'MuJoCo Menagerie', config: './assets/config_mujoco_menagerie.json' },
  { id: 'playground', name: 'MuJoCo Playground', config: './assets/config_mujoco_playground.json' },
  { id: 'myosuite', name: 'MyoSuite', config: './assets/config_myosuite.json' },
]

// Get config path from URL hash
function getConfigFromHash(): string {
  const hash = window.location.hash.slice(1).split('?')[0] // Remove # and query params
  const project = projects.find(p => p.id === hash || `/${p.id}` === hash)
  return project ? project.config : projects[0].config
}

// Load and build viewer using imperative API
async function buildViewer(containerEl: HTMLElement, configPath: string) {
  // Fetch the config file
  const response = await fetch(configPath)
  const config: LegacyAppConfig = await response.json()

  // Create viewer instance (headless - no Vue component)
  const viewer = new MwxViewer(containerEl)

  // Build project using imperative API
  const project = viewer.addProject({
    project_name: config.project_name,
    project_link: config.project_link,
  })

  // Add scenes (tasks in legacy config)
  if (config.tasks) {
    for (const task of config.tasks) {
      const scene = project.addScene({
        id: task.id,
        name: task.name,
        model_xml: task.model_xml,
        asset_meta: task.asset_meta,
      })

      // Set default policy if specified
      if (task.default_policy) {
        scene.setDefaultPolicy(task.default_policy)
      }

      // Add policies to scene
      if (task.policies) {
        for (const policyConfig of task.policies) {
          scene.addPolicy({
            id: policyConfig.id,
            name: policyConfig.name,
            path: policyConfig.path,
            ui_controls: policyConfig.ui_controls,
          })
        }
      }
    }
  }

  // Initialize the viewer
  await viewer.initialize()

  return viewer
}

// Create root component inline
const App = {
  setup() {
    const configPath = getConfigFromHash()
    const configObject = ref(null)

    // Load and build config using imperative approach
    async function loadImperativeConfig() {
      const response = await fetch(configPath)
      const legacyConfig: LegacyAppConfig = await response.json()

      // Build config object imperatively (demonstrating the pattern)
      configObject.value = {
        project_name: legacyConfig.project_name,
        project_link: legacyConfig.project_link,
        tasks: legacyConfig.tasks?.map(task => ({
          id: task.id,
          name: task.name,
          model_xml: task.model_xml,
          asset_meta: task.asset_meta,
          default_policy: task.default_policy,
          policies: task.policies?.map(policy => ({
            id: policy.id,
            name: policy.name,
            path: policy.path,
            ui_controls: policy.ui_controls,
          })) || []
        })) || []
      }
    }

    // Load config on setup
    loadImperativeConfig()

    return () => h('div', {
      style: {
        width: '100%',
        height: '100vh',
        margin: 0,
        padding: 0,
        overflow: 'hidden'
      }
    }, [
      configObject.value ? h(MwxViewerComponent, {
        config: configObject.value,
        key: configPath // Force re-mount on config change
      }) : h('div', 'Loading...')
    ])
  }
}

// Create and mount app
const app = createApp(App)
registerPlugins(app)
app.mount('#app')

// Handle hash changes (back/forward navigation)
window.addEventListener('hashchange', () => {
  // Reload page to reinitialize MuJoCo runtime with new config
  window.location.reload()
})
