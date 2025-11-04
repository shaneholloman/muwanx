// plugins/router.js
import { createRouter, createWebHistory, createWebHashHistory } from 'vue-router'
import MuwanxViewer from '../MuwanxViewer.vue'

const routes = [
  {
    path: '/',
    name: 'Muwanx',
    component: MuwanxViewer,
    props: () => ({ configPath: './assets/config.json' })
  },
  {
    path: '/mujoco_menagerie',
    name: 'MuJoCo Menagerie',
    component: MuwanxViewer,
    props: () => ({ configPath: './assets/config_mujoco_menagerie.json' })
  },
  {
    path: '/mujoco_playground',
    name: 'MuJoCo Playground',
    component: MuwanxViewer,
    props: () => ({ configPath: './assets/config_mujoco_playground.json' })
  },
  {
    path: '/myosuite',
    name: 'MyoSuite',
    component: MuwanxViewer,
    props: () => ({ configPath: './assets/config_myosuite.json' })
  },
]

const router = createRouter({
  history: createWebHashHistory('/'),
  routes,
})

// Reinitialize only when the base route (path) changes.
// Allow query-only updates (e.g., scene/policy) without full reload.
router.beforeEach((to, from, next) => {
  if (from.name && to.path !== from.path) {
    window.location.hash = '#' + to.fullPath
    window.location.reload()
  } else {
    next();
  }
});

export default router
