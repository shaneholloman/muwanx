// plugins/router.js
import { createRouter, createWebHistory, createWebHashHistory } from 'vue-router'
import Demo from '@/views/Demo.vue'

const routes = [
  {
    path: '/',
    name: 'Muwanx',
    component: Demo,
    props: () => ({ configPath: './config.json' })
  },
  {
    path: '/mujoco_menagerie',
    name: 'MuJoCo Menagerie',
    component: Demo,
    props: () => ({ configPath: './config_mujoco_menagerie.json' })
  },
  {
    path: '/mujoco_playground',
    name: 'MuJoCo Playground',
    component: Demo,
    props: () => ({ configPath: './config_mujoco_playground.json' })
  },
  {
    path: '/myosuite',
    name: 'MyoSuite',
    component: Demo,
    props: () => ({ configPath: './config_myosuite.json' })
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
