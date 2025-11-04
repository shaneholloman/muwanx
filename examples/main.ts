/**
 * main.ts
 *
 * Bootstraps Vuetify and other plugins then mounts the App
 */

import { registerPlugins } from '@/viewer/plugins'
import App from './App.vue'
import { createApp } from 'vue'
import 'unfonts.css'

const app = createApp(App)

registerPlugins(app)

app.mount('#app')
