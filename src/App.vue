<template>
  <v-app>
    <v-main>
      <router-view />
    </v-main>
  </v-app>
</template>

<script>
export default {
  name: 'App',
}
</script>

<style>
html,
body {
  height: 100%;
  overflow: hidden;
}

body,
.v-application {
  font-family: 'Google Sans', 'Noto Sans SC', Arial, sans-serif !important;
}

/* Minimal design tokens for a cleaner, consistent UI */
:root {
  --ui-surface: rgba(255, 255, 255, 0.9);
  --ui-border: rgba(0, 0, 0, 0.08);
  --ui-text: #111827;
  --ui-muted: #6b7280;
  --ui-radius: 8px;
  --ui-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

/* Refine default text color for better readability on overlays */
body {
  color: var(--ui-text);
}

/* Make Vuetify cards feel lighter by default without changing theme */
.v-card {
  border-radius: var(--ui-radius) !important;
}
</style>
