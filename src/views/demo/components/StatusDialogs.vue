<template>
  <v-dialog :model-value="state === 0" persistent :max-width="isMobile ? '90vw' : '600px'"
    :fullscreen="isMobile && isSmallScreen" scrollable>
    <v-card class="status-dialog-card" :title="!isMobile ? 'Loading Simulation Environment' : undefined">
      <v-card-text class="dialog-content">
        <div v-if="isMobile" class="mobile-dialog-title">Loading...</div>
        <v-progress-linear indeterminate color="primary" class="mb-4"></v-progress-linear>
        <div class="loading-text">
          <span v-if="isMobile">Setting up simulation,<br />Please wait...</span>
          <span v-else>Setting up simulation, Please wait...</span>
        </div>
      </v-card-text>
    </v-card>
  </v-dialog>

  <v-dialog :model-value="state < 0 || !!urlParamErrorMessage" :persistent="state < 0"
    :max-width="isMobile ? '90vw' : '600px'" :fullscreen="isMobile && isSmallScreen" scrollable>
    <v-card class="status-dialog-card"
      :title="urlParamErrorMessage ? 'Invalid URL Parameters' : (isMobile ? 'Error' : 'Simulation Environment Loading Error')">
      <v-card-text class="dialog-content">
        <div class="error-text" v-if="state < 0">
          <span v-if="state == -1">Unexpected JS error, please refresh the page<br />{{ extraErrorMessage }}</span>
          <span v-if="state == -2">Your browser does not support WebAssembly, please use latest
            Chrome/Edge/Firefox</span>
        </div>
        <div v-if="urlParamErrorMessage" class="warning-content-inline">
          <v-icon color="warning" size="48" class="mb-3">mdi-alert</v-icon>
          <div class="warning-message">{{ urlParamErrorMessage }}</div>
        </div>
      </v-card-text>
      <v-card-actions v-if="urlParamErrorMessage">
        <v-spacer></v-spacer>
        <v-btn color="primary" variant="text" @click="$emit('clearWarning')">OK</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
defineProps({
  state: { type: Number, required: true },
  extraErrorMessage: { type: String, default: '' },
  urlParamErrorMessage: { type: String, default: '' },
  isMobile: { type: Boolean, default: false },
  isSmallScreen: { type: Boolean, default: false },
})
defineEmits(['clearWarning'])
</script>
