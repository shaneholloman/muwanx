<template>
  <v-card-text v-if="showPlayback" :class="{ 'mobile-padding': isMobile }">
    <div class="control-row">
      <div class="control-label">Playback</div>
      <div class="control-unit">
        <v-btn-toggle v-model="localState" mandatory class="mb-1">
          <v-btn @click="emit('play')" value="play" :size="isMobile ? 'default' : 'small'"
            prepend-icon="mdi-play">Play</v-btn>
          <v-btn @click="emit('stop')" value="stop" :size="isMobile ? 'default' : 'small'"
            prepend-icon="mdi-stop">Stop</v-btn>
          <v-btn @click="emit('reset')" value="reset" :size="isMobile ? 'default' : 'small'"
            prepend-icon="mdi-refresh">Reset</v-btn>
        </v-btn-toggle>
      </div>
    </div>
    <div class="control-row">
      <div class="control-label">Loop</div>
      <div class="control-unit">
        <v-checkbox v-model="localLoop" density="compact" hide-details class="inline-checkbox" />
      </div>
    </div>
  </v-card-text>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  selectedPolicy: { type: Object, default: null },
  state: { type: String, default: 'stop' },
  loop: { type: Boolean, default: false },
  isMobile: { type: Boolean, default: false },
})
const emit = defineEmits(['play', 'stop', 'reset', 'update:loop'])

const showPlayback = computed(() => Array.isArray(props.selectedPolicy?.ui_controls) && props.selectedPolicy.ui_controls.includes('trajectory_playback'))
const localState = computed({ get: () => props.state, set: () => { } })
const localLoop = computed({ get: () => props.loop, set: (v) => emit('update:loop', v) })
</script>
