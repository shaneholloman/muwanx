<template>
  <v-overlay :model-value="isTransitioning.value" :z-index="5000" class="switch-overlay" scrim="rgba(0,0,0,0.35)">
    <div class="overlay-content">
      <v-progress-circular indeterminate color="primary" size="64" />
      <div class="transition-text">{{ transitionMessage.value || 'Loading...' }}</div>
    </div>
  </v-overlay>

  <div id="mujoco-container" class="mujoco-container" />

  <div class="control-panel" :class="{ 'panel-collapsed': isPanelCollapsed }">
    <v-btn v-if="isMobile" class="panel-toggle" @click="togglePanel" icon size="small" color="primary" elevation="2">
      <v-icon>{{ isPanelCollapsed ? 'mdi-chevron-up' : 'mdi-chevron-down' }}</v-icon>
    </v-btn>

    <ControlPanel :project-name="projectName" :project-link="projectLink" :route-items="routeItems"
      :current-route-name="$route.name" :is-mobile="isMobile" :task-items="taskItems" :task-id="task"
      :policy-items="policyItems" :policy-id="policy" :selected-task="selectedTask" :selected-policy="selectedPolicy"
      :use-setpoint="use_setpoint" :command-vel-x="command_vel_x" :compliant-mode="compliant_mode" :facet-kp="facet_kp"
      :trajectory-state="trajectoryPlaybackState" :trajectory-loop="trajectoryLoop" @navigateRoute="goToRoute"
      @selectTask="onSelectTask" @selectPolicy="onSelectPolicy"
      @update:useSetpoint="onUpdateUseSetpoint"
      @update:commandVelX="onUpdateCommandVelX"
      @update:facetKp="onUpdateFacetKp"
      @update:compliantMode="onUpdateCompliantMode"
      @playTrajectory="playTrajectory" @stopTrajectory="stopTrajectory" @resetTrajectory="resetTrajectory"
      @update:trajectoryLoop="updateTrajectoryLoop" @impulse="triggerImpulse" @reset="reset" />
  </div>

  <StatusDialogs :state="state" :extra-error-message="extra_error_message"
    :url-param-error-message="urlParamErrorMessage" :is-mobile="isMobile" :is-small-screen="isSmallScreen"
    @clearWarning="clearUrlWarning" />

  <Notice />

  <div class="help-button-container" v-if="!isMobile || isPanelCollapsed">
    <v-btn @click="showHelpDialog = true" icon size="small" variant="text" class="help-btn" title="Help Button (?)">
      <v-icon color="white">mdi-help</v-icon>
    </v-btn>
  </div>

  <HelpDialog v-model="showHelpDialog" :is-mobile="isMobile" @toggleHelp="() => (showHelpDialog = !showHelpDialog)"
    @toggleUI="toggleUIVisibility" @reset="reset" @navigateScene="navigateScene" @navigatePolicy="navigatePolicy" />
</template>

<script setup>
import { onMounted, onBeforeUnmount, watch, ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import ControlPanel from '@/views/demo/components/ControlPanel.vue'
import StatusDialogs from '@/views/demo/components/StatusDialogs.vue'
import HelpDialog from '@/views/demo/components/HelpDialog.vue'
import Notice from '@/views/demo/components/Notice.vue'
import { useConfig } from '@/views/demo/composables/useConfig'
import { useRuntime } from '@/views/demo/composables/useRuntime'
import { useScenePolicy } from '@/views/demo/composables/useScenePolicy'
import { useUrlSync } from '@/views/demo/composables/useUrlSync'
import { useTransition } from '@/views/demo/composables/useTransition'
import { useResponsive } from '@/views/demo/composables/useResponsive'
import { createShortcuts } from '@/utils/shortcuts.js'

const props = defineProps({ configPath: { type: String, default: './config.json' } })

const { isTransitioning, transitionMessage, withTransition } = useTransition()
const { isMobile, isSmallScreen, isPanelCollapsed, togglePanel } = useResponsive()
const rt = useRuntime()
const {
  state,
  extra_error_message,
  use_setpoint,
  command_vel_x,
  compliant_mode,
  facet_kp,
  trajectoryPlaybackState,
  trajectoryLoop,
  initRuntime,
  onTaskChange,
  onPolicyChange,
  updateFacetKp,
  updateUseSetpoint,
  updateCommandVelX,
  updateCompliantMode,
  triggerImpulse,
  playTrajectory,
  stopTrajectory,
  resetTrajectory,
  updateTrajectoryLoop,
  reset,
  dispose,
} = rt

const conf = useConfig(props.configPath)
const { config: appConfig, task, policy, taskItems, policyItems, selectedTask, selectedPolicy, urlParamErrorMessage, resolveDefaultPolicy, loadConfig } = conf
const projectName = computed(() => appConfig.value?.project_name)
const projectLink = computed(() => appConfig.value?.project_link)

const { routeItems, goToRoute, sync } = useUrlSync({
  router: useRouter(),
  route: useRoute(),
  getSceneName: () => selectedTask.value?.name || null,
  getPolicyName: () => selectedPolicy.value?.name || null,
})

const { selectTask, selectPolicy, navigateScene, navigatePolicy } = useScenePolicy({
  config: appConfig,
  task,
  policy,
  resolveDefaultPolicy,
  onTaskChange,
  onPolicyChange,
  withTransition,
})

const showHelpDialog = ref(false)
let shortcuts = null

function toggleUIVisibility() {
  document.body.classList.toggle('interactive-mode')
}

async function onSelectTask(id) { await selectTask(id) }
async function onSelectPolicy(id) { await selectPolicy(id) }

function onUpdateUseSetpoint(v) {
  use_setpoint.value = v
  updateUseSetpoint()
}

function onUpdateCommandVelX(v) {
  command_vel_x.value = v
  updateCommandVelX()
}

function onUpdateFacetKp(v) {
  facet_kp.value = v
  updateFacetKp()
}

function onUpdateCompliantMode(v) {
  compliant_mode.value = v
  updateCompliantMode()
}

function clearUrlWarning() { urlParamErrorMessage.value = '' }

watch([selectedTask, selectedPolicy], () => sync())

onMounted(async () => {
  await loadConfig()
  if (selectedTask.value) {
    const initialTask = selectedTask.value
    const initialPolicy = selectedPolicy.value
    await initRuntime(initialTask, initialPolicy)
  }

  const urlParams = new URLSearchParams(window.location.hash.split('?')[1])
  if (urlParams.get('hidectrl') === '1') {
    document.body.classList.add('interactive-mode')
  } else if (urlParams.get('hidectrl') === '0') {
    document.body.classList.remove('interactive-mode')
  }

  shortcuts = createShortcuts({
    onReset: () => reset(),
    onToggleUI: () => toggleUIVisibility(),
    onNavigateScene: (d) => navigateScene(d),
    onNavigatePolicy: (d) => navigatePolicy(d),
    getHelpVisible: () => showHelpDialog.value,
    setHelpVisible: (v) => { showHelpDialog.value = v },
  })
})

onBeforeUnmount(() => {
  dispose()
  if (shortcuts && typeof shortcuts.detach === 'function') shortcuts.detach()
  document.body.classList.remove('interactive-mode')
})
</script>

<style scoped>
.mujoco-container {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100vh;
  z-index: 1;
}

.control-panel {
  position: fixed;
  top: 20px;
  right: 20px;
  width: 260px;
  z-index: 1000;
}

@media (max-width: 768px) {
  .control-panel {
    top: auto;
    bottom: 0;
    left: 0;
    right: 0;
    width: 100%;
    padding: 6px 10px 10px;
  }

  .panel-toggle {
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    bottom: calc(100% + 8px);
    background: primary !important;
    border: 1px solid var(--ui-border) !important;
    box-shadow: var(--ui-shadow) !important;
  }

  .panel-collapsed .control-card {
    display: none;
  }
}

.overlay-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.transition-text {
  color: white;
  font-size: 0.95rem;
  letter-spacing: 0.01em;
}

.help-button-container {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 1001;
  transition: bottom 0.2s ease;
}

.help-btn {
  background: transparent !important;
  transition: opacity 0.2s ease;
  opacity: 0.7;
}

.help-btn:hover {
  opacity: 1;
}
</style>

<style>
body.interactive-mode .control-panel {
  display: none !important;
}

body.interactive-mode .transition-overlay,
body.interactive-mode .notice-container,
body.interactive-mode .help-button-container {
  display: block !important;
}
</style>
