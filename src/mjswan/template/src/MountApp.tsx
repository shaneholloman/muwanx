import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MantineProvider } from '@mantine/core';
import MjswanViewer from './components/MjswanViewer';
import ControlPanel from './ControlPanel';
import type { mjswanRuntime } from './core/engine/runtime';
import type { SplatConfig } from './core/scene/splat';
import { theme } from './AppTheme';
import { LoadingProvider, useLoading } from './contexts/LoadingContext';
import { Loader } from './components/Loader';
import { signalReady, signalError } from './core/utils/readySignal';
import {
  type AppConfig,
  type ProjectConfig,
  type SceneConfig,
  pickMotion,
  pickPolicy,
  pickScene,
  resolveProjectAsset,
  resolveScenePath,
} from './core/appConfig';
import './App.css';

interface MountAppProps {
  config: AppConfig;
  /** Absolute URL of the config.json directory; all assets resolve against it. */
  baseUrl: string;
  onReady?: () => void;
  onError?: (error: Error) => void;
  /** Surfaces the runtime to the caller (used by `mount()` for `captureThumbnail`). */
  onRuntimeReady?: (runtime: mjswanRuntime) => void;
}

function MountAppContent({ config, baseUrl, onReady, onError, onRuntimeReady }: MountAppProps) {
  const initialProject = useMemo(
    () => config.projects.find((p) => p.id === null) ?? config.projects[0] ?? null,
    [config]
  );

  const [currentProject, setCurrentProject] = useState<ProjectConfig | null>(initialProject);
  const [currentScene, setCurrentScene] = useState<SceneConfig | null>(
    initialProject ? pickScene(initialProject, null) : null
  );
  const [selectedPolicy, setSelectedPolicy] = useState<string | null>(() => {
    const scene = initialProject ? pickScene(initialProject, null) : null;
    return scene ? pickPolicy(scene, null) : null;
  });
  const [selectedMotion, setSelectedMotion] = useState<string | null>(null);
  const [showReferenceMotion, setShowReferenceMotion] = useState(false);
  const [selectedSplat, setSelectedSplat] = useState<string | null>(null);
  const [customSplatUrl, setCustomSplatUrl] = useState<string | null>(null);
  const [panelVisible, setPanelVisible] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const runtimeRef = useRef<mjswanRuntime | null>(null);
  const { showLoading, hideLoading, setLoadingMessage } = useLoading();

  useEffect(() => {
    showLoading('Loading…');
  }, [showLoading]);

  const scenePath = useMemo(() => {
    if (!currentProject || !currentScene) {
      return null;
    }
    return resolveScenePath(currentProject, currentScene);
  }, [currentProject, currentScene]);

  const selectedPolicyConfig = useMemo(() => {
    if (!currentScene || !selectedPolicy) {
      return null;
    }
    return currentScene.policies.find((policy) => policy.name === selectedPolicy) ?? null;
  }, [currentScene, selectedPolicy]);

  const policyConfigPath = useMemo(() => {
    if (!currentProject || !selectedPolicyConfig?.config) {
      return null;
    }
    return resolveProjectAsset(currentProject, selectedPolicyConfig.config);
  }, [currentProject, selectedPolicyConfig]);

  const motionOptions = useMemo(() => {
    if (!selectedPolicyConfig?.motions?.length) {
      return [] as { value: string; label: string }[];
    }
    return selectedPolicyConfig.motions.map((motion) => ({ value: motion.name, label: motion.name }));
  }, [selectedPolicyConfig]);

  const resolvedSplats = useMemo(() => {
    if (!currentProject || !currentScene?.splats?.length) return [] as SplatConfig[];
    return currentScene.splats.map((splat) =>
      splat.path ? { ...splat, url: resolveProjectAsset(currentProject, splat.path) } : splat
    );
  }, [currentProject, currentScene?.splats]);

  const resolvedSplatConfig = useMemo(() => {
    if (!selectedSplat) return customSplatUrl ? { name: 'Custom', url: customSplatUrl } : null;
    return resolvedSplats.find((s) => s.name === selectedSplat) ?? null;
  }, [resolvedSplats, selectedSplat, customSplatUrl]);

  const projectOptions = useMemo(
    () => config.projects.map((p) => ({ value: p.id ?? 'main', label: p.name || (p.id ?? 'Main') })),
    [config]
  );
  const sceneOptions = useMemo(
    () => (currentProject ? currentProject.scenes.map((s) => ({ value: s.name, label: s.name })) : []),
    [currentProject]
  );
  const policyOptions = useMemo(
    () => (currentScene ? currentScene.policies.map((p) => ({ value: p.name, label: p.name })) : []),
    [currentScene]
  );
  const splatOptions = useMemo(
    () => (currentScene?.splats?.length ? currentScene.splats.map((s) => ({ value: s.name, label: s.name })) : []),
    [currentScene?.splats]
  );

  // Reset splat selection when switching scenes.
  useEffect(() => {
    const firstSplat = currentScene?.splats?.[0];
    setSelectedSplat(firstSplat ? firstSplat.name : null);
    setCustomSplatUrl(null);
  }, [currentScene]);

  useEffect(() => {
    setSelectedMotion(pickMotion(selectedPolicyConfig, null));
    setShowReferenceMotion(Boolean(selectedPolicyConfig?.motions?.length));
  }, [selectedPolicyConfig]);

  const handleViewerError = useCallback(
    (err: Error) => {
      setError(err.message);
      hideLoading();
      signalError();
      onError?.(err);
    },
    [hideLoading, onError]
  );

  const handleViewerReady = useCallback(() => {
    hideLoading();
    signalReady();
    onReady?.();
  }, [hideLoading, onReady]);

  const handleViewerStatus = useCallback(
    (status: string) => {
      if (status === 'Running simulation' || status === 'Failed to load scene') {
        return;
      }
      setLoadingMessage(status);
    },
    [setLoadingMessage]
  );

  const handleRuntimeReady = useCallback(
    (runtime: mjswanRuntime) => {
      runtimeRef.current = runtime;
      onRuntimeReady?.(runtime);
    },
    [onRuntimeReady]
  );

  const handleProjectChange = useCallback(
    (value: string | null) => {
      if (!value) return;
      const normalized = value === 'main' ? null : value;
      const project = config.projects.find((p) => (p.id ?? 'main') === (normalized ?? 'main'));
      if (!project) return;
      const nextScene = pickScene(project, null);
      showLoading(nextScene ? `Loading scene "${nextScene.name}"…` : 'Loading…');
      setCurrentProject(project);
      setCurrentScene(nextScene);
      setSelectedPolicy(nextScene ? pickPolicy(nextScene, null) : null);
    },
    [config, showLoading]
  );

  const handleSceneChange = useCallback(
    (value: string | null) => {
      if (!currentProject || !value) return;
      const scene = currentProject.scenes.find((s) => s.name === value);
      if (!scene) return;
      showLoading(`Loading scene "${scene.name}"…`);
      setCurrentScene(scene);
      setSelectedPolicy(pickPolicy(scene, null));
    },
    [currentProject, showLoading]
  );

  const handlePolicyChange = useCallback(
    (value: string | null) => {
      if (value !== selectedPolicy) {
        showLoading(value ? `Loading policy "${value}"…` : 'Loading policy…');
      }
      setSelectedPolicy(value);
    },
    [selectedPolicy, showLoading]
  );

  const handleMotionChange = useCallback(
    async (value: string | null) => {
      const previousMotion = selectedMotion;
      setSelectedMotion(value);
      const runtime = runtimeRef.current;
      if (!runtime) return;
      showLoading(value === null ? 'Clearing motion…' : `Loading motion "${value}"…`);
      try {
        const accepted = await runtime.setSelectedMotion(value);
        if (accepted === false && value !== null) {
          setSelectedMotion(runtime.getSelectedMotionName() ?? previousMotion);
        }
      } catch (e) {
        console.error('Failed to load motion:', e);
      } finally {
        hideLoading();
      }
    },
    [selectedMotion, showLoading, hideLoading]
  );

  const handleShowReferenceChange = useCallback((value: boolean) => {
    setShowReferenceMotion(value);
    runtimeRef.current?.setReferenceVisible(value);
  }, []);

  const handleSplatChange = useCallback(
    async (value: string | null) => {
      setSelectedSplat(value);
      setCustomSplatUrl(null);
      const runtime = runtimeRef.current;
      if (!runtime) return;
      const splat = value === null ? null : (resolvedSplats.find((s) => s.name === value) ?? null);
      if (value !== null && !splat) return;
      showLoading(value === null ? 'Removing splat…' : `Loading splat "${value}"…`);
      try {
        await runtime.setSplat(splat);
      } catch (e) {
        console.error('Failed to load splat:', e);
      } finally {
        hideLoading();
      }
    },
    [resolvedSplats, showLoading, hideLoading]
  );

  const handleSplatUrlLoad = useCallback(
    async (url: string): Promise<boolean> => {
      try {
        const res = await fetch(url, { method: 'HEAD' });
        if (!res.ok) return false;
      } catch {
        return false;
      }
      const runtime = runtimeRef.current;
      if (!runtime) return false;
      showLoading('Loading splat "Custom"…');
      try {
        await runtime.setSplat({ name: 'Custom', url });
        setCustomSplatUrl(url);
        return true;
      } catch (e) {
        console.error('Failed to load custom splat:', e);
        return false;
      } finally {
        hideLoading();
      }
    },
    [showLoading, hideLoading]
  );

  const handleCalibrateSplat = useCallback(
    (scale: number, x: number, y: number, z: number, roll: number, pitch: number, yaw: number) => {
      const splat = resolvedSplatConfig ?? (customSplatUrl ? { name: 'Custom', url: customSplatUrl } : null);
      if (splat) {
        runtimeRef.current?.calibrateSplat({ ...splat, scale, xOffset: x, yOffset: y, zOffset: z, roll, pitch, yaw });
      }
    },
    [resolvedSplatConfig, customSplatUrl]
  );

  if (error) {
    return (
      <div className="hud hud-error">
        <h1 className="hud-title">mjswan</h1>
        <p className="hud-message">{error}</p>
      </div>
    );
  }

  if (!currentProject || !currentScene || !scenePath) {
    return null;
  }

  return (
    <>
      <Loader />
      <ControlPanel
        visible={panelVisible}
        onVisibleChange={setPanelVisible}
        projects={projectOptions}
        projectValue={currentProject.id ?? 'main'}
        projectLabel={currentProject.name ?? 'mjswan'}
        onProjectChange={handleProjectChange}
        scenes={sceneOptions}
        sceneValue={currentScene.name}
        onSceneChange={handleSceneChange}
        splats={splatOptions}
        splatSection={currentScene.splatSection ?? false}
        splatValue={selectedSplat}
        onSplatChange={handleSplatChange}
        splatConfig={resolvedSplatConfig}
        onCalibrateSplat={handleCalibrateSplat}
        onSplatUrlLoad={handleSplatUrlLoad}
        policies={policyOptions}
        policyValue={selectedPolicy}
        onPolicyChange={handlePolicyChange}
        motions={motionOptions}
        motionValue={selectedMotion}
        onMotionChange={handleMotionChange}
        showReferenceMotion={showReferenceMotion}
        onShowReferenceMotionChange={handleShowReferenceChange}
        commandsEnabled={!!policyConfigPath}
      />
      <MjswanViewer
        scenePath={scenePath}
        baseUrl={baseUrl}
        policyConfigPath={policyConfigPath}
        splatConfig={resolvedSplatConfig}
        cameraConfig={currentScene.camera}
        eventsConfig={currentScene.events}
        terrainData={currentScene.terrainData}
        selectedMotion={selectedMotion}
        showReferenceMotion={showReferenceMotion}
        onError={handleViewerError}
        onReady={handleViewerReady}
        onStatusChange={handleViewerStatus}
        onRuntimeReady={handleRuntimeReady}
      />
    </>
  );
}

/**
 * Self-contained root rendered by the library `mount()` entry. Owns its own
 * Mantine + loading providers so it composes cleanly into any host page DOM.
 */
export function MountApp(props: MountAppProps) {
  return (
    <MantineProvider theme={theme} defaultColorScheme="auto">
      <LoadingProvider>
        <div className="app">
          <MountAppContent {...props} />
        </div>
      </LoadingProvider>
    </MantineProvider>
  );
}

export default MountApp;
