import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MantineProvider } from '@mantine/core';
import ControlPanel from './ControlPanel';
import { theme } from './AppTheme';
import { LoadingProvider, useLoading } from './contexts/LoadingContext';
import { Loader } from './components/Loader';
import { signalReady, signalError } from './core/utils/readySignal';
import { createEngine } from './engine';
import type { EnginePlugins, MjswanEngine, MjswanEngineState, SceneInput } from './engine';
import { parseManifest, sanitizeName, type Catalog, type ByteSource } from './manifest';
import './App.css';

const PANEL_QUERY_PARAM = 'panel';
const REF_QUERY_PARAM = 'ref';

function paramFlag(param: string): boolean {
  return new URLSearchParams(window.location.search).get(param) !== '0';
}

/** Lazy fetch of a build-relative asset path against the SPA base URL. */
function makeByteSource(base: string): ByteSource {
  return (relPath) => async () => {
    const url = `${base}${relPath}`.replace(/([^:])\/{2,}/g, '$1/');
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch ${url}: ${res.status}`);
    }
    return res.arrayBuffer();
  };
}

async function loadCatalog(base: string): Promise<Catalog> {
  const params = new URLSearchParams(window.location.search);
  const override = params.get('config');
  const configUrl = override
    ? new URL(override, window.location.href).toString()
    : `${base}assets/config.json`.replace(/([^:])\/{2,}/g, '$1/');
  const res = await fetch(configUrl, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${configUrl}: ${res.status}`);
  }
  return parseManifest(await res.text(), makeByteSource(base));
}

function pickByName<T extends { name: string }>(
  items: T[],
  query: string | null,
  fallback: T | undefined,
): T | undefined {
  if (!query) return fallback;
  const normalized = query.trim().toLowerCase();
  return (
    items.find((i) => i.name.toLowerCase() === normalized) ??
    items.find((i) => sanitizeName(i.name) === normalized) ??
    fallback
  );
}

function AppContent() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sceneName, setSceneName] = useState<string | null>(null);
  const [policyName, setPolicyName] = useState<string | null>(null);
  const [motionName, setMotionName] = useState<string | null>(null);
  const [splatName, setSplatName] = useState<string | null>(null);
  const [showReference, setShowReference] = useState(() => paramFlag(REF_QUERY_PARAM));
  const [panelVisible, setPanelVisible] = useState(() => paramFlag(PANEL_QUERY_PARAM));
  const [engineState, setEngineState] = useState<MjswanEngineState | null>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const engineRef = useRef<MjswanEngine | null>(null);
  // Author custom-MDP terms (trusted contexts only). Passed to both scene and
  // policy inputs; the engine registers each kind at the relevant load (ADR §10).
  const pluginsRef = useRef<EnginePlugins | null>(null);
  const { showLoading, hideLoading, setLoadingMessage } = useLoading();

  const base = import.meta.env.BASE_URL || '/';

  const withPlugins = useCallback((input: SceneInput): SceneInput => {
    const plugins = pluginsRef.current;
    if (!plugins) return input;
    input.plugins = plugins;
    if (input.policy) input.policy.plugins = plugins;
    return input;
  }, []);

  // ── load the catalog + pick the initial selection from the URL query ──────
  useEffect(() => {
    showLoading('Loading…');
    loadCatalog(base)
      .then(async (cat) => {
        // Trusted-only: load the author plugin ESM before the first scene so the
        // engine has the custom terms at registration time (ADR 0004 §10).
        if (cat.pluginsPath) {
          const url = `${base}${cat.pluginsPath}`.replace(/([^:])\/{2,}/g, '$1/');
          pluginsRef.current = (await import(/* @vite-ignore */ url)) as EnginePlugins;
        }
        const params = new URLSearchParams(window.location.search);
        const scene = pickByName(cat.scenes, params.get('scene'), cat.scenes[0]);
        const policy = scene
          ? pickByName(scene.policies, params.get('policy'), scene.policies.find((p) => p.default) ?? scene.policies[0])
          : undefined;
        const motion = policy?.motions.find((m) => m.default) ?? policy?.motions[0];
        setCatalog(cat);
        setSceneName(scene?.name ?? null);
        setPolicyName(policy?.name ?? null);
        setMotionName(motion?.name ?? null);
        setSplatName(scene?.splats[0]?.name ?? null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
        hideLoading();
        signalError();
      });
  }, []);

  const sceneEntry = useMemo(
    () => catalog?.scenes.find((s) => s.name === sceneName) ?? null,
    [catalog, sceneName],
  );

  // ── create the engine once, load the initial scene, subscribe to state ────
  useEffect(() => {
    const container = containerRef.current;
    if (!catalog || !sceneEntry || !container) {
      return;
    }
    let disposed = false;
    (async () => {
      showLoading('Loading MuJoCo…');
      const engine = await createEngine(container, { multithreaded: __MUJOCO_MT__ });
      if (disposed) {
        engine.dispose();
        return;
      }
      engineRef.current = engine;
      engine.subscribe(setEngineState);
      try {
        setLoadingMessage('Loading scene…');
        const input = await sceneEntry.buildScene({ policy: policyName, splat: splatName });
        await engine.loadScene(withPlugins(input));
        if (motionName) await engine.setMotion(motionName);
        engine.setReferenceVisible(showReference);
        hideLoading();
        signalReady();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        hideLoading();
        signalError();
      }
    })();
    return () => {
      disposed = true;
      engineRef.current?.dispose();
      engineRef.current = null;
    };
    // Recreate only when the catalog itself changes; scene switches are verbs.
  }, [catalog]);

  // ── mirror engine loading state into the overlay ──────────────────────────
  useEffect(() => {
    if (!engineState) return;
    if (engineState.loading) {
      showLoading(engineState.loadingMessage ?? 'Loading…');
    } else if (!engineState.error) {
      hideLoading();
    }
    if (engineState.error) {
      setError(engineState.error.message);
      hideLoading();
      signalError();
    }
  }, [engineState, showLoading, hideLoading]);

  const syncUrl = useCallback(
    (next: { scene?: string | null; policy?: string | null; panel?: boolean; ref?: boolean }) => {
      const params = new URLSearchParams(window.location.search);
      const setOrDelete = (key: string, value: string | null | undefined, keep: boolean) =>
        value != null && keep ? params.set(key, value) : params.delete(key);
      setOrDelete('scene', next.scene ?? sceneName, true);
      setOrDelete('policy', next.policy ?? policyName, true);
      params.delete(PANEL_QUERY_PARAM);
      if (!(next.panel ?? panelVisible)) params.set(PANEL_QUERY_PARAM, '0');
      params.delete(REF_QUERY_PARAM);
      if (!(next.ref ?? showReference)) params.set(REF_QUERY_PARAM, '0');
      const search = params.toString();
      window.history.replaceState({}, '', `${window.location.pathname}${search ? `?${search}` : ''}${window.location.hash}`);
    },
    [sceneName, policyName, panelVisible, showReference],
  );

  const handleSceneChange = useCallback(async (value: string | null) => {
    const engine = engineRef.current;
    const scene = catalog?.scenes.find((s) => s.name === value);
    if (!engine || !scene) return;
    const policy = scene.policies.find((p) => p.default) ?? scene.policies[0];
    const motion = policy?.motions.find((m) => m.default) ?? policy?.motions[0];
    const splat = scene.splats[0]?.name ?? null;
    setSceneName(scene.name);
    setPolicyName(policy?.name ?? null);
    setMotionName(motion?.name ?? null);
    setSplatName(splat);
    syncUrl({ scene: scene.name, policy: policy?.name ?? null });
    showLoading(`Loading scene "${scene.name}"…`);
    try {
      await engine.loadScene(withPlugins(await scene.buildScene({ policy: policy?.name, splat })));
      if (motion) await engine.setMotion(motion.name);
      engine.setReferenceVisible(showReference);
    } catch (err) {
      console.error('Failed to load scene:', err);
    } finally {
      hideLoading();
    }
  }, [catalog, showReference, syncUrl, showLoading, hideLoading]);

  const handlePolicyChange = useCallback(async (value: string | null) => {
    const engine = engineRef.current;
    const policy = sceneEntry?.policies.find((p) => p.name === value) ?? null;
    setPolicyName(value);
    const motion = policy?.motions.find((m) => m.default) ?? policy?.motions[0];
    setMotionName(motion?.name ?? null);
    syncUrl({ policy: value });
    if (!engine) return;
    showLoading(value ? `Loading policy "${value}"…` : 'Clearing policy…');
    try {
      const input = policy ? await policy.build() : null;
      if (input && pluginsRef.current) input.plugins = pluginsRef.current;
      await engine.setPolicy(input);
      if (motion) await engine.setMotion(motion.name);
      engine.setReferenceVisible(showReference);
    } catch (err) {
      console.error('Failed to load policy:', err);
    } finally {
      hideLoading();
    }
  }, [sceneEntry, showReference, syncUrl, showLoading, hideLoading]);

  const handleMotionChange = useCallback(async (value: string | null) => {
    const engine = engineRef.current;
    const previous = motionName;
    setMotionName(value);
    if (!engine) return;
    showLoading(value === null ? 'Clearing motion…' : `Loading motion "${value}"…`);
    try {
      const accepted = await engine.setMotion(value);
      if (!accepted && value !== null) setMotionName(previous);
    } catch (err) {
      console.error('Failed to load motion:', err);
    } finally {
      hideLoading();
    }
  }, [motionName, showLoading, hideLoading]);

  const handleSplatChange = useCallback(async (value: string | null) => {
    const engine = engineRef.current;
    const splat = value === null ? null : sceneEntry?.splats.find((s) => s.name === value) ?? null;
    setSplatName(value);
    if (!engine || (value !== null && !splat)) return;
    showLoading(value === null ? 'Removing splat…' : `Loading splat "${value}"…`);
    try {
      await engine.setSplat(splat ? await splat.build() : null);
    } catch (err) {
      console.error('Failed to load splat:', err);
    } finally {
      hideLoading();
    }
  }, [sceneEntry, showLoading, hideLoading]);

  const handleShowReferenceChange = useCallback((value: boolean) => {
    setShowReference(value);
    engineRef.current?.setReferenceVisible(value);
    syncUrl({ ref: value });
  }, [syncUrl]);

  const handlePanelVisibleChange = useCallback((visible: boolean) => {
    setPanelVisible(visible);
    syncUrl({ panel: visible });
  }, [syncUrl]);

  const options = useCallback(
    <T extends { name: string }>(items: T[]) => items.map((i) => ({ value: i.name, label: i.name })),
    [],
  );

  if (error) {
    return (
      <MantineProvider theme={theme} defaultColorScheme="auto">
        <div className="app">
          <div className="hud hud-error">
            <h1 className="hud-title">mjswan</h1>
            <p className="hud-message">{error}</p>
          </div>
        </div>
      </MantineProvider>
    );
  }

  return (
    <MantineProvider theme={theme} defaultColorScheme="auto">
      <div className="app">
        <Loader />
        {catalog && sceneEntry && (
          <ControlPanel
            visible={panelVisible}
            onVisibleChange={handlePanelVisibleChange}
            projects={[{ value: catalog.name, label: catalog.name }]}
            projectValue={catalog.name}
            projectLabel={catalog.name}
            onProjectChange={() => {}}
            scenes={options(catalog.scenes)}
            sceneValue={sceneName}
            onSceneChange={handleSceneChange}
            splats={options(sceneEntry.splats)}
            splatSection={sceneEntry.splatSection}
            splatValue={splatName}
            onSplatChange={handleSplatChange}
            splatConfig={(() => {
              const s = sceneEntry.splats.find((x) => x.name === splatName);
              return s ? { name: s.name, control: s.control, ...s.transform } : null;
            })()}
            onCalibrateSplat={(scale, xOffset, yOffset, zOffset, roll, pitch, yaw) =>
              engineRef.current?.calibrateSplat({ scale, xOffset, yOffset, zOffset, roll, pitch, yaw })
            }
            policies={options(sceneEntry.policies)}
            policyValue={policyName}
            onPolicyChange={handlePolicyChange}
            motions={options(sceneEntry.policies.find((p) => p.name === policyName)?.motions ?? [])}
            motionValue={motionName}
            onMotionChange={handleMotionChange}
            showReferenceMotion={showReference}
            onShowReferenceMotionChange={handleShowReferenceChange}
            commandsEnabled={!!policyName}
            commands={engineState?.commands ? [...engineState.commands] : []}
            commandValues={engineState?.commandValues ?? {}}
            onCommandChange={(id, value) => engineRef.current?.commands.set(id, value)}
            onReset={() => engineRef.current?.reset()}
          />
        )}
        <div ref={containerRef} className="viewer" />
      </div>
    </MantineProvider>
  );
}

function App() {
  return (
    <LoadingProvider>
      <AppContent />
    </LoadingProvider>
  );
}

export default App;
