import { computed, watch } from 'vue';
import type { RouteLocationNormalizedLoaded, Router } from 'vue-router';

export function useUrlSync(options: {
  router: Router;
  route: RouteLocationNormalizedLoaded;
  getSceneName: () => string | null;
  getPolicyName: () => string | null;
}) {
  const { router, route, getSceneName, getPolicyName } = options;

  function sync() {
    try {
      const sceneName = getSceneName();
      const policyName = getPolicyName();
      const currentQuery = { ...(route?.query || {}) } as Record<string, any>;
      const nextQuery: Record<string, any> = { ...currentQuery };

      if (sceneName) nextQuery.scene = sceneName; else delete nextQuery.scene;
      if (policyName) nextQuery.policy = policyName; else delete nextQuery.policy;

      const changed = Object.keys({ ...currentQuery, ...nextQuery }).some(k => currentQuery[k] !== nextQuery[k])
        || Object.keys(currentQuery).length !== Object.keys(nextQuery).length;
      if (changed) {
        router?.replace({ query: nextQuery });
      }
    } catch (e) {
      console.warn('Failed to sync URL with selection:', e);
    }
  }

  const routeItems = computed(() => {
    try {
      const routes = (router?.getRoutes?.() || []).filter(r => r.name && r.path);
      const seen = new Set<string>();
      const items: { name: any; path: string; title: string }[] = [];
      for (const r of routes) {
        const key = String(r.name);
        if (seen.has(key)) continue;
        seen.add(key);
        items.push({ name: r.name, path: r.path, title: key });
      }
      return items;
    } catch {
      return [] as { name: any; path: string; title: string }[];
    }
  });

  function goToRoute(r: { name: any; path: string }) {
    try {
      if (r?.path && r.name !== route?.name) {
        router.push({ path: r.path });
      }
    } catch (e) {
      console.warn('Failed to navigate route:', e);
    }
  }

  return { sync, routeItems, goToRoute };
}

