import type { AppConfig, TaskConfigItem, PolicyConfigItem } from '@/types/config';

/**
 * Resolves the default policy for a given task
 */
export function resolveDefaultPolicy(taskItem: TaskConfigItem | null): string | null {
  if (!taskItem) return null;

  if (taskItem.default_policy !== null && taskItem.default_policy !== undefined) {
    return taskItem.default_policy as string;
  }

  return taskItem.policies?.[0]?.id ?? null;
}

/**
 * Parses URL parameters from the hash part of the URL
 */
export function parseUrlParams(): { scene: string | null; policy: string | null } {
  const urlParams = new URLSearchParams(window.location.hash.split('?')[1]);
  return {
    scene: urlParams.get('scene'),
    policy: urlParams.get('policy'),
  };
}

/**
 * Finds a task by name (case-insensitive)
 */
export function findTaskByName(
  config: AppConfig,
  sceneName: string
): TaskConfigItem | null {
  return (config.tasks || []).find(t =>
    t.name.toLowerCase() === sceneName.toLowerCase()
  ) || null;
}

/**
 * Finds a policy by name within a task (case-insensitive)
 */
export function findPolicyByName(
  task: TaskConfigItem,
  policyName: string
): PolicyConfigItem | null {
  return (task.policies || []).find(p =>
    p.name.toLowerCase() === policyName.toLowerCase()
  ) || null;
}

/**
 * Loads configuration from a given path
 */
export async function loadConfig(configPath: string): Promise<AppConfig> {
  const response = await fetch(configPath);
  const json = await response.json();
  return json as AppConfig;
}
