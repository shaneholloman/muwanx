export interface PolicyConfigItem {
  id: string;
  name: string;
  path?: string | null;
  type?: string | null; // e.g. 'trajectory'
  trajectory_path?: string | null;
  model_xml?: string | null; // allow policy override
  asset_meta?: string | null; // allow policy override
  ui_controls?: string[]; // e.g. ['setpoint', 'stiffness', 'trajectory_playback']
  show_setpoint?: boolean;
}

export interface TaskConfigItem {
  id: string;
  name: string;
  model_xml: string;
  asset_meta?: string | null;
  default_policy?: string | null;
  policies: PolicyConfigItem[];
}

export interface AppConfig {
  project_name?: string;
  project_link?: string;
  tasks: TaskConfigItem[];
}

