# Muwanx Examples

This directory contains example applications demonstrating different usage patterns of the Muwanx package.

## Directory Structure

```
examples/
├── declarative/          # Pattern A: Declarative API (load from config)
│   └── index.html
├── imperative/          # Pattern B: Imperative API (programmatic)
│   └── index.html
├── assets/              # Shared assets (scenes, policies, configs)
│   ├── config.json      # Main demo configuration
│   ├── scene/           # MuJoCo scene files
│   └── policy/          # Policy configurations and ONNX models
├── App.vue              # Vue-based demo app
├── main.ts              # Vue demo entry point
└── router.ts            # Vue router configuration
```

## Usage Patterns

### Pattern A: Declarative (Load from Configuration)

The declarative pattern loads all configuration from a JSON file. This is ideal for:
- Static configurations
- Quick prototyping
- Configuration-driven applications

**Example:** `examples/declarative/index.html`

```typescript
import { MwxViewer } from 'muwanx';

const viewer = new MwxViewer('#container');
await viewer.loadConfig('./config.json');
```

**Configuration structure:**
```json
{
  "projects": {
    "project_name": "My Project",
    "scenes": [
      {
        "id": "scene1",
        "name": "My Scene",
        "model_xml": "./assets/scene.xml",
        "asset_meta": "./assets/metadata.json",
        "policies": [
          {
            "id": "policy1",
            "name": "My Policy",
            "path": "./assets/policy.json"
          }
        ]
      }
    ]
  }
}
```

### Pattern B: Imperative (Programmatic Construction)

The imperative pattern builds the viewer programmatically using method chaining. This is ideal for:
- Dynamic configurations
- Runtime-generated content
- Complex conditional logic

**Example:** `examples/imperative/index.html`

```typescript
import { MwxViewer } from 'muwanx';

const viewer = new MwxViewer('#container');

const project = viewer.addProject({
  project_name: "My Project"
});

const scene = project.addScene({
  id: "scene1",
  name: "My Scene",
  model_xml: "./assets/scene.xml"
});

const policy = scene.addPolicy({
  id: "policy1",
  name: "My Policy",
  path: "./assets/policy.json"
});

await viewer.initialize();
await viewer.selectScene('scene1');
await viewer.selectPolicy('policy1');
```

## Running the Examples

### Option 1: Development Server (Vite)

From the root directory:

```bash
npm install
npm run dev
```

Then navigate to:
- Declarative example: http://localhost:3000/declarative/
- Imperative example: http://localhost:3000/imperative/
- Vue demo: http://localhost:3000/

### Option 2: Build and Serve

```bash
# Build the demo
npm run build:demo

# Serve the built files
npm run preview
```

### Option 3: Library Mode

To use Muwanx as a library in your own project:

```bash
# Build the library
npm run build:lib

# This generates:
# - dist/muwanx.es.js (ES module)
# - dist/muwanx.umd.js (UMD)
# - dist/index.d.ts (TypeScript declarations)
```

## Available Assets

The `assets/` directory contains pre-configured scenes and policies:

### Scenes
- **Unitree Go2**: Quadruped robot (`scene/unitree_go2/`)
- **Unitree Go1**: Quadruped robot (`scene/unitree_go1/`)
- **Unitree G1**: Humanoid robot (`scene/unitree_g1/`)

### Policies
- **Vanilla Locomotion**: Basic walking policy
- **Rough Terrain**: Policy for uneven surfaces
- **Stairs**: Policy for climbing stairs

## API Reference

### MwxViewer

Main viewer class that manages projects, scenes, and policies.

**Constructor:**
```typescript
new MwxViewer(container?: HTMLElement | string)
```

**Methods:**
- `loadConfig(config: ViewerConfig | string): Promise<void>` - Load from config
- `addProject(config: ProjectConfig): Project` - Add project programmatically
- `initialize(): Promise<void>` - Initialize MuJoCo runtime
- `selectProject(projectId: string): Promise<void>` - Switch project
- `selectScene(sceneId: string): Promise<void>` - Load scene
- `selectPolicy(policyId: string): Promise<void>` - Load policy
- `play(): void` - Start simulation
- `pause(): void` - Pause simulation
- `reset(): Promise<void>` - Reset simulation
- `updateParams(params: Partial<RuntimeParams>): void` - Update runtime parameters
- `on(event: string, callback: Function): void` - Add event listener
- `off(event: string, callback: Function): void` - Remove event listener

**Events:**
- `project-changed` - Fired when project changes
- `scene-changed` - Fired when scene loads
- `policy-changed` - Fired when policy loads
- `params-changed` - Fired when parameters update
- `error` - Fired on errors

### Project Builder

```typescript
const project = viewer.addProject(config);
project.addScene(sceneConfig);
project.setMetadata({ name, link });
project.setDefaultScene(sceneId);
```

### Scene Builder

```typescript
const scene = project.addScene(config);
scene.addPolicy(policyConfig);
scene.setMetadata(metadata);
scene.setCamera(cameraConfig);
scene.setBackgroundColor(color);
scene.setDefaultPolicy(policyId);
```

### Policy Builder

```typescript
const policy = scene.addPolicy(config);
policy.setONNX(onnxConfig);
policy.setObservationConfig(obsConfig);
policy.setPDParams({ stiffness, damping });
policy.setActionScale(scale);
policy.setUIControls(['setpoint', 'stiffness']);
```

## TypeScript Support

The package includes full TypeScript type definitions:

```typescript
import type {
  ViewerConfig,
  ProjectConfig,
  SceneConfig,
  PolicyConfig,
  RuntimeParams,
} from 'muwanx';
```

## Advanced Usage

### Custom Observation Components

```typescript
import { ObservationManager } from 'muwanx';

// Define custom observation configuration
const obsConfig = {
  policy: [
    { name: 'ProjectedGravity', history_steps: 3 },
    { name: 'JointPositions', joint_names: 'isaac', history_steps: 3 },
    { name: 'JointVelocities', joint_names: 'isaac', history_steps: 1 },
  ]
};
```

### Custom Action Managers

```typescript
import { IsaacActionManager } from 'muwanx';

const actionManager = new IsaacActionManager();
// Configure action manager...
```

### Direct Runtime Access

For advanced use cases, access the underlying MujocoRuntime:

```typescript
const runtime = viewer.getRuntime();
// Access MuJoCo model, data, and low-level APIs
```

## Contributing

To add new examples:

1. Create a new directory under `examples/`
2. Add an `index.html` or entry point
3. Document the example in this README
4. Ensure it works with both `npm run dev` and `npm run build`

## License

Apache-2.0
