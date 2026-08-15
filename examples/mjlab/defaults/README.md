# mjlab Default Tasks

Source: https://github.com/mujocolab/mjlab

Demo: https://mjswan-mjlab.pages.dev/

Every MDP term here reaches the browser as a traced ONNX graph — no hand-written
TypeScript, so the build reports `uses_custom_js: false`. Two things still need
task-side Python, and neither is a browser implementation:

- `terminations/` injects the terrain generator's `limit_x`/`limit_y`/`half_x`/`half_y`
  into mjlab's own termination params, because those constants live on the generator
  rather than on the function.
- `commands/` registers what each mjlab command cfg traces as — a trace-friendly rewrite
  for `UniformVelocityCommandCfg`, a debug-vis marker for `LiftingCommandCfg`, and the
  RSI reset graph for `MotionCommandCfg` (whose clip lookup stays native).
