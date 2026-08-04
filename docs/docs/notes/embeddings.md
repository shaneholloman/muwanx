---
icon: octicons/screen-full-16
---

# Visualization Embeddings

Embed mjswan visualizations in web pages or Jupyter notebooks for interactive demonstrations and sharing.

## Web Page

After hosting your mjswan application as a static site (e.g., via GitHub Pages), you can embed mjswan visualizations directly into your web pages (e.g., research articles, blogs) using an iframe.

For example:

```html
<iframe
  src="https://ttktjmt.github.io/mjswan/"
  width="100%"
  height="600px"
  frameborder="0">
</iframe>
```

## Google Colab

You can also embed mjswan visualizations directly in Google Colab notebooks.

### Build the app

```python
import os

import mujoco
import mjswan

# Build your application
builder = mjswan.Builder()
project = builder.add_project(name="Demo")
project.add_scene(spec=mujoco.MjSpec.from_file("path/to/model.xml"), name="Scene")
app = builder.build()
```

### Serve in Colab

```python
import http.server
import socketserver
import threading
from google.colab import output

PORT = 8000
DIRECTORY = "dist"  # directory produced by builder.build()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Required for SharedArrayBuffer (used by MuJoCo WASM threading)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()


thread = threading.Thread(target=start_server, daemon=True)
thread.start()

# Display in Colab
output.serve_kernel_port_as_iframe(PORT, height=600)
```

➔ Check out the full example: [examples/colab/demo.ipynb](https://github.com/ttktjmt/mjswan/blob/main/examples/colab/demo.ipynb){:target="_blank"}
