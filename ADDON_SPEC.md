# Sionna RT GUI — Addon specification (API v1)

The GUI can be extended with addons, following the same model as Blender
addons: a small Python package with an `ADDON_INFO` dict and a
`register()` / `unregister()` pair. Addons interact with the application
exclusively through the `AddonAPI` object passed to `register()`.

## Addon layout

An addon is a Python package (a directory with an `__init__.py`):

```python
# <addon>/__init__.py

ADDON_INFO = {
    "name": "My addon",           # required
    "version": "0.1.0",           # recommended
    "author": "Your Name",        # recommended
    "description": "One line.",   # recommended, shown in the Addons panel
    "min_gui_version": "0.1",     # optional; addon refuses to load on older GUIs
}


def register(api):
    """Called when the addon is enabled. Add panels, keep a reference to api."""


def unregister(api):
    """Called when the addon is disabled. Undo everything register() did."""
```

## Discovery and installation

Addons are discovered from two channels:

1. **User addons directory**: `~/.sionna-rt-gui/addons/<name>/`.
   The "Install from zip" field in the *Addons* panel extracts a zip archive
   here. Distribute addons as their own repositories with zip releases; do not
   vendor them into this repository.
2. **pip entry points**: packages installed in the same environment can expose
   an addon module via:

   ```toml
   [project.entry-points."sionna_rt_gui.addons"]
   my_addon = "my_package.addon"
   ```

The user directory takes precedence when both channels provide the same name.
Enabled/disabled choices are persisted to `~/.sionna-rt-gui/addons.yaml`;
newly discovered addons are enabled by default. The `addons.enabled` mapping
in the GUI config file overrides the persisted state.

## The `AddonAPI` (v1)

The object passed to `register()` / `unregister()`. This is the **only**
stable surface; `api.version` is `(1, 0)`.

| Member | Contract |
| --- | --- |
| `api.add_panel(title, draw_fn)` | Adds an ImGui panel drawn every frame. `draw_fn(psim)` receives the `polyscope.imgui` module and must only issue ImGui calls. Panels are removed automatically when the addon is disabled. Closing a panel window disables the addon. |
| `api.load_scene(path)` | Requests loading a Sionna RT scene (path to an `.xml` file). Deferred to the start of the next frame, so it is safe to call from `run_async` completion callbacks. |
| `api.get_scene()` | Returns the current `sionna.rt.Scene` (or `None`). |
| `api.notify(message)` | Shows a transient in-app notification. Safe to call from any thread. |
| `api.run_async(fn, on_done=None)` | Runs `fn()` on a background thread. `on_done(result)` is then called on the main thread. `fn` must not touch the GUI/Polyscope; do that in `on_done`. |
| `api.gui` | The `SionnaRtGui` instance. Escape hatch, explicitly **unstable**: anything reached through it may change without notice in any release. |

## Error handling

Every addon callback (`register`, `unregister`, panel `draw_fn`, `on_done`)
is wrapped in try/except by the GUI. An exception notifies the user and
**disables the addon**; it never crashes the application. An exception inside
a `run_async` background function only notifies (background failures such as
network timeouts are expected and should not take the addon down);
`on_done` is not called in that case.

Auto-disabling after an error is not persisted: the addon will be tried
again on the next application start.

## Hard rules

Addons that do not follow these rules will not be listed or recommended.

1. **100% in-app UI.** All addon UI goes through `api.add_panel` and ImGui
   popups (`psim.OpenPopup` / `psim.BeginPopupModal`). **No web servers, no
   browser launches, no external windows** of any kind.
2. **Never block the frame loop.** Panel draw functions run every frame and
   must return immediately. Any long work — network requests, file
   conversion, heavy computation — goes through `api.run_async`.
3. **No secrets in repositories.** Addons that need API keys must read them
   from configuration outside any repository (environment variables or a
   keyring), never from tracked files. Prefer keyless services.
4. **Clean unregister.** Disabling an addon must remove everything it added.
   The GUI removes the addon's panels automatically, but any other state the
   addon created (threads, files, scene modifications) is the addon's
   responsibility to clean up in `unregister()`.
5. **Network access is allowed** within the rules above, and only from
   `api.run_async` background functions.

## Minimal example

```python
ADDON_INFO = {"name": "Hello", "version": "0.1", "min_gui_version": "0.1"}

_api = None


def _draw(psim):
    psim.Text("Hello from an addon!")
    if psim.Button("Notify"):
        _api.notify("Hello!")


def register(api):
    global _api
    _api = api
    api.add_panel("Hello", _draw)


def unregister(api):
    pass  # Panels are removed automatically.
```
