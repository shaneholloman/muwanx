"""Record an mjlab command term's viser GUI as an mjswan UI descriptor.

mjlab declares each command's viewer controls in one place —
``CommandTerm.create_gui``, calling viser's ``server.gui.add_*``. Running it
against a recording stand-in makes that the browser control panel's only
definition, instead of hand-copying slider ranges into ``CommandBinding(ui=...)``
and watching them drift. A stand-in rather than a real ``ViserServer``, which
would open a port and serve a page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Handle:
    """One recorded control, doubling as the handle ``create_gui`` keeps.

    mjlab stores handles to read ``.value`` later and its ``on_update`` callbacks
    assign ``.min``/``.max``; plain attributes cover both. Callbacks are never
    replayed — they only re-derive what the recording already holds.
    """

    kind: str
    label: str
    value: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None

    def on_update(self, fn: Any) -> Any:
        return fn

    def on_click(self, fn: Any) -> Any:
        return fn


@dataclass
class _Folder:
    """Context manager only. The title is dropped: mjlab uses
    ``name.capitalize()``, which the browser already derives from the term name
    (``ControlPanel.formatGroupName``)."""

    label: str

    def __enter__(self) -> _Folder:
        return self

    def __exit__(self, *_: Any) -> None:
        pass


@dataclass
class _GuiRecorder:
    """``server.gui``, recording in call order.

    Only the ``add_*`` mjlab's command terms use — an unknown one raises
    ``AttributeError`` rather than silently dropping a control the viewer shows.
    """

    recorded: list[_Handle] = field(default_factory=list)

    def _record(self, handle: _Handle) -> _Handle:
        self.recorded.append(handle)
        return handle

    def add_folder(self, label: str, **_: Any) -> _Folder:
        return _Folder(label)

    def add_checkbox(
        self, label: str, initial_value: bool = False, **_: Any
    ) -> _Handle:
        return self._record(_Handle("checkbox", label, value=bool(initial_value)))

    def add_slider(
        self,
        label: str,
        min: float | None = None,
        max: float | None = None,
        step: float | None = None,
        initial_value: float | None = None,
        **_: Any,
    ) -> _Handle:
        return self._record(
            _Handle("slider", label, value=initial_value, min=min, max=max, step=step)
        )

    def add_button(self, label: str, **_: Any) -> _Handle:
        return self._record(_Handle("button", label))


@dataclass
class _ServerRecorder:
    gui: _GuiRecorder = field(default_factory=_GuiRecorder)


def _slug(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def to_ui_descriptor(handles: list[_Handle]) -> dict[str, Any] | None:
    """Recorded controls -> a ``commands.<term>.ui`` descriptor, ``None`` if empty.

    mjlab identifies handles by variable reference and treats labels as display
    text, so all three conventions here are structural:

    - First checkbox takes the name ``enabled`` whatever its label, since
      ``OnnxCommand.isUiEnabled`` looks for exactly that.
    - A one-sided slider is a "Max <label>" companion rescaling the next axis's
      reach, never a command axis (mjlab keeps these out of ``_joystick_sliders``,
      but that list is a local). An axis straddles zero.
    - Order is the contract: mjlab and the browser both map axis sliders onto the
      command vector positionally.
    """
    inputs: list[dict[str, Any]] = []
    enable_name: str | None = None
    companion: dict[str, Any] | None = None

    for handle in handles:
        if handle.kind == "checkbox":
            name = "enabled" if enable_name is None else _slug(handle.label)
            enable_name = enable_name or name
            inputs.append(
                {
                    "type": "checkbox",
                    "name": name,
                    "label": handle.label,
                    "default": bool(handle.value),
                }
            )
        elif handle.kind == "slider":
            if handle.min is not None and handle.min >= 0:
                companion = {
                    "min": handle.min,
                    "max": handle.max,
                    "step": handle.step,
                    "default": handle.value,
                }
                continue
            entry: dict[str, Any] = {
                "type": "slider",
                "name": _slug(handle.label),
                "label": handle.label,
                "min": handle.min,
                "max": handle.max,
                "step": handle.step,
                "default": handle.value,
            }
            if enable_name is not None:
                entry["enabled_when"] = enable_name
            if companion is not None:
                # No `label`: the browser synthesizes `Max <label>`, matching mjlab.
                entry["adjustable_range"] = companion
                companion = None
            inputs.append(entry)
        elif handle.kind == "button":
            inputs.append(
                {"type": "button", "name": _slug(handle.label), "label": handle.label}
            )

    return {"inputs": inputs} if inputs else None


def record_gui(term: Any, name: str) -> dict[str, Any] | None:
    """A built term's ``create_gui`` recorded as a UI descriptor.

    ``None`` when it declares no controls: ``CommandTerm.create_gui`` is a
    base-class no-op, so every term has the method and an empty recording is the
    only signal that this one does not override it.
    """
    server = _ServerRecorder()
    term.create_gui(name, server, lambda: 0)
    return to_ui_descriptor(server.gui.recorded)


__all__ = ["record_gui", "to_ui_descriptor"]
