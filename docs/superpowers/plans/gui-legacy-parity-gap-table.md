# GUI Legacy-Shell Parity Gap Table

Source: `docs/superpowers/specs/2026-07-12-gui-legacy-parity-closure-design.md` §2.1.
Every legacy menu/toolbar/dialog action was traced to its real handler (not just its
menu declaration) in both `rheojax/gui/app/{menu_bar.py,toolbar.py,main_window.py}` and
`rheojax/gui/workspace/{window.py,library_rail.py}`.

| Feature | Legacy location | Workspace equivalent (Y/N) | Disposition | Notes |
|---|---|---|---|---|
| Delete Dataset | `app/menu_bar.py` Data menu, `main_window.py:_on_delete_dataset` | N | Implement | `LibraryRail`'s context menu only has "Preview…" (`library_rail.py:55-64`). Task 2/3. |
| Help menu (Docs/Tutorials/Shortcuts/About) | `app/menu_bar.py` Help menu, `main_window.py:3445-3492` | N | Implement | Trivial handlers (open URL, static dialogs); `workspace/window.py` builds only File/View menus. Task 4. |
| Undo/Redo | `app/menu_bar.py` Edit menu, `state/store.py` real undo/redo stack | N | Justified-drop (deferred to own future spec) | Real, working in legacy; workspace's plain-dataclass state has no history infrastructure — disproportionate to this spec's scope. |
| New Dataset | `app/menu_bar.py` Data menu | Y | Not a gap | Legacy's own handler is `self._on_import()`; `LibraryRail`'s "+ Import data…" button covers it 1:1. |
| Preferences | `app/menu_bar.py` Edit menu | Y | Not a gap | Same shared `PreferencesDialog`, already wired in `workspace/window.py:106`. |
| Theme (Light/Dark/Auto) | `app/menu_bar.py` View menu | Y | Not a gap | Already in `workspace/window.py:173-190`. |
| Log Panel toggle | `app/menu_bar.py` View menu | Y | Not a gap | Already in `workspace/window.py:164-171`. |
| Models/Transforms/Analysis/Pipeline quick-launch | `app/menu_bar.py` | Y (different paradigm) | Not a gap | These are shortcuts into functionality that's the primary step-based UI in the workspace shell. |
| Set Test Mode / Auto-detect Test Mode | `app/menu_bar.py` Data menu | Y | Not a gap | Test-mode detection happens automatically in the shared `DataService.load_file_multi()` path both shells use; confirmed by tracing `window._on_import_requested` → `_launch_import` → `ImportWorker.run()`. |
| Zoom In/Out/Reset | `app/menu_bar.py`/`toolbar.py` | N | Justified-drop | Dead code in legacy itself — no page implements `zoom_in`/`zoom_out`/`reset_zoom`; the handler's `hasattr()` guard is always false. |
| Recent Files | `app/menu_bar.py` File menu | N | Justified-drop | Legacy's own `_populate_recent_files()` is an explicit placeholder; always shows "No recent files." |
| Cut/Copy/Paste | `app/menu_bar.py` Edit menu | N | Justified-drop | Generic passthrough to whatever native Qt widget has focus; no rheojax-specific logic, already free via Ctrl+C/right-click on any text field. |
| Data Panel dock toggle | `app/menu_bar.py` View menu | N (architecture difference) | Justified-drop | Legacy toggles a closable dock; `LibraryRail` is an always-visible rail by design. |
| Tools menu (Python Console/JAX Profiler/Memory Monitor) | `app/menu_bar.py` Tools menu | N | Justified-drop | All three are `setEnabled(False)` "(coming soon)" stubs in legacy itself. |
