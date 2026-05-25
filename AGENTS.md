# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Psychrometric Chart Lab — a single-process PyQt6 + Matplotlib desktop GUI for HVAC psychrometric chart exploration. No backend services, databases, or network dependencies.

### Running the app

```bash
source .venv/bin/activate
DISPLAY=:1 python -m psychro_chart_app
```

The VM has an X server on `:1`. PyQt6 requires the following system libraries to load the xcb platform plugin: `libxcb-cursor0`, `libxkbcommon-x11-0`, `libxcb-icccm4`, `libxcb-keysyms1`, `libxcb-xkb1`, `libegl1`. These are installed by the update script.

### Running tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

### Linting

No linter is configured in the project. For ad-hoc checks:

```bash
source .venv/bin/activate
ruff check .
pyright
```

Note: pyright reports false-positive `reportOptionalMemberAccess` on PyQt6 methods (pre-existing, not actionable).

### Gotchas

- After installing new system libraries (e.g. via `apt`), run `sudo ldconfig` and start a **fresh** shell/process for the Qt xcb plugin to find them. Existing processes cache the library search path.
- The app exits immediately with `SystemExit` via `__main__.py` — to keep it running in background for testing, use `&` or launch via tmux.
