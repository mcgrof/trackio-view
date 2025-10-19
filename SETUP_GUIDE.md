# TrackIO View - Setup Guide

This document explains the setup of trackio-view as a standalone PyPI package, following the feedback from the upstream TrackIO maintainers.

## Background

The trackio-view functionality was originally developed as part of the main TrackIO repository. The maintainers provided this feedback:

> "This is very cool @mcgrof but is a bit out of scope and will be hard for us to maintain. Trackio is designed to be pretty lightweight and extensible so I'd recommend instead creating a separate trackio-view pypi package which we could link to from our documentation."

This package is the result of extracting trackio-view into a standalone, separately maintained PyPI package.

## Package Structure

```
trackio-view/
├── trackio_view/              # Main package
│   ├── __init__.py           # Package initialization
│   ├── view.py               # Main terminal dashboard
│   ├── gpu_monitor.py        # GPU monitoring backend
│   ├── gpu_dashboard.py      # Simplified GPU dashboard
│   └── gpu_dashboard_gputop.py  # Full gputop-style monitoring
├── tests/                    # Test suite
│   ├── conftest.py          # Pytest fixtures
│   ├── __init__.py
│   └── test_cli_view.py     # Main tests
├── README.md                 # User documentation
├── CONTRIBUTING             # Developer guide (DCO)
├── LICENSE                   # MIT license
├── LICENSES/                 # SPDX license files
├── pyproject.toml           # Package configuration
├── MANIFEST.in              # Distribution manifest
└── .gitignore               # Git ignore rules
```

## What Was Moved

The following files were extracted from the trackio repository and adapted:

### Source Code (from trackio/):
- `view.py` → `trackio_view/view.py`
- `gpu_monitor.py` → `trackio_view/gpu_monitor.py`
- `gpu_dashboard.py` → `trackio_view/gpu_dashboard.py`
- `gpu_dashboard_gputop.py` → `trackio_view/gpu_dashboard_gputop.py`

### Tests (from tests/):
- `test_cli_view.py` → `tests/test_cli_view.py`

### Changes Made:
1. Updated all imports from `trackio.X` to `trackio_view.X`
2. Created standalone package configuration in `pyproject.toml`
3. Added dependency on `trackio>=0.1.0`
4. Created comprehensive README and documentation
5. Set up proper package structure with __init__.py
6. Added LICENSE (MIT) following gputop style
7. Created CONTRIBUTING (DCO) for developers

## Installation

### For Users

Once published to PyPI:
```bash
pip install trackio-view
```

With optional dependencies:
```bash
# Enhanced terminal graphics
pip install trackio-view[rich]

# NVIDIA GPU monitoring
pip install trackio-view[nvidia]

# All optional features
pip install trackio-view[all]
```

### For Development

1. Create a virtual environment:
```bash
cd /home/mcgrof/devel/trackio-view
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install in development mode:
```bash
pip install -e .[dev]
```

3. Run tests:
```bash
pytest
```

## Usage

The CLI command remains the same:
```bash
# Monitor training metrics
trackio-view --project my-experiment

# GPU monitoring
trackio-view --gpu

# Quick status check
trackio-view --once
```

## Publishing to PyPI

When ready to publish (requires PyPI account and credentials):

1. Install build tools:
```bash
pip install build twine
```

2. Build the distribution:
```bash
python -m build
```

3. Upload to PyPI:
```bash
# Test PyPI first (recommended)
python -m twine upload --repository testpypi dist/*

# Production PyPI
python -m twine upload dist/*
```

## Next Steps

1. **Initialize Git Repository:**
```bash
cd /home/mcgrof/devel/trackio-view
git init
git add .
git commit -m "Initial commit: Extract trackio-view as standalone package"
```

2. **Create GitHub Repository:**
   - Create a new repository on GitHub (e.g., `trackio-view`)
   - Push the code:
```bash
git remote add origin https://github.com/your-username/trackio-view.git
git push -u origin main
```

3. **Test the Package:**
```bash
# In a virtual environment
python3 -m venv test-venv
source test-venv/bin/activate
pip install -e .
trackio-view --help
pytest
```

4. **Publish to PyPI:**
   - Follow the publishing instructions above
   - Start with TestPyPI to verify everything works

5. **Coordinate with TrackIO Maintainers:**
   - Inform them the package is ready
   - Request documentation link from main TrackIO repo
   - Possibly contribute a PR to trackio to remove the trackio-view code

## Maintenance

### Version Bumping

Update versions in both:
- `pyproject.toml` (version field)
- `trackio_view/__init__.py` (__version__ variable)

### Code Quality

Always run before committing:
```bash
ruff check --fix --select I && ruff format
pytest
```

## Dependencies

### Required:
- Python >=3.8
- trackio >=0.1.0

### Optional:
- rich >=13.0.0 (enhanced terminal graphics)
- nvidia-ml-py >=11.0.0 (NVIDIA GPU monitoring)

## Support

- **Issues**: File on GitHub repository
- **Questions**: Use GitHub Discussions
- **Documentation**: See README.md and CONTRIBUTING.md

## License

MIT - Following gputop project style
