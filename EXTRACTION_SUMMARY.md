# TrackIO View Extraction Summary

This document summarizes the extraction of trackio-view from the main TrackIO repository into a standalone PyPI package.

## Motivation

Following feedback from TrackIO maintainers:

> "This is very cool @mcgrof but is a bit out of scope and will be hard for us to maintain. Trackio is designed to be pretty lightweight and extensible so I'd recommend instead creating a separate trackio-view pypi package which we could link to from our documentation."

## Source Location

- **Original Repository**: https://github.com/wandb/trackio
- **Original Branch**: `20250921-trackio-view`
- **Commits**:
  - ada1991: [Add] GPU hardware monitoring to trackio and trackio-view
  - b16e769: [Add] trackio-view terminal-based dashboard command

## Files Extracted

### Source Code (from trackio/)

| Original Location | New Location | Changes |
|------------------|--------------|---------|
| `trackio/view.py` | `trackio_view/view.py` | Updated imports: `trackio.X` → `trackio_view.X` |
| `trackio/gpu_monitor.py` | `trackio_view/gpu_monitor.py` | Updated imports: `trackio.X` → `trackio_view.X` |
| `trackio/gpu_dashboard.py` | `trackio_view/gpu_dashboard.py` | Updated imports: `trackio.X` → `trackio_view.X` |
| `trackio/gpu_dashboard_gputop.py` | `trackio_view/gpu_dashboard_gputop.py` | Updated imports: `trackio.X` → `trackio_view.X` |

### Tests (from tests/)

| Original Location | New Location | Changes |
|------------------|--------------|---------|
| `tests/test_cli_view.py` | `tests/test_cli_view.py` | Updated all imports and CLI invocations |

### Assets (from trackio/assets/)

| Original Location | New Location |
|------------------|--------------|
| `trackio/assets/trackio-view-demo-01.png` | `assets/trackio-view-demo-01.png` |
| `trackio/assets/trackio-view-zoom.png` | `assets/trackio-view-zoom.png` |
| `trackio/assets/trackio-view-zoom-x2.png` | `assets/trackio-view-zoom-x2.png` |
| `trackio/assets/trackio-view-gpu.png` | `assets/trackio-view-gpu.png` |

## New Files Created

### Package Configuration
- `pyproject.toml` - Package metadata and dependencies
- `MANIFEST.in` - Distribution file list
- `LICENSE` - MIT license (following gputop style)
- `CONTRIBUTING` - Developer Certificate of Origin (DCO)
- `.gitignore` - Python and project-specific ignores

### Documentation
- `README.md` - User-facing documentation with:
  - Background and motivation
  - Installation instructions
  - Usage examples
  - GPU monitoring features
  - Visual screenshots
  - Architecture overview
- `CONTRIBUTING.md` - Developer guide
- `SETUP_GUIDE.md` - Setup and publishing instructions
- `EXTRACTION_SUMMARY.md` - This document

### Package Structure
- `trackio_view/__init__.py` - Package initialization
- `tests/__init__.py` - Test package initialization
- `tests/conftest.py` - Pytest fixtures

## Import Changes

All imports were updated from:
```python
from trackio.view import X
from trackio.gpu_monitor import X
```

To:
```python
from trackio_view.view import X
from trackio_view.gpu_monitor import X
```

## Dependencies

### Required
- Python >=3.8
- trackio >=0.1.0

### Optional
- rich >=13.0.0 (enhanced terminal graphics)
- nvidia-ml-py >=11.0.0 (NVIDIA GPU monitoring)

## Package Distribution

### PyPI Package Name
`trackio-view`

### CLI Entry Point
`trackio-view` (same as before)

### Installation Commands
```bash
pip install trackio-view           # Base package
pip install trackio-view[rich]     # With enhanced graphics
pip install trackio-view[nvidia]   # With NVIDIA GPU support
pip install trackio-view[all]      # All optional dependencies
```

## Testing

All original tests were preserved and updated:
- Import statements updated to use `trackio_view`
- CLI invocations updated to use `trackio_view.view`
- Test fixtures adapted for standalone package

Run tests with:
```bash
pytest
```

## Directory Structure

```
trackio-view/
├── assets/                         # Screenshots and images
│   ├── trackio-view-demo-01.png
│   ├── trackio-view-gpu.png
│   ├── trackio-view-zoom.png
│   └── trackio-view-zoom-x2.png
├── LICENSES/                       # SPDX license files
│   └── preferred/
│       └── MIT
├── trackio_view/                   # Main package
│   ├── __init__.py
│   ├── view.py
│   ├── gpu_monitor.py
│   ├── gpu_dashboard.py
│   └── gpu_dashboard_gputop.py
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   └── test_cli_view.py
├── .gitignore
├── CONTRIBUTING
├── EXTRACTION_SUMMARY.md
├── LICENSE
├── MANIFEST.in
├── NEXT_STEPS.md
├── pyproject.toml
├── README.md
└── SETUP_GUIDE.md
```

## What Remains in TrackIO

The following should be removed from the main TrackIO repository after this package is published:

### Files to Remove
- `trackio/view.py`
- `trackio/gpu_monitor.py`
- `trackio/gpu_dashboard.py`
- `trackio/gpu_dashboard_gputop.py`
- `tests/test_cli_view.py`
- `trackio/assets/trackio-view-*.png`
- `docs/source/cli_trackio_view.md`

### Configuration Updates
- Remove `trackio-view` entry point from `pyproject.toml`
- Update documentation to reference the standalone package
- Add link to `trackio-view` in main README

## Publishing Checklist

- [ ] Test package installation in clean virtual environment
- [ ] Run full test suite
- [ ] Build distribution: `python -m build`
- [ ] Test on TestPyPI: `python -m twine upload --repository testpypi dist/*`
- [ ] Verify installation from TestPyPI
- [ ] Publish to production PyPI: `python -m twine upload dist/*`
- [ ] Create GitHub repository for trackio-view
- [ ] Tag release version
- [ ] Update TrackIO repository to remove extracted code
- [ ] Update TrackIO documentation to link to trackio-view

## Coordination with TrackIO Maintainers

After publishing:
1. Inform maintainers that trackio-view is available on PyPI
2. Request documentation link from main TrackIO repo
3. Submit PR to trackio to:
   - Remove trackio-view code
   - Add link to trackio-view in documentation
   - Update README to mention trackio-view as optional extension

## License

Uses MIT license following the gputop project style.

## Maintenance

This package is now independently maintained from TrackIO core. Updates and maintenance will be tracked separately.

## Contact

- GitHub Issues: (to be created after repository setup)
- Original Developer: Luis Chamberlain (@mcgrof)
- TrackIO Repository: https://github.com/wandb/trackio
