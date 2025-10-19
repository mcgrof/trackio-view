# Next Steps for TrackIO View

This document outlines the immediate next steps to complete the trackio-view package extraction and publication.

## Immediate Actions (Before Publishing)

### 1. Initialize Git Repository

```bash
cd /home/mcgrof/devel/trackio-view
git init
git add .
git commit -m "Initial commit: Extract trackio-view as standalone package

Extracted from trackio repository (branch 20250921-trackio-view)
following maintainer feedback to create a separate PyPI package.

Includes:
- Terminal-based dashboard for TrackIO metrics
- GPU monitoring (NVIDIA, AMD, Intel)
- Interactive zoom controls
- ASCII graphs with gradient coloring
- Complete test suite

Original commits:
- ada1991: GPU hardware monitoring
- b16e769: trackio-view terminal dashboard"
```

### 2. Create GitHub Repository

Option A: Create new repository
```bash
# On GitHub: Create repository named 'trackio-view'
# Then:
git remote add origin https://github.com/YOUR-USERNAME/trackio-view.git
git branch -M main
git push -u origin main
```

Option B: Fork and use separate repository
```bash
# Fork wandb/trackio on GitHub
# Clone your fork
# Create trackio-view as separate repo
```

### 3. Test Package Locally

```bash
# Create fresh virtual environment
python3 -m venv test-env
source test-env/bin/activate

# Install in development mode
cd /home/mcgrof/devel/trackio-view
pip install -e .

# Test CLI
trackio-view --help

# Run tests (requires trackio to be installed)
pip install trackio
pytest

# Test with actual data
python << EOF
import trackio
trackio.init(project="test-extraction")
for i in range(50):
    trackio.log({"loss": 5.0 - i * 0.05, "step": i})
trackio.finish()
EOF

trackio-view --project test-extraction --once
trackio-view --project test-extraction  # Live mode - press q to quit
```

### 4. Build Package

```bash
# Install build tools
pip install build twine

# Build distribution
python -m build

# This creates:
# - dist/trackio_view-0.1.0-py3-none-any.whl
# - dist/trackio-view-0.1.0.tar.gz
```

### 5. Test on TestPyPI (Recommended)

```bash
# Create account on test.pypi.org if needed
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple trackio-view

# Verify it works
trackio-view --help
```

### 6. Publish to Production PyPI

```bash
# Create account on pypi.org if needed
# Upload to production PyPI
python -m twine upload dist/*

# Verify installation
pip install trackio-view
trackio-view --help
```

## Post-Publication Actions

### 7. Update TrackIO Repository

Create PR to remove extracted code:

```bash
cd /home/mcgrof/devel/trackio
git checkout -b remove-trackio-view

# Remove files
git rm trackio/view.py
git rm trackio/gpu_monitor.py
git rm trackio/gpu_dashboard.py
git rm trackio/gpu_dashboard_gputop.py
git rm tests/test_cli_view.py
git rm docs/source/cli_trackio_view.md

# Update pyproject.toml - remove trackio-view entry point
# (Edit manually)

# Commit
git commit -m "Remove trackio-view code (now separate package)

trackio-view has been extracted to a standalone PyPI package
as requested by maintainers. Users can install it separately:

    pip install trackio-view

See: https://github.com/YOUR-USERNAME/trackio-view"

# Push and create PR
git push origin remove-trackio-view
```

### 8. Add Documentation Link to TrackIO

Create PR to add trackio-view reference:

```markdown
# In TrackIO README.md, add a section:

## Optional Extensions

### TrackIO View - Terminal Dashboard

For terminal-based monitoring of your experiments, check out the [trackio-view](https://github.com/YOUR-USERNAME/trackio-view) package:

```bash
pip install trackio-view
trackio-view --project my-experiment
```

Features:
- Real-time terminal dashboard
- GPU hardware monitoring
- Interactive zoom controls
- ASCII graphs with gradients
- Perfect for remote servers and SSH
```

### 9. Coordinate with Maintainers

Email or GitHub issue:

```
Hi TrackIO Team,

Following your suggestion to create a separate trackio-view package,
I've completed the extraction and published it to PyPI:

- PyPI: https://pypi.org/project/trackio-view/
- GitHub: https://github.com/YOUR-USERNAME/trackio-view
- Documentation: Full README with installation and usage

The package is fully functional and tested. I've also prepared PRs to:
1. Remove the trackio-view code from the main repo
2. Add a documentation link to the new package

Please let me know if you'd like me to submit these PRs or if you
have any feedback on the package structure.

Thanks for the guidance on keeping TrackIO lightweight!
```

## Optional: Create Release

```bash
cd /home/mcgrof/devel/trackio-view

# Tag release
git tag -a v0.1.0 -m "Release v0.1.0

Initial release of trackio-view as standalone package.

Features:
- Terminal-based dashboard for TrackIO
- GPU monitoring (NVIDIA, AMD, Intel)
- Interactive zoom controls
- ASCII graphs with gradient colors
- Complete test suite"

# Push tag
git push origin v0.1.0

# Create GitHub release from tag
# Add CHANGELOG.md entries
```

## Future Maintenance

### Version Bumping

When releasing new versions:

1. Update version in `trackio_view/__init__.py`
2. Update version in `pyproject.toml`
3. Update CHANGELOG.md (create if needed)
4. Commit changes
5. Tag release
6. Build and publish

```bash
# Example for v0.1.1
# Edit files...
git commit -m "Bump version to 0.1.1"
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin main --tags
python -m build
python -m twine upload dist/*
```

### Code Quality Checks

Before each release:

```bash
# Format code
ruff check --fix --select I && ruff format

# Run tests
pytest

# Check types (if using mypy)
# mypy trackio_view/

# Build and verify
python -m build
twine check dist/*
```

## Checklist

- [ ] Git repository initialized
- [ ] GitHub repository created
- [ ] Package tested locally
- [ ] Package built successfully
- [ ] Tested on TestPyPI
- [ ] Published to production PyPI
- [ ] TrackIO cleanup PR created
- [ ] TrackIO documentation PR created
- [ ] Maintainers notified
- [ ] Release tagged on GitHub
- [ ] CHANGELOG.md created (optional)

## Support

If you encounter issues:
1. Check SETUP_GUIDE.md for detailed instructions
2. Review CONTRIBUTING.md for development setup
3. Check EXTRACTION_SUMMARY.md for technical details
4. File issues on GitHub repository

## Resources

- Python Packaging Guide: https://packaging.python.org/
- PyPI Upload Guide: https://packaging.python.org/tutorials/packaging-projects/
- TestPyPI: https://test.pypi.org/
- Production PyPI: https://pypi.org/
