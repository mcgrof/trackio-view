# License Migration to MIT

This document tracks the migration from Apache 2.0 to MIT license, following the gputop project style.

## Changes Made

### 1. License Files

**Removed:**
- Old Apache 2.0 LICENSE file (11KB)

**Added:**
- `LICENSE` - Short MIT license statement (matching gputop style)
- `LICENSES/preferred/MIT` - Full SPDX-compliant MIT license text

### 2. CONTRIBUTING File

**Changed:**
- `CONTRIBUTING.md` (detailed developer guide) → `CONTRIBUTING` (DCO-focused)
- Now follows Developer Certificate of Origin (DCO) model from gputop
- Requires Signed-off-by tags in commit messages

### 3. Source Code Headers

All Python source files now include SPDX license identifiers:

```python
# SPDX-License-Identifier: MIT
```

Files updated:
- `trackio_view/__init__.py`
- `trackio_view/view.py`
- `trackio_view/gpu_monitor.py`
- `trackio_view/gpu_dashboard.py`
- `trackio_view/gpu_dashboard_gputop.py`

### 4. Package Configuration

Updated `pyproject.toml`:
- License: `Apache-2.0` → `MIT`
- Classifier: `License :: OSI Approved :: Apache Software License` → `License :: OSI Approved :: MIT License`

### 5. Documentation Updates

Updated license references in:
- `README.md`
- `SETUP_GUIDE.md`
- `EXTRACTION_SUMMARY.md`
- `MANIFEST.in` (added CONTRIBUTING and LICENSES/)

## License Compatibility

MIT license is:
- ✅ More permissive than Apache 2.0
- ✅ Compatible with Apache 2.0 projects
- ✅ Simpler and shorter
- ✅ Widely used and well-understood
- ✅ Matches gputop project style

## Why MIT?

Following the gputop project style provides:
1. **Consistency** - Same license as the inspiration project
2. **Simplicity** - MIT is simpler than Apache 2.0
3. **Compatibility** - MIT works with virtually any project
4. **Developer-friendly** - Familiar DCO contribution model

## Developer Certificate of Origin

Contributors must now add `Signed-off-by` tags to commits:

```bash
git commit -s -m "Your commit message"
```

This certifies that the contribution complies with the DCO as stated in the CONTRIBUTING file.

## SPDX Compliance

The project now follows SPDX best practices:
- All source files have SPDX-License-Identifier headers
- Full license text in LICENSES/preferred/MIT
- SPDX-URL and usage guide included

## Copyright Holder

Copyright (c) 2025 Luis Chamberlain <mcgrof@kernel.org>

## Effective Date

This license change is effective from version 0.1.0 onwards, as the package was extracted from TrackIO as a new standalone project before initial publication.
