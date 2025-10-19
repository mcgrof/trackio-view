# SPDX-License-Identifier: MIT
"""
TrackIO View - Terminal-based dashboard for TrackIO experiment tracking.

A lightweight, terminal-based monitoring tool for machine learning experiments tracked with TrackIO.
Perfect for monitoring training progress on remote servers or in terminal-only environments.
"""

__version__ = "0.1.0"

from trackio_view.view import TrackIOViewer, main

__all__ = ["TrackIOViewer", "main", "__version__"]
