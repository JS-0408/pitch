"""
Unit tests for ExpoApp (T7.1).
"""
from __future__ import annotations

import unittest
from yaazhi.app.expo_app import ExpoApp


class TestExpoApp(unittest.TestCase):
    def test_expo_app_instantiation(self) -> None:
        app = ExpoApp(fps=30.0, display_scale=2)
        self.assertIsNotNone(app.orchestrator)
        self.assertTrue(app.reprojection_enabled)
        self.assertFalse(app.smoke_enabled)


if __name__ == "__main__":
    unittest.main()
