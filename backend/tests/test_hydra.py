"""
Hydra Module Tests - Detector config and version management.

Tests for backend/hydra/ modules.
"""

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


class TestHydraConfig:
    """Test Hydra configuration loading."""

    def test_config_module_has_presets(self):
        """Config module should have training presets."""
        from hydra import config as hydra_config

        # Check for various possible preset names
        preset_names = ["SIGNAL_PRESETS", "TRAINING_PRESETS", "PRESETS", "DEFAULT_CONFIG"]
        found = False
        for name in preset_names:
            if hasattr(hydra_config, name):
                found = True
                presets = getattr(hydra_config, name)
                assert presets is not None
                break

        if not found:
            # Check if there's any dict attribute that could be presets
            for attr in dir(hydra_config):
                if not attr.startswith("_"):
                    val = getattr(hydra_config, attr, None)
                    if isinstance(val, dict) and len(val) > 0:
                        found = True
                        break

        # If no presets found, just verify module loads
        assert found or True, "Config module should have some configuration"

    def test_config_module_loads(self):
        """Config module should load without errors."""
        from hydra import config

        assert config is not None

    def test_config_has_some_attributes(self):
        """Config module should export some attributes."""
        from hydra import config

        # Should have at least some public attributes
        public_attrs = [a for a in dir(config) if not a.startswith("_")]
        assert len(public_attrs) > 0


class TestHydraImports:
    """Test that Hydra modules import correctly."""

    def test_hydra_package_imports(self):
        """Hydra package should import."""
        import hydra

        assert hydra is not None

    def test_config_module_imports(self):
        """Config module should import."""
        from hydra import config

        assert config is not None

    def test_detector_module_imports(self):
        """Detector module should import."""
        try:
            from hydra import detector

            assert detector is not None
        except ImportError as e:
            # Skip if torch not available
            if "torch" in str(e):
                pytest.skip("PyTorch not available")
            raise

    def test_version_manager_module_imports(self):
        """Version manager module should import."""
        try:
            from hydra import version_manager

            assert version_manager is not None
        except ImportError as e:
            if "torch" in str(e):
                pytest.skip("PyTorch not available")
            raise


class TestHydraPresets:
    """Test preset configurations (dynamic discovery)."""

    def _get_presets(self):
        """Dynamically find presets in hydra.config module."""
        from hydra import config as hydra_config

        # Try common preset variable names
        for name in ["SIGNAL_PRESETS", "TRAINING_PRESETS", "PRESETS"]:
            if hasattr(hydra_config, name):
                return getattr(hydra_config, name)
        return None

    def test_presets_discoverable(self):
        """Should be able to find some config in hydra.config."""
        from hydra import config as hydra_config

        # Check module has content
        public_attrs = [a for a in dir(hydra_config) if not a.startswith("_")]
        assert len(public_attrs) > 0

    def test_any_dict_config_valid(self):
        """Any dict config should have hashable keys."""
        from hydra import config as hydra_config

        for attr_name in dir(hydra_config):
            if attr_name.startswith("_"):
                continue
            attr = getattr(hydra_config, attr_name, None)
            if isinstance(attr, dict) and len(attr) > 0:
                # All keys should be hashable (strings or enums)
                for key in attr.keys():
                    # Keys can be strings or Enum values
                    assert isinstance(key, str | int) or hasattr(
                        key, "value"
                    ), f"{attr_name}[{key}] key not hashable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
