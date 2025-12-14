"""Base classes and extendable configuration for feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional
import copy

# Sensor-specific sampling rates
CURRENT_SAMPLING_RATE = 10000.0   # LTR11 - Current sensors
VIBRATION_SAMPLING_RATE = 26041.0 # LTR22 - Vibration sensors
ENV_CARRIER_FREQUENCY = 1670.0    # Hz - Expected carrier frequency for Hilbert envelope analysis


@dataclass
class FeatureFamilyConfig:
    """Container for per-feature-family configuration."""

    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "FeatureFamilyConfig":
        """Return a deep copy of the family configuration."""
        return FeatureFamilyConfig(enabled=self.enabled, params=copy.deepcopy(self.params))


def _default_family_config() -> Dict[str, FeatureFamilyConfig]:
    """Factory for the default feature family setup."""
    return {
        "time_domain": FeatureFamilyConfig(enabled=True, params={"channels": None}),
        "frequency_domain": FeatureFamilyConfig(enabled=False, params={"window_type": "hann"}),
        "hilbert_envelope": FeatureFamilyConfig(
            enabled=True,
            params={
                "expected_carrier": ENV_CARRIER_FREQUENCY,
                "carrier_bandwidth": 50.0,
            },
        ),
        "cross_channel": FeatureFamilyConfig(
            enabled=True,
            params={
                "pairs": None,
                "time_domain": True,
                "frequency_domain": False,
                "hilbert_envelope": True,
            },
        ),
    }


_DEFAULT_FAMILY_FLAGS = {
    name: cfg.enabled for name, cfg in _default_family_config().items()
}


@dataclass
class FeatureConfig:
    """Configuration object orchestrating feature family selection and tuning."""

    sampling_rate: float = CURRENT_SAMPLING_RATE
    window_type: str = "hann"
    families: Dict[str, FeatureFamilyConfig] = field(default_factory=_default_family_config)
    sensor_type: Optional[str] = None
    channel_scope: Optional[List[str]] = None

    # Default sensor profiles for convenience. These profiles can be extended as new
    # sensor modalities are introduced without modifying call-sites.
    SENSOR_PROFILES: ClassVar[Dict[str, Dict[str, Any]]] = {
        "current": {
            "sampling_rate": CURRENT_SAMPLING_RATE,
            "channel_scope": ["ph_a"],
            "families": {
                "time_domain": {"enabled": True},
                "frequency_domain": {"enabled": False},
                "hilbert_envelope": {
                    "enabled": True,
                    "params": {"expected_carrier": ENV_CARRIER_FREQUENCY},
                },
                "cross_channel": {"enabled": True},
            },
        },
        "vibration": {
            "sampling_rate": VIBRATION_SAMPLING_RATE,
            "channel_scope": None,
            "families": {
                "time_domain": {"enabled": True},
                "frequency_domain": {
                    "enabled": True,
                    "params": {"window_type": "hann"},
                },
                "hilbert_envelope": {"enabled": False},
                "cross_channel": {"enabled": True},
            },
        },
    }

    def copy(self) -> "FeatureConfig":
        """Return a deep copy of the configuration."""
        return FeatureConfig(
            sampling_rate=self.sampling_rate,
            window_type=self.window_type,
            families={name: fam.copy() for name, fam in self.families.items()},
            sensor_type=self.sensor_type,
            channel_scope=copy.deepcopy(self.channel_scope),
        )

    # -- family registration and manipulation -------------------------------------------------
    def register_family(
        self,
        name: str,
        *,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """Register (or replace) a feature family configuration."""

        if not overwrite and name in self.families:
            raise ValueError(
                f"Feature family '{name}' already exists. Use overwrite=True to replace it."
            )

        self.families[name] = FeatureFamilyConfig(
            enabled=enabled, params=copy.deepcopy(params or {})
        )

    def set_family(
        self,
        name: str,
        *,
        enabled: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a feature family while preserving unspecified fields."""

        if name not in self.families:
            self.register_family(
                name,
                enabled=True if enabled is None else enabled,
                params=params or {},
            )
            return

        family = self.families[name]
        if enabled is not None:
            family.enabled = enabled
        if params:
            family.params.update(params)

    def is_enabled(self, name: str) -> bool:
        """Return whether a family is enabled."""

        family = self.families.get(name)
        return family.enabled if family else False

    def get_params(self, name: str) -> Dict[str, Any]:
        """Return the parameter dictionary for a family (copy not required)."""

        family = self.families.get(name)
        return family.params if family else {}

    def enable(self, name: str, **params: Any) -> None:
        """Enable a feature family and optionally update parameters."""

        self.set_family(name, enabled=True, params=params or None)

    def disable(self, name: str) -> None:
        """Disable a feature family."""

        self.set_family(name, enabled=False)

    # -- family accessors for backwards compatibility -----------------------------------------
    @property
    def time_domain(self) -> bool:
        return self.is_enabled("time_domain")

    @time_domain.setter
    def time_domain(self, value: bool) -> None:
        self.set_family("time_domain", enabled=value)

    @property
    def frequency_domain(self) -> bool:
        return self.is_enabled("frequency_domain")

    @frequency_domain.setter
    def frequency_domain(self, value: bool) -> None:
        self.set_family("frequency_domain", enabled=value)

    @property
    def hilbert_envelope(self) -> bool:
        return self.is_enabled("hilbert_envelope")

    @hilbert_envelope.setter
    def hilbert_envelope(self, value: bool) -> None:
        self.set_family("hilbert_envelope", enabled=value)

    @property
    def cross_channel(self) -> bool:
        return self.is_enabled("cross_channel")

    @cross_channel.setter
    def cross_channel(self, value: bool) -> None:
        self.set_family("cross_channel", enabled=value)

    # -- sensor profiles ----------------------------------------------------------------------
    def apply_sensor_profile(self, sensor_type: str, *, override: bool = False) -> None:
        """Apply sensor-specific defaults to the configuration."""

        if not sensor_type:
            return

        profile = self.SENSOR_PROFILES.get(sensor_type.lower())
        if not profile:
            return

        target_rate = profile.get("sampling_rate")
        if target_rate is not None and (override or self.sampling_rate == CURRENT_SAMPLING_RATE):
            self.sampling_rate = target_rate

        profile_scope = profile.get("channel_scope")
        if override or self.channel_scope is None:
            self.channel_scope = copy.deepcopy(profile_scope)

        for family_name, family_data in profile.get("families", {}).items():
            default_flag = _DEFAULT_FAMILY_FLAGS.get(family_name)
            current_family = self.families.get(family_name)

            should_override = (
                override
                or current_family is None
                or (default_flag is not None and current_family.enabled == default_flag)
            )

            if not should_override and not override:
                # Still merge params if we have none set yet.
                if current_family and not current_family.params and family_data.get("params"):
                    current_family.params.update(copy.deepcopy(family_data["params"]))
                continue

            self.set_family(
                family_name,
                enabled=family_data.get("enabled"),
                params=copy.deepcopy(family_data.get("params")) if family_data.get("params") else None,
            )

        freq_profile = profile.get("families", {}).get("frequency_domain", {})
        freq_params = freq_profile.get("params", {})
        if "window_type" in freq_params and (override or self.window_type == "hann"):
            self.window_type = freq_params["window_type"]

        self.sensor_type = sensor_type.lower()

    @classmethod
    def for_sensor(cls, sensor_type: str, **overrides: Any) -> "FeatureConfig":
        """Create a configuration pre-populated with the given sensor profile."""

        config = cls()
        config.apply_sensor_profile(sensor_type, override=True)
        if overrides:
            config.update_from_dict(overrides)
        return config

    # -- helpers ------------------------------------------------------------------------------
    def resolve_channel_scope(self, available_names: List[str]) -> List[str]:
        """Return the channel names to use for per-channel features."""

        if not available_names:
            return []

        if self.channel_scope:
            selected = [name for name in self.channel_scope if name in available_names]
            return selected if selected else available_names

        return available_names

    def update_from_dict(self, overrides: Dict[str, Any]) -> None:
        """Bulk update configuration from a dictionary payload."""

        for key, value in overrides.items():
            if key == "sampling_rate":
                self.sampling_rate = value
            elif key == "window_type":
                self.window_type = value
            elif key == "channel_scope":
                self.channel_scope = copy.deepcopy(value)
            elif key == "families" and isinstance(value, dict):
                for family_name, family_data in value.items():
                    params = family_data.get("params") if isinstance(family_data, dict) else None
                    enabled = family_data.get("enabled") if isinstance(family_data, dict) else None
                    self.set_family(family_name, enabled=enabled, params=params)
            else:
                # Fallback: treat scalar bool overrides as toggles
                if isinstance(value, bool):
                    self.set_family(key, enabled=value)
                else:
                    self.set_family(key, params=value if isinstance(value, dict) else None)
