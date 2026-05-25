"""Psychrometric relationships for moist air in SI units.

This module wraps PsychroLib so the GUI has a small, stable API while relying
on a standard psychrometric implementation for the underlying HVAC properties.
Values are returned in SI units unless a function name states otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from psychrolib import (
    GetHumRatioFromRelHum,
    GetHumRatioFromTWetBulb,
    GetMoistAirEnthalpy,
    GetMoistAirVolume,
    GetRelHumFromHumRatio,
    GetSatHumRatio,
    GetSatVapPres,
    GetStandardAtmPressure,
    GetTDewPointFromHumRatio,
    GetTWetBulbFromHumRatio,
    GetVapPresFromHumRatio,
    SI,
    SetUnitSystem,
)

EPSILON = 0.621945
R_DA = 287.055
SEA_LEVEL_PRESSURE_PA = 101_325.0

SetUnitSystem(SI)


def standard_atmospheric_pressure_pa(altitude_m: float) -> float:
    """Return PsychroLib standard atmospheric pressure in Pa."""

    return float(GetStandardAtmPressure(float(altitude_m)))


def saturation_vapor_pressure_pa(temperature_c: float) -> float:
    """Return PsychroLib saturation vapor pressure in Pa."""

    return float(GetSatVapPres(float(temperature_c)))


def saturation_humidity_ratio_kg_per_kg(
    temperature_c: float, pressure_pa: float = SEA_LEVEL_PRESSURE_PA
) -> float:
    """Return saturation humidity ratio in kg water / kg dry air."""

    try:
        return float(GetSatHumRatio(float(temperature_c), float(pressure_pa)))
    except ValueError:
        return math.inf


def humidity_ratio_from_rh(
    temperature_c: float,
    relative_humidity: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Return humidity ratio from dry-bulb temperature and RH fraction."""

    rh = min(max(float(relative_humidity), 0.0), 1.0)
    return float(GetHumRatioFromRelHum(float(temperature_c), rh, float(pressure_pa)))


def vapor_pressure_from_humidity_ratio(
    humidity_ratio: float, pressure_pa: float = SEA_LEVEL_PRESSURE_PA
) -> float:
    """Return water vapor partial pressure in Pa."""

    return float(
        GetVapPresFromHumRatio(
            max(float(humidity_ratio), 0.0), float(pressure_pa)
        )
    )


def relative_humidity_from_humidity_ratio(
    temperature_c: float,
    humidity_ratio: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Return relative humidity as a fraction for a dry-bulb/W state."""

    return float(
        GetRelHumFromHumRatio(
            float(temperature_c), max(float(humidity_ratio), 0.0), float(pressure_pa)
        )
    )


def dew_point_c_from_vapor_pressure(vapor_pressure_pa: float) -> float:
    """Invert PsychroLib saturation pressure to dew-point temperature."""

    vapor_pressure_pa = max(float(vapor_pressure_pa), 1.0)
    low = -100.0
    high = 100.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if saturation_vapor_pressure_pa(mid) < vapor_pressure_pa:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def dew_point_c(
    temperature_c: float,
    relative_humidity: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Return PsychroLib dew point in degrees C for a dry-bulb/RH state."""

    w = humidity_ratio_from_rh(temperature_c, relative_humidity, pressure_pa)
    return float(
        GetTDewPointFromHumRatio(float(temperature_c), w, float(pressure_pa))
    )


def dew_point_c_from_humidity_ratio(
    humidity_ratio: float, pressure_pa: float = SEA_LEVEL_PRESSURE_PA
) -> float:
    """Return dew point in degrees C for a humidity ratio."""

    return dew_point_c_from_vapor_pressure(
        vapor_pressure_from_humidity_ratio(humidity_ratio, pressure_pa)
    )


def enthalpy_kj_per_kg_da(temperature_c: float, humidity_ratio: float) -> float:
    """PsychroLib moist air enthalpy in kJ / kg dry air."""

    return (
        float(
            GetMoistAirEnthalpy(
                float(temperature_c), max(float(humidity_ratio), 0.0)
            )
        )
        / 1000.0
    )


def humidity_ratio_from_enthalpy(
    temperature_c: float, enthalpy_kj_per_kg_da_value: float
) -> float:
    """Return humidity ratio for a dry-bulb temperature and enthalpy."""

    # PsychroLib exposes enthalpy from W, but not the inverse needed for chart
    # guide lines. This is the algebraic inverse of PsychroLib's SI formula.
    return max(
        (float(enthalpy_kj_per_kg_da_value) * 1000.0 - 1006.0 * float(temperature_c))
        / (2_501_000.0 + 1860.0 * float(temperature_c)),
        0.0,
    )


def specific_volume_m3_per_kg_da(
    temperature_c: float,
    humidity_ratio: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """PsychroLib specific volume of moist air in m3 / kg dry air."""

    return float(
        GetMoistAirVolume(
            float(temperature_c), max(float(humidity_ratio), 0.0), float(pressure_pa)
        )
    )


def humidity_ratio_from_specific_volume(
    temperature_c: float,
    specific_volume_m3_per_kg_da_value: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Return humidity ratio for a dry-bulb temperature and specific volume."""

    # Inverse of PsychroLib's moist-air specific-volume relationship.
    temperature_k = float(temperature_c) + 273.15
    numerator = float(specific_volume_m3_per_kg_da_value) * float(pressure_pa)
    return max((numerator / (R_DA * temperature_k) - 1.0) / 1.607858, 0.0)


def humidity_ratio_from_wet_bulb(
    dry_bulb_c: float,
    wet_bulb_c: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Return PsychroLib humidity ratio from dry-bulb and wet-bulb."""

    dry_bulb_c = float(dry_bulb_c)
    wet_bulb_c = min(float(wet_bulb_c), dry_bulb_c)
    return float(
        GetHumRatioFromTWetBulb(dry_bulb_c, wet_bulb_c, float(pressure_pa))
    )


def wet_bulb_c_from_state(
    dry_bulb_c: float,
    humidity_ratio: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Return PsychroLib wet-bulb temperature from dry-bulb/W state."""

    return float(
        GetTWetBulbFromHumRatio(
            float(dry_bulb_c), max(float(humidity_ratio), 0.0), float(pressure_pa)
        )
    )


@dataclass(frozen=True)
class PsychrometricState:
    """Computed state point for a dry-bulb/RH pair."""

    dry_bulb_c: float
    relative_humidity: float
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA

    @property
    def humidity_ratio(self) -> float:
        return humidity_ratio_from_rh(
            self.dry_bulb_c, self.relative_humidity, self.pressure_pa
        )

    @property
    def humidity_ratio_g_per_kg(self) -> float:
        return self.humidity_ratio * 1000.0

    @property
    def dew_point_c(self) -> float:
        return float(
            GetTDewPointFromHumRatio(
                float(self.dry_bulb_c), self.humidity_ratio, float(self.pressure_pa)
            )
        )

    @property
    def wet_bulb_c(self) -> float:
        return wet_bulb_c_from_state(
            self.dry_bulb_c, self.humidity_ratio, self.pressure_pa
        )

    @property
    def enthalpy_kj_per_kg_da(self) -> float:
        return enthalpy_kj_per_kg_da(self.dry_bulb_c, self.humidity_ratio)

    @property
    def specific_volume_m3_per_kg_da(self) -> float:
        return specific_volume_m3_per_kg_da(
            self.dry_bulb_c, self.humidity_ratio, self.pressure_pa
        )

    @property
    def vapor_pressure_pa(self) -> float:
        return vapor_pressure_from_humidity_ratio(self.humidity_ratio, self.pressure_pa)


def sensible_heat_ratio(
    start: PsychrometricState, end: PsychrometricState
) -> float | None:
    """Return approximate sensible heat ratio between two state points."""

    delta_h = end.enthalpy_kj_per_kg_da - start.enthalpy_kj_per_kg_da
    if abs(delta_h) < 1.0e-9:
        return None
    sensible = 1.006 * (end.dry_bulb_c - start.dry_bulb_c)
    return sensible / delta_h
