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


def process_line_humidity_ratio(
    temperature_c: float, start: PsychrometricState, end: PsychrometricState
) -> float | None:
    """Return humidity ratio on the straight process line at a temperature."""

    delta_t = end.dry_bulb_c - start.dry_bulb_c
    if abs(delta_t) < 1.0e-9:
        return None
    slope = (end.humidity_ratio - start.humidity_ratio) / delta_t
    return start.humidity_ratio + slope * (float(temperature_c) - start.dry_bulb_c)


def apparatus_dew_point_c(
    start: PsychrometricState,
    end: PsychrometricState,
    *,
    low_temperature_c: float = -80.0,
    high_temperature_c: float | None = None,
) -> float | None:
    """Return the ADP where the process line intersects saturation.

    The apparatus dew point is found by extending the straight line through the
    entering and leaving states until it crosses the PsychroLib saturation
    curve. The root nearest the leaving state is returned, which matches the
    common coil-analysis use case.
    """

    if abs(end.dry_bulb_c - start.dry_bulb_c) < 1.0e-9:
        return None

    pressure_pa = end.pressure_pa
    high = (
        max(start.dry_bulb_c, end.dry_bulb_c) + 10.0
        if high_temperature_c is None
        else float(high_temperature_c)
    )
    low = min(float(low_temperature_c), min(start.dry_bulb_c, end.dry_bulb_c) - 30.0)

    def residual(temperature_c: float) -> float | None:
        line_w = process_line_humidity_ratio(temperature_c, start, end)
        if line_w is None or line_w < 0.0:
            return None
        saturation_w = saturation_humidity_ratio_kg_per_kg(temperature_c, pressure_pa)
        if not math.isfinite(saturation_w):
            return None
        return line_w - saturation_w

    roots: list[float] = []
    sample_count = 1200
    previous_t: float | None = None
    previous_value: float | None = None

    for index in range(sample_count + 1):
        temperature = low + (high - low) * index / sample_count
        value = residual(temperature)
        if value is None:
            continue
        if abs(value) < 1.0e-8:
            roots.append(temperature)
        elif previous_value is not None and previous_t is not None:
            if value * previous_value < 0.0:
                left = previous_t
                right = temperature
                left_value = previous_value
                for _ in range(80):
                    middle = (left + right) / 2.0
                    middle_value = residual(middle)
                    if middle_value is None:
                        break
                    if abs(middle_value) < 1.0e-10:
                        left = right = middle
                        break
                    if left_value * middle_value <= 0.0:
                        right = middle
                    else:
                        left = middle
                        left_value = middle_value
                roots.append((left + right) / 2.0)
        previous_t = temperature
        previous_value = value

    if not roots:
        return None

    return min(roots, key=lambda root: abs(root - end.dry_bulb_c))


def coil_bypass_factor(
    start: PsychrometricState, end: PsychrometricState, adp_c: float
) -> float | None:
    """Return temperature-based coil bypass factor from entering/leaving/ADP."""

    denominator = start.dry_bulb_c - adp_c
    if abs(denominator) < 1.0e-9:
        return None
    return (end.dry_bulb_c - adp_c) / denominator


def sensible_heat_ratio(
    start: PsychrometricState, end: PsychrometricState
) -> float | None:
    """Return approximate sensible heat ratio between two state points."""

    delta_h = end.enthalpy_kj_per_kg_da - start.enthalpy_kj_per_kg_da
    if abs(delta_h) < 1.0e-9:
        return None
    sensible = 1.006 * (end.dry_bulb_c - start.dry_bulb_c)
    return sensible / delta_h
