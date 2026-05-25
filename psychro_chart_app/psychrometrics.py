"""Psychrometric relationships for moist air in SI units.

The equations here intentionally avoid GUI dependencies so the numerical layer
can be tested independently of Qt. Values are suitable for visualization and
engineering exploration across ordinary HVAC temperature ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

EPSILON = 0.621945
R_DA = 287.055
SEA_LEVEL_PRESSURE_PA = 101_325.0


def standard_atmospheric_pressure_pa(altitude_m: float) -> float:
    """Return standard atmospheric pressure in Pa for a geometric altitude."""

    # International Standard Atmosphere, troposphere. Clamp just below the
    # singularity so an adventurous UI slider still returns a meaningful value.
    altitude_m = min(float(altitude_m), 44_000.0)
    base = max(1.0 - 2.25577e-5 * altitude_m, 1.0e-6)
    return SEA_LEVEL_PRESSURE_PA * base**5.2559


def saturation_vapor_pressure_pa(temperature_c: float) -> float:
    """Buck saturation vapor pressure over water/ice in Pa."""

    temperature_c = float(temperature_c)
    if temperature_c >= 0.0:
        exponent = (18.678 - temperature_c / 234.5) * (
            temperature_c / (257.14 + temperature_c)
        )
        return 611.21 * math.exp(exponent)

    exponent = (23.036 - temperature_c / 333.7) * (
        temperature_c / (279.82 + temperature_c)
    )
    return 611.15 * math.exp(exponent)


def saturation_humidity_ratio_kg_per_kg(
    temperature_c: float, pressure_pa: float = SEA_LEVEL_PRESSURE_PA
) -> float:
    """Return saturation humidity ratio in kg water / kg dry air."""

    p_ws = saturation_vapor_pressure_pa(temperature_c)
    if p_ws >= pressure_pa:
        return math.inf
    return EPSILON * p_ws / (pressure_pa - p_ws)


def humidity_ratio_from_rh(
    temperature_c: float,
    relative_humidity: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Return humidity ratio from dry-bulb temperature and RH fraction."""

    rh = min(max(float(relative_humidity), 0.0), 1.0)
    p_v = rh * saturation_vapor_pressure_pa(temperature_c)
    if p_v >= pressure_pa:
        return math.inf
    return EPSILON * p_v / (pressure_pa - p_v)


def vapor_pressure_from_humidity_ratio(
    humidity_ratio: float, pressure_pa: float = SEA_LEVEL_PRESSURE_PA
) -> float:
    """Return water vapor partial pressure in Pa."""

    w = max(float(humidity_ratio), 0.0)
    return pressure_pa * w / (EPSILON + w)


def relative_humidity_from_humidity_ratio(
    temperature_c: float,
    humidity_ratio: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Return relative humidity as a fraction for a dry-bulb/W state."""

    p_v = vapor_pressure_from_humidity_ratio(humidity_ratio, pressure_pa)
    return p_v / saturation_vapor_pressure_pa(temperature_c)


def dew_point_c_from_vapor_pressure(vapor_pressure_pa: float) -> float:
    """Invert saturation pressure to dew-point temperature with bisection."""

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
    """Return dew point in degrees C for a dry-bulb/RH state."""

    w = humidity_ratio_from_rh(temperature_c, relative_humidity, pressure_pa)
    return dew_point_c_from_humidity_ratio(w, pressure_pa)


def dew_point_c_from_humidity_ratio(
    humidity_ratio: float, pressure_pa: float = SEA_LEVEL_PRESSURE_PA
) -> float:
    """Return dew point in degrees C for a humidity ratio."""

    return dew_point_c_from_vapor_pressure(
        vapor_pressure_from_humidity_ratio(humidity_ratio, pressure_pa)
    )


def enthalpy_kj_per_kg_da(temperature_c: float, humidity_ratio: float) -> float:
    """Moist air enthalpy in kJ / kg dry air."""

    return 1.006 * temperature_c + humidity_ratio * (2501.0 + 1.86 * temperature_c)


def humidity_ratio_from_enthalpy(
    temperature_c: float, enthalpy_kj_per_kg_da_value: float
) -> float:
    """Return humidity ratio for a dry-bulb temperature and enthalpy."""

    return (enthalpy_kj_per_kg_da_value - 1.006 * temperature_c) / (
        2501.0 + 1.86 * temperature_c
    )


def specific_volume_m3_per_kg_da(
    temperature_c: float,
    humidity_ratio: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Specific volume of moist air in m3 / kg dry air."""

    temperature_k = temperature_c + 273.15
    return R_DA * temperature_k * (1.0 + 1.607858 * humidity_ratio) / pressure_pa


def humidity_ratio_from_specific_volume(
    temperature_c: float,
    specific_volume_m3_per_kg_da_value: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Return humidity ratio for a dry-bulb temperature and specific volume."""

    temperature_k = temperature_c + 273.15
    numerator = specific_volume_m3_per_kg_da_value * pressure_pa
    return (numerator / (R_DA * temperature_k) - 1.0) / 1.607858


def humidity_ratio_from_wet_bulb(
    dry_bulb_c: float,
    wet_bulb_c: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Approximate humidity ratio from dry-bulb and thermodynamic wet bulb."""

    dry_bulb_c = float(dry_bulb_c)
    wet_bulb_c = min(float(wet_bulb_c), dry_bulb_c)
    w_s_wb = saturation_humidity_ratio_kg_per_kg(wet_bulb_c, pressure_pa)
    numerator = (2501.0 - 2.326 * wet_bulb_c) * w_s_wb
    numerator -= 1.006 * (dry_bulb_c - wet_bulb_c)
    denominator = 2501.0 + 1.86 * dry_bulb_c - 4.186 * wet_bulb_c
    return max(numerator / denominator, 0.0)


def wet_bulb_c_from_state(
    dry_bulb_c: float,
    humidity_ratio: float,
    pressure_pa: float = SEA_LEVEL_PRESSURE_PA,
) -> float:
    """Solve approximate wet-bulb temperature from dry-bulb/W state."""

    dew_point = dew_point_c_from_humidity_ratio(humidity_ratio, pressure_pa)
    low = min(dew_point, dry_bulb_c)
    high = dry_bulb_c
    target_w = max(float(humidity_ratio), 0.0)

    for _ in range(70):
        mid = (low + high) / 2.0
        candidate_w = humidity_ratio_from_wet_bulb(dry_bulb_c, mid, pressure_pa)
        if candidate_w < target_w:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


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
        return dew_point_c_from_humidity_ratio(self.humidity_ratio, self.pressure_pa)

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
