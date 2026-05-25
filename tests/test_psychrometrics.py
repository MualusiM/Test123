import unittest

import psychrolib

from psychro_chart_app.psychrometrics import (
    PsychrometricState,
    apparatus_dew_point_c,
    coil_bypass_factor,
    enthalpy_kj_per_kg_da,
    humidity_ratio_from_rh,
    process_line_humidity_ratio,
    relative_humidity_from_humidity_ratio,
    saturation_humidity_ratio_kg_per_kg,
    saturation_vapor_pressure_pa,
    standard_atmospheric_pressure_pa,
    wet_bulb_c_from_state,
)


class PsychrometricFormulaTests(unittest.TestCase):
    def test_standard_pressure_at_sea_level_uses_psychrolib(self):
        self.assertAlmostEqual(
            standard_atmospheric_pressure_pa(0),
            psychrolib.GetStandardAtmPressure(0),
            delta=1.0e-9,
        )

    def test_saturation_pressure_uses_psychrolib(self):
        self.assertAlmostEqual(
            saturation_vapor_pressure_pa(20),
            psychrolib.GetSatVapPres(20),
            delta=1.0e-9,
        )

    def test_relative_humidity_round_trip(self):
        pressure = standard_atmospheric_pressure_pa(500)
        humidity_ratio = humidity_ratio_from_rh(25.0, 0.55, pressure)
        rh = relative_humidity_from_humidity_ratio(25.0, humidity_ratio, pressure)
        self.assertAlmostEqual(rh, 0.55, places=6)

    def test_humidity_ratio_uses_psychrolib(self):
        pressure = standard_atmospheric_pressure_pa(250)
        self.assertAlmostEqual(
            humidity_ratio_from_rh(28.0, 0.6, pressure),
            psychrolib.GetHumRatioFromRelHum(28.0, 0.6, pressure),
            delta=1.0e-12,
        )

    def test_enthalpy_converts_psychrolib_joules_to_kilojoules(self):
        humidity_ratio = humidity_ratio_from_rh(24.0, 0.50)
        self.assertAlmostEqual(
            enthalpy_kj_per_kg_da(24.0, humidity_ratio),
            psychrolib.GetMoistAirEnthalpy(24.0, humidity_ratio) / 1000.0,
            delta=1.0e-9,
        )

    def test_state_properties_are_plausible(self):
        state = PsychrometricState(24.0, 0.50)
        self.assertGreater(state.humidity_ratio_g_per_kg, 8.0)
        self.assertLess(state.humidity_ratio_g_per_kg, 10.0)
        self.assertGreater(state.dew_point_c, 12.0)
        self.assertLess(state.dew_point_c, 14.5)
        self.assertGreater(state.enthalpy_kj_per_kg_da, 45.0)
        self.assertLess(state.enthalpy_kj_per_kg_da, 50.0)


    def test_process_line_humidity_ratio_interpolates_between_states(self):
        start = PsychrometricState(30.0, 0.50)
        end = PsychrometricState(20.0, 0.70)
        midpoint = process_line_humidity_ratio(25.0, start, end)
        expected = (start.humidity_ratio + end.humidity_ratio) / 2.0
        self.assertAlmostEqual(midpoint, expected, delta=1.0e-12)

    def test_apparatus_dew_point_and_bypass_factor(self):
        pressure = standard_atmospheric_pressure_pa(0)
        start = PsychrometricState(30.0, 0.50, pressure)
        adp_c = 10.0
        bypass_factor = 0.25
        adp_w = saturation_humidity_ratio_kg_per_kg(adp_c, pressure)
        end_t = adp_c + bypass_factor * (start.dry_bulb_c - adp_c)
        end_w = adp_w + bypass_factor * (start.humidity_ratio - adp_w)
        end_rh = relative_humidity_from_humidity_ratio(end_t, end_w, pressure)
        end = PsychrometricState(end_t, end_rh, pressure)

        calculated_adp = apparatus_dew_point_c(start, end)

        self.assertIsNotNone(calculated_adp)
        self.assertAlmostEqual(calculated_adp, adp_c, places=4)
        self.assertAlmostEqual(
            coil_bypass_factor(start, end, calculated_adp),
            bypass_factor,
            places=4,
        )

    def test_wet_bulb_is_between_dew_point_and_dry_bulb(self):
        state = PsychrometricState(30.0, 0.45)
        wet_bulb = wet_bulb_c_from_state(
            state.dry_bulb_c, state.humidity_ratio, state.pressure_pa
        )
        self.assertGreaterEqual(wet_bulb, state.dew_point_c - 0.05)
        self.assertLessEqual(wet_bulb, state.dry_bulb_c + 0.05)


if __name__ == "__main__":
    unittest.main()
