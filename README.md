# Psychrometric Chart Lab

A desktop psychrometric chart app built with PyQt, Matplotlib, and PsychroLib.
It plots the classic HVAC chart families, lets you add state points, and computes
live moist air properties as you explore.

## Features

- Saturation boundary and relative humidity curves
- Enthalpy, wet-bulb, and specific-volume guide lines
- Altitude-adjusted standard atmospheric pressure
- PsychroLib-backed SI psychrometric calculations
- Comfort-zone overlay
- Add, update, remove, and annotate state points
- Process arrows between consecutive points
- Apparatus dew point (ADP), sensible heat ratio (SHR), and bypass factor reporting
- Live table for humidity ratio, enthalpy, dew point, wet bulb, and specific volume
- Cursor readout for dry bulb, humidity ratio, RH, and enthalpy
- PNG/SVG/PDF export from the running app

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
psychro-chart
```

You can also launch it with:

```bash
python -m psychro_chart_app
```

## Test the calculation layer

```bash
python -m unittest discover -s tests
```

The GUI uses SI units throughout: deg C, Pa/kPa, kg or g water per kg dry air,
kJ/kg dry air, and m3/kg dry air.