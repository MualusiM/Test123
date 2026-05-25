"""PyQt + Matplotlib psychrometric chart explorer."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .psychrometrics import (
    PsychrometricState,
    apparatus_dew_point_c,
    coil_bypass_factor,
    enthalpy_kj_per_kg_da,
    humidity_ratio_from_enthalpy,
    humidity_ratio_from_rh,
    humidity_ratio_from_specific_volume,
    humidity_ratio_from_wet_bulb,
    relative_humidity_from_humidity_ratio,
    saturation_humidity_ratio_kg_per_kg,
    sensible_heat_ratio,
    specific_volume_m3_per_kg_da,
    standard_atmospheric_pressure_pa,
)


@dataclass
class StatePoint:
    """User-visible psychrometric chart point."""

    label: str
    dry_bulb_c: float
    relative_humidity: float
    color: str

    def state(self, pressure_pa: float) -> PsychrometricState:
        return PsychrometricState(
            self.dry_bulb_c, self.relative_humidity / 100.0, pressure_pa
        )


class PsychroChartCanvas(FigureCanvas):
    """Matplotlib canvas that renders psychrometric chart families."""

    def __init__(self, status_callback: Callable[[str], None] | None = None) -> None:
        self.figure = Figure(figsize=(11, 7), dpi=110, constrained_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.status_callback = status_callback
        self.mpl_connect("motion_notify_event", self._on_motion)

        self.points: list[StatePoint] = []
        self.temperature_min_c = -10.0
        self.temperature_max_c = 55.0
        self.max_humidity_g_per_kg = 35.0
        self.pressure_pa = standard_atmospheric_pressure_pa(0.0)
        self.show_rh = True
        self.show_wet_bulb = True
        self.show_enthalpy = True
        self.show_specific_volume = True
        self.show_comfort_zone = True
        self.show_process = True
        self.show_adp_shr = True

    def update_config(
        self,
        *,
        points: list[StatePoint],
        temperature_min_c: float,
        temperature_max_c: float,
        max_humidity_g_per_kg: float,
        pressure_pa: float,
        show_rh: bool,
        show_wet_bulb: bool,
        show_enthalpy: bool,
        show_specific_volume: bool,
        show_comfort_zone: bool,
        show_process: bool,
        show_adp_shr: bool,
    ) -> None:
        self.points = list(points)
        self.temperature_min_c = float(temperature_min_c)
        self.temperature_max_c = float(temperature_max_c)
        self.max_humidity_g_per_kg = float(max_humidity_g_per_kg)
        self.pressure_pa = float(pressure_pa)
        self.show_rh = show_rh
        self.show_wet_bulb = show_wet_bulb
        self.show_enthalpy = show_enthalpy
        self.show_specific_volume = show_specific_volume
        self.show_comfort_zone = show_comfort_zone
        self.show_process = show_process
        self.show_adp_shr = show_adp_shr
        self.draw_chart()

    def draw_chart(self) -> None:
        ax = self.axes
        ax.clear()
        ax.set_facecolor("#fbfcff")

        t_values = np.linspace(
            self.temperature_min_c, self.temperature_max_c, 700, dtype=float
        )
        saturation_w = np.array(
            [
                saturation_humidity_ratio_kg_per_kg(t, self.pressure_pa) * 1000.0
                for t in t_values
            ]
        )
        chart_ceiling = np.minimum(saturation_w, self.max_humidity_g_per_kg)

        ax.fill_between(
            t_values,
            0,
            chart_ceiling,
            color="#e9f5ff",
            alpha=0.75,
            label="Valid moist-air region",
        )
        ax.plot(
            t_values,
            saturation_w,
            color="#125c9c",
            linewidth=2.5,
            label="100% RH / saturation",
        )

        if self.show_rh:
            self._draw_relative_humidity_curves(ax, t_values)
        if self.show_enthalpy:
            self._draw_enthalpy_lines(ax, t_values)
        if self.show_wet_bulb:
            self._draw_wet_bulb_lines(ax)
        if self.show_specific_volume:
            self._draw_specific_volume_lines(ax, t_values)
        if self.show_comfort_zone:
            self._draw_comfort_zone(ax)
        if self.show_adp_shr:
            self._draw_process_analysis(ax)

        self._draw_state_points(ax)

        ax.set_xlim(self.temperature_min_c, self.temperature_max_c)
        ax.set_ylim(0, self.max_humidity_g_per_kg)
        ax.set_xlabel("Dry-bulb temperature (deg C)", fontweight="bold")
        ax.set_ylabel("Humidity ratio (g water / kg dry air)", fontweight="bold")
        ax.set_title(
            "Psychrometric Chart - PyQt + Matplotlib",
            fontsize=15,
            fontweight="bold",
            pad=14,
        )
        ax.grid(True, which="major", color="#d7dde8", linewidth=0.8)
        ax.grid(True, which="minor", color="#edf1f7", linewidth=0.5)
        ax.minorticks_on()
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.94)
        self.draw_idle()

    def _valid_curve_mask(self, t_values: np.ndarray, w_g_per_kg: np.ndarray) -> np.ndarray:
        saturation = np.array(
            [
                saturation_humidity_ratio_kg_per_kg(t, self.pressure_pa) * 1000.0
                for t in t_values
            ]
        )
        return (
            np.isfinite(w_g_per_kg)
            & (w_g_per_kg >= 0.0)
            & (w_g_per_kg <= saturation)
            & (w_g_per_kg <= self.max_humidity_g_per_kg)
        )

    def _draw_relative_humidity_curves(self, ax, t_values: np.ndarray) -> None:
        for rh in np.arange(0.1, 1.0, 0.1):
            w_values = np.array(
                [
                    humidity_ratio_from_rh(t, float(rh), self.pressure_pa) * 1000.0
                    for t in t_values
                ]
            )
            mask = self._valid_curve_mask(t_values, w_values)
            if not mask.any():
                continue
            ax.plot(
                t_values[mask],
                w_values[mask],
                color="#2379bf",
                alpha=0.55,
                linewidth=0.95,
            )
            self._label_last_visible(
                ax,
                t_values[mask],
                w_values[mask],
                f"{int(rh * 100)}%",
                color="#2379bf",
                x_offset=-1.6,
            )

    def _draw_enthalpy_lines(self, ax, t_values: np.ndarray) -> None:
        min_h = int(np.floor(enthalpy_kj_per_kg_da(self.temperature_min_c, 0) / 10) * 10)
        max_h = int(
            np.ceil(
                enthalpy_kj_per_kg_da(
                    self.temperature_max_c, self.max_humidity_g_per_kg / 1000.0
                )
                / 10
            )
            * 10
        )
        for enthalpy in range(min_h, max_h + 1, 10):
            w_values = np.array(
                [humidity_ratio_from_enthalpy(t, enthalpy) * 1000.0 for t in t_values]
            )
            mask = self._valid_curve_mask(t_values, w_values)
            if mask.sum() < 8:
                continue
            ax.plot(
                t_values[mask],
                w_values[mask],
                color="#bd6b00",
                alpha=0.35,
                linewidth=0.85,
                linestyle="-.",
            )
            idx = min(max(mask.sum() // 3, 0), mask.sum() - 1)
            ax.text(
                t_values[mask][idx],
                w_values[mask][idx],
                f"h {enthalpy}",
                color="#985600",
                fontsize=7,
                rotation=-28,
                alpha=0.75,
            )

    def _draw_wet_bulb_lines(self, ax) -> None:
        wet_bulb_values = np.arange(
            np.floor(self.temperature_min_c / 5.0) * 5.0,
            min(self.temperature_max_c, 40.0) + 0.1,
            5.0,
        )
        for wet_bulb in wet_bulb_values:
            start_t = max(self.temperature_min_c, wet_bulb)
            t_values = np.linspace(start_t, self.temperature_max_c, 260)
            if len(t_values) < 2:
                continue
            w_values = np.array(
                [
                    humidity_ratio_from_wet_bulb(t, wet_bulb, self.pressure_pa) * 1000.0
                    for t in t_values
                ]
            )
            mask = self._valid_curve_mask(t_values, w_values)
            if mask.sum() < 8:
                continue
            ax.plot(
                t_values[mask],
                w_values[mask],
                color="#7a4cc2",
                alpha=0.36,
                linewidth=0.8,
                linestyle="--",
            )
            ax.text(
                t_values[mask][0] + 0.35,
                w_values[mask][0] + 0.2,
                f"wb {wet_bulb:g}",
                color="#6a40ad",
                fontsize=7,
                alpha=0.78,
            )

    def _draw_specific_volume_lines(self, ax, t_values: np.ndarray) -> None:
        dry_min = specific_volume_m3_per_kg_da(
            self.temperature_min_c, 0.0, self.pressure_pa
        )
        humid_max = specific_volume_m3_per_kg_da(
            self.temperature_max_c, self.max_humidity_g_per_kg / 1000.0, self.pressure_pa
        )
        start = np.floor(dry_min / 0.02) * 0.02
        stop = np.ceil(humid_max / 0.02) * 0.02
        for specific_volume in np.arange(start, stop + 0.001, 0.02):
            w_values = np.array(
                [
                    humidity_ratio_from_specific_volume(
                        t, specific_volume, self.pressure_pa
                    )
                    * 1000.0
                    for t in t_values
                ]
            )
            mask = self._valid_curve_mask(t_values, w_values)
            if mask.sum() < 10:
                continue
            ax.plot(
                t_values[mask],
                w_values[mask],
                color="#008a6a",
                alpha=0.28,
                linewidth=0.75,
                linestyle=":",
            )
            self._label_last_visible(
                ax,
                t_values[mask],
                w_values[mask],
                f"v {specific_volume:.2f}",
                color="#00775c",
                x_offset=-2.5,
                fontsize=6,
            )

    def _draw_comfort_zone(self, ax) -> None:
        low_t = np.linspace(20.0, 27.0, 80)
        high_t = low_t[::-1]
        top_w = np.array(
            [humidity_ratio_from_rh(t, 0.60, self.pressure_pa) * 1000.0 for t in low_t]
        )
        bottom_w = np.array(
            [
                humidity_ratio_from_rh(t, 0.30, self.pressure_pa) * 1000.0
                for t in high_t
            ]
        )
        polygon_t = np.concatenate([low_t, high_t])
        polygon_w = np.concatenate([top_w, bottom_w])
        visible = (
            (polygon_t >= self.temperature_min_c)
            & (polygon_t <= self.temperature_max_c)
            & (polygon_w >= 0)
            & (polygon_w <= self.max_humidity_g_per_kg)
        )
        if visible.any():
            ax.fill(
                polygon_t,
                polygon_w,
                color="#68c277",
                alpha=0.22,
                label="30-60% RH comfort band",
                zorder=3,
            )
            ax.text(
                20.2,
                float(np.nanmean(polygon_w)),
                "comfort",
                color="#267236",
                fontsize=9,
                fontweight="bold",
                alpha=0.8,
            )

    def _draw_process_analysis(self, ax) -> None:
        if len(self.points) < 2:
            return

        adp_label_used = False
        shr_label_used = False
        for start, end in zip(self.points, self.points[1:]):
            start_state = start.state(self.pressure_pa)
            end_state = end.state(self.pressure_pa)
            start_w = start_state.humidity_ratio_g_per_kg
            end_w = end_state.humidity_ratio_g_per_kg
            shr = sensible_heat_ratio(start_state, end_state)

            if shr is not None:
                mid_t = (start.dry_bulb_c + end.dry_bulb_c) / 2.0
                mid_w = (start_w + end_w) / 2.0
                if (
                    self.temperature_min_c <= mid_t <= self.temperature_max_c
                    and 0.0 <= mid_w <= self.max_humidity_g_per_kg
                ):
                    ax.text(
                        mid_t,
                        mid_w,
                        f"SHR {shr:.2f}",
                        color="#8a1c7c",
                        fontsize=8,
                        fontweight="bold",
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "fc": "white",
                            "ec": "#8a1c7c",
                            "alpha": 0.75,
                        },
                        zorder=6,
                    )

            adp_c = apparatus_dew_point_c(start_state, end_state)
            if adp_c is None:
                continue

            adp_w = (
                saturation_humidity_ratio_kg_per_kg(adp_c, self.pressure_pa)
                * 1000.0
            )
            if not np.isfinite(adp_w):
                continue

            ax.plot(
                [end.dry_bulb_c, adp_c],
                [end_w, adp_w],
                color="#c2185b",
                linewidth=1.2,
                linestyle=(0, (4, 3)),
                alpha=0.74,
                label="ADP extension" if not adp_label_used else None,
                zorder=5,
            )
            adp_label_used = True

            if (
                self.temperature_min_c <= adp_c <= self.temperature_max_c
                and 0.0 <= adp_w <= self.max_humidity_g_per_kg
            ):
                ax.scatter(
                    [adp_c],
                    [adp_w],
                    marker="*",
                    s=145,
                    color="#c2185b",
                    edgecolor="white",
                    linewidth=1.0,
                    label="ADP" if not shr_label_used else None,
                    zorder=7,
                )
                shr_label_used = True
                ax.annotate(
                    f"ADP {adp_c:.1f} C",
                    xy=(adp_c, adp_w),
                    xytext=(8, -18),
                    textcoords="offset points",
                    color="#9b1749",
                    fontsize=8,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "fc": "white",
                        "ec": "#c2185b",
                        "alpha": 0.82,
                    },
                    zorder=8,
                )

    def _draw_state_points(self, ax) -> None:
        previous: tuple[float, float] | None = None
        for point in self.points:
            state = point.state(self.pressure_pa)
            w = state.humidity_ratio_g_per_kg
            if w > self.max_humidity_g_per_kg:
                continue

            ax.scatter(
                [point.dry_bulb_c],
                [w],
                s=82,
                color=point.color,
                edgecolor="white",
                linewidth=1.4,
                zorder=8,
            )
            ax.annotate(
                point.label,
                xy=(point.dry_bulb_c, w),
                xytext=(8, 8),
                textcoords="offset points",
                color=point.color,
                fontsize=9,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "white",
                    "ec": point.color,
                    "alpha": 0.86,
                },
                zorder=9,
            )
            if self.show_process and previous is not None:
                ax.annotate(
                    "",
                    xy=(point.dry_bulb_c, w),
                    xytext=previous,
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#202733",
                        "lw": 2.0,
                        "shrinkA": 7,
                        "shrinkB": 7,
                        "alpha": 0.75,
                    },
                    zorder=7,
                )
            previous = (point.dry_bulb_c, w)

    def _label_last_visible(
        self,
        ax,
        x_values: np.ndarray,
        y_values: np.ndarray,
        label: str,
        *,
        color: str,
        x_offset: float = 0.0,
        fontsize: int = 7,
    ) -> None:
        if len(x_values) == 0:
            return
        idx = -1
        ax.text(
            x_values[idx] + x_offset,
            y_values[idx],
            label,
            color=color,
            fontsize=fontsize,
            alpha=0.82,
            va="center",
        )

    def _on_motion(self, event) -> None:
        if event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            return
        humidity_ratio = max(event.ydata / 1000.0, 0.0)
        rh = relative_humidity_from_humidity_ratio(
            event.xdata, humidity_ratio, self.pressure_pa
        )
        h = enthalpy_kj_per_kg_da(event.xdata, humidity_ratio)
        status = (
            f"Tdb {event.xdata:5.1f} deg C | W {event.ydata:5.1f} g/kg | "
            f"RH {rh * 100:5.1f}% | h {h:5.1f} kJ/kg_da"
        )
        if self.status_callback is not None:
            self.status_callback(status)


class MainWindow(QMainWindow):
    """Main application window."""

    COLORS = [
        "#e53935",
        "#1e88e5",
        "#43a047",
        "#fb8c00",
        "#8e24aa",
        "#00acc1",
        "#6d4c41",
        "#3949ab",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Psychrometric Chart Lab")
        self.resize(1500, 940)
        self.points: list[StatePoint] = []

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        self.canvas = PsychroChartCanvas(self.statusBar().showMessage)
        root_layout.addWidget(self._build_control_panel(), 0)
        root_layout.addWidget(self.canvas, 1)

        self._seed_demo_points()
        self._sync_all()

    def _build_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(385)
        scroll.setMaximumWidth(440)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        layout.addWidget(self._build_atmosphere_group())
        layout.addWidget(self._build_chart_group())
        layout.addWidget(self._build_layers_group())
        layout.addWidget(self._build_point_group())
        layout.addWidget(self._build_table_group())
        layout.addWidget(self._build_report_group())
        layout.addStretch(1)

        scroll.setWidget(panel)
        return scroll

    def _build_atmosphere_group(self) -> QGroupBox:
        group = QGroupBox("Atmosphere")
        form = QFormLayout(group)

        self.altitude_spin = QDoubleSpinBox()
        self.altitude_spin.setRange(-500.0, 5000.0)
        self.altitude_spin.setSingleStep(100.0)
        self.altitude_spin.setSuffix(" m")
        self.altitude_spin.valueChanged.connect(self._sync_all)

        self.pressure_label = QLabel()
        self.pressure_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        form.addRow("Altitude", self.altitude_spin)
        form.addRow("Pressure", self.pressure_label)
        return group

    def _build_chart_group(self) -> QGroupBox:
        group = QGroupBox("Chart range")
        grid = QGridLayout(group)

        self.temp_min_spin = self._make_double_spin(-40.0, 80.0, -10.0, " deg C")
        self.temp_max_spin = self._make_double_spin(-20.0, 100.0, 55.0, " deg C")
        self.max_w_spin = self._make_double_spin(5.0, 80.0, 35.0, " g/kg")

        self.temp_min_spin.valueChanged.connect(self._sync_all)
        self.temp_max_spin.valueChanged.connect(self._sync_all)
        self.max_w_spin.valueChanged.connect(self._sync_all)

        grid.addWidget(QLabel("Min dry bulb"), 0, 0)
        grid.addWidget(self.temp_min_spin, 0, 1)
        grid.addWidget(QLabel("Max dry bulb"), 1, 0)
        grid.addWidget(self.temp_max_spin, 1, 1)
        grid.addWidget(QLabel("Max humidity"), 2, 0)
        grid.addWidget(self.max_w_spin, 2, 1)
        return group

    def _build_layers_group(self) -> QGroupBox:
        group = QGroupBox("Layers")
        grid = QGridLayout(group)
        self.rh_check = self._make_check("RH curves", True)
        self.wet_bulb_check = self._make_check("Wet-bulb lines", True)
        self.enthalpy_check = self._make_check("Enthalpy lines", True)
        self.volume_check = self._make_check("Specific volume", True)
        self.comfort_check = self._make_check("Comfort band", True)
        self.process_check = self._make_check("Process arrows", True)
        self.adp_shr_check = self._make_check("ADP + SHR", True)

        checks = [
            self.rh_check,
            self.wet_bulb_check,
            self.enthalpy_check,
            self.volume_check,
            self.comfort_check,
            self.process_check,
            self.adp_shr_check,
        ]
        for index, check in enumerate(checks):
            check.stateChanged.connect(self._sync_all)
            grid.addWidget(check, index // 2, index % 2)
        return group

    def _build_point_group(self) -> QGroupBox:
        group = QGroupBox("State point lab")
        layout = QGridLayout(group)

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("e.g. Room, Coil leaving, Mars lobby")
        self.point_temp_spin = self._make_double_spin(-40.0, 90.0, 24.0, " deg C")
        self.point_rh_spin = self._make_double_spin(0.0, 100.0, 50.0, " %")

        add_button = QPushButton("Add point")
        add_button.clicked.connect(self._add_point)
        update_button = QPushButton("Update selected")
        update_button.clicked.connect(self._update_selected_point)
        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self._remove_selected_point)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear_points)
        demo_button = QPushButton("Reset demo")
        demo_button.clicked.connect(self._reset_demo)
        export_button = QPushButton("Export PNG")
        export_button.clicked.connect(self._export_png)

        layout.addWidget(QLabel("Label"), 0, 0)
        layout.addWidget(self.label_edit, 0, 1, 1, 2)
        layout.addWidget(QLabel("Dry bulb"), 1, 0)
        layout.addWidget(self.point_temp_spin, 1, 1, 1, 2)
        layout.addWidget(QLabel("Relative humidity"), 2, 0)
        layout.addWidget(self.point_rh_spin, 2, 1, 1, 2)
        layout.addWidget(add_button, 3, 0)
        layout.addWidget(update_button, 3, 1)
        layout.addWidget(remove_button, 3, 2)
        layout.addWidget(clear_button, 4, 0)
        layout.addWidget(demo_button, 4, 1)
        layout.addWidget(export_button, 4, 2)
        return group

    def _build_table_group(self) -> QGroupBox:
        group = QGroupBox("State table")
        layout = QVBoxLayout(group)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Label", "Tdb", "RH", "W", "h", "Tdp", "Twb", "v"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._load_selected_point)
        layout.addWidget(self.table)
        return group

    def _build_report_group(self) -> QGroupBox:
        group = QGroupBox("Live process report")
        layout = QVBoxLayout(group)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(230)
        layout.addWidget(self.report)
        return group

    def _make_double_spin(
        self, minimum: float, maximum: float, value: float, suffix: str
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def _make_check(self, text: str, checked: bool) -> QCheckBox:
        check = QCheckBox(text)
        check.setChecked(checked)
        return check

    def _pressure_pa(self) -> float:
        return standard_atmospheric_pressure_pa(self.altitude_spin.value())

    def _seed_demo_points(self) -> None:
        self.points = [
            StatePoint("Outdoor", 32.0, 62.0, self.COLORS[0]),
            StatePoint("Mixed air", 27.0, 54.0, self.COLORS[1]),
            StatePoint("Coil leaving", 13.0, 92.0, self.COLORS[2]),
            StatePoint("Room", 24.0, 50.0, self.COLORS[3]),
        ]

    def _sync_all(self) -> None:
        if not hasattr(self, "canvas"):
            return

        if self.temp_min_spin.value() >= self.temp_max_spin.value():
            self.temp_max_spin.blockSignals(True)
            self.temp_max_spin.setValue(self.temp_min_spin.value() + 1.0)
            self.temp_max_spin.blockSignals(False)

        pressure = self._pressure_pa()
        self.pressure_label.setText(f"{pressure / 1000.0:.2f} kPa")
        self.canvas.update_config(
            points=self.points,
            temperature_min_c=self.temp_min_spin.value(),
            temperature_max_c=self.temp_max_spin.value(),
            max_humidity_g_per_kg=self.max_w_spin.value(),
            pressure_pa=pressure,
            show_rh=self.rh_check.isChecked(),
            show_wet_bulb=self.wet_bulb_check.isChecked(),
            show_enthalpy=self.enthalpy_check.isChecked(),
            show_specific_volume=self.volume_check.isChecked(),
            show_comfort_zone=self.comfort_check.isChecked(),
            show_process=self.process_check.isChecked(),
            show_adp_shr=self.adp_shr_check.isChecked(),
        )
        self._refresh_table()
        self._refresh_report()

    def _add_point(self) -> None:
        label = self.label_edit.text().strip() or f"Point {len(self.points) + 1}"
        color = self.COLORS[len(self.points) % len(self.COLORS)]
        self.points.append(
            StatePoint(
                label=label,
                dry_bulb_c=self.point_temp_spin.value(),
                relative_humidity=self.point_rh_spin.value(),
                color=color,
            )
        )
        self.label_edit.clear()
        self._sync_all()

    def _selected_row(self) -> int | None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return None
        row = selected[0].row()
        if row < 0 or row >= len(self.points):
            return None
        return row

    def _update_selected_point(self) -> None:
        row = self._selected_row()
        if row is None:
            QMessageBox.information(self, "No selection", "Select a state point first.")
            return
        label = self.label_edit.text().strip() or self.points[row].label
        self.points[row] = StatePoint(
            label=label,
            dry_bulb_c=self.point_temp_spin.value(),
            relative_humidity=self.point_rh_spin.value(),
            color=self.points[row].color,
        )
        self._sync_all()
        self.table.selectRow(row)

    def _remove_selected_point(self) -> None:
        row = self._selected_row()
        if row is None:
            QMessageBox.information(self, "No selection", "Select a state point first.")
            return
        del self.points[row]
        self._sync_all()

    def _clear_points(self) -> None:
        self.points = []
        self._sync_all()

    def _reset_demo(self) -> None:
        self._seed_demo_points()
        self._sync_all()

    def _load_selected_point(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        point = self.points[row]
        self.label_edit.setText(point.label)
        self.point_temp_spin.setValue(point.dry_bulb_c)
        self.point_rh_spin.setValue(point.relative_humidity)

    def _refresh_table(self) -> None:
        pressure = self._pressure_pa()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.points))
        for row, point in enumerate(self.points):
            state = point.state(pressure)
            values = [
                point.label,
                f"{state.dry_bulb_c:.1f} C",
                f"{state.relative_humidity * 100.0:.1f}%",
                f"{state.humidity_ratio_g_per_kg:.2f} g/kg",
                f"{state.enthalpy_kj_per_kg_da:.1f}",
                f"{state.dew_point_c:.1f} C",
                f"{state.wet_bulb_c:.1f} C",
                f"{state.specific_volume_m3_per_kg_da:.3f}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                    if column
                    else Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)

    def _refresh_report(self) -> None:
        pressure = self._pressure_pa()
        lines = [
            "Psychrometric process report",
            f"Pressure: {pressure / 1000.0:.2f} kPa",
            "",
        ]

        for index, point in enumerate(self.points, start=1):
            state = point.state(pressure)
            lines.extend(
                [
                    f"{index}. {point.label}",
                    f"   Tdb {state.dry_bulb_c:.1f} C | RH {state.relative_humidity * 100:.1f}% | W {state.humidity_ratio_g_per_kg:.2f} g/kg",
                    f"   h {state.enthalpy_kj_per_kg_da:.1f} kJ/kg_da | Tdp {state.dew_point_c:.1f} C | Twb {state.wet_bulb_c:.1f} C | v {state.specific_volume_m3_per_kg_da:.3f} m3/kg_da",
                ]
            )

        if len(self.points) >= 2:
            lines.extend(["", "Consecutive process legs"])
        for start, end in zip(self.points, self.points[1:]):
            start_state = start.state(pressure)
            end_state = end.state(pressure)
            delta_w = end_state.humidity_ratio_g_per_kg - start_state.humidity_ratio_g_per_kg
            delta_h = end_state.enthalpy_kj_per_kg_da - start_state.enthalpy_kj_per_kg_da
            delta_t = end_state.dry_bulb_c - start_state.dry_bulb_c
            shr = sensible_heat_ratio(start_state, end_state)
            shr_text = "n/a" if shr is None else f"{shr:.2f}"
            adp_c = apparatus_dew_point_c(start_state, end_state)
            if adp_c is None:
                adp_text = "ADP n/a"
                bypass_text = "BF n/a"
            else:
                adp_w = (
                    saturation_humidity_ratio_kg_per_kg(adp_c, pressure) * 1000.0
                )
                bypass_factor = coil_bypass_factor(start_state, end_state, adp_c)
                bypass_text = (
                    "BF n/a"
                    if bypass_factor is None
                    else f"BF {bypass_factor:.2f}"
                )
                adp_text = f"ADP {adp_c:.1f} C / {adp_w:.2f} g/kg"
            mode = self._classify_process(delta_t, delta_w)
            lines.append(
                f"   {start.label} -> {end.label}: {mode}; "
                f"dT {delta_t:+.1f} C, dW {delta_w:+.2f} g/kg, "
                f"dh {delta_h:+.1f} kJ/kg_da, SHR {shr_text}, "
                f"{adp_text}, {bypass_text}"
            )

        self.report.setPlainText("\n".join(lines))

    def _classify_process(self, delta_t: float, delta_w: float) -> str:
        temp_word = "heating" if delta_t > 0.05 else "cooling" if delta_t < -0.05 else "isothermal"
        moisture_word = (
            "humidifying"
            if delta_w > 0.05
            else "dehumidifying"
            if delta_w < -0.05
            else "constant humidity"
        )
        return f"{temp_word}, {moisture_word}"

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export psychrometric chart",
            "psychrometric_chart.png",
            "PNG image (*.png);;SVG vector (*.svg);;PDF document (*.pdf)",
        )
        if not path:
            return
        self.canvas.figure.savefig(path, dpi=220, bbox_inches="tight")
        self.statusBar().showMessage(f"Saved chart to {path}", 7000)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
