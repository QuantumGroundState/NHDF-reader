"""
GUI application to read and save NHDF (Nion HDF5) data and metadata files.

Required packages:
  pip install numpy h5py niondata matplotlib hyperspy
  conda install -c conda-forge numpy h5py niondata matplotlib hyperspy
"""

import h5py
import json
import numpy
import numpy.typing
import pathlib
import pprint
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import typing

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.utils import Converter

_NDArray = numpy.typing.NDArray[typing.Any]
PHI = 1.618033988749895

try:
    import hyperspy.api as hs
    HAS_HYPERSPY = True
except ImportError:
    HAS_HYPERSPY = False


def read_data_and_metadata(path: pathlib.Path) -> DataAndMetadata.DataAndMetadata:
    with h5py.File(str(path), "r") as f:
        dg = f["data"]
        key0 = list(sorted(dg.keys()))[0]
        ds = dg[key0]
        json_properties = json.loads(ds.attrs["properties"])
        data = numpy.array(ds)
        data_descriptor = DataAndMetadata.DataDescriptor(
            is_sequence=json_properties.get("is_sequence", False),
            collection_dimension_count=json_properties.get("collection_dimension_count", 0),
            datum_dimension_count=json_properties.get("datum_dimension_count", 0),
        )
        data_metadata = DataAndMetadata.DataMetadata(
            data_shape_and_dtype=(data.shape, data.dtype),
            intensity_calibration=Calibration.Calibration.from_rpc_dict(
                json_properties.get("intensity_calibration", {})
            ),
            dimensional_calibrations=[
                typing.cast(
                    Calibration.Calibration, Calibration.Calibration.from_rpc_dict(d)
                )
                for d in json_properties.get("dimensional_calibrations", [])
            ],
            metadata=json_properties.get("metadata", {}),
            timestamp=Converter.DatetimeToStringConverter().convert_back(
                json_properties.get("created", "")
            ),
            data_descriptor=data_descriptor,
            timezone=json_properties.get("timezone", None),
            timezone_offset=json_properties.get("timezone_offset", None),
        )
        return DataAndMetadata.DataAndMetadata(
            lambda: data,
            data_shape_and_dtype=data_metadata.data_shape_and_dtype,
            intensity_calibration=data_metadata.intensity_calibration,
            dimensional_calibrations=data_metadata.dimensional_calibrations,
            metadata=data_metadata.metadata,
            timestamp=data_metadata.timestamp,
            data_descriptor=data_metadata.data_descriptor,
            timezone=data_metadata.timezone,
            timezone_offset=data_metadata.timezone_offset,
        )


def save_data_and_metadata(
    path: pathlib.Path, d: DataAndMetadata.DataAndMetadata
) -> None:
    properties: typing.Dict[str, typing.Any] = {}
    properties["is_sequence"] = d.data_descriptor.is_sequence
    properties["collection_dimension_count"] = (
        d.data_descriptor.collection_dimension_count
    )
    properties["datum_dimension_count"] = d.data_descriptor.datum_dimension_count
    if d.intensity_calibration:
        properties["intensity_calibration"] = d.intensity_calibration.rpc_dict
    if d.dimensional_calibrations:
        properties["dimensional_calibrations"] = [
            cal.rpc_dict for cal in d.dimensional_calibrations
        ]
    if d.metadata:
        properties["metadata"] = d.metadata
    if d.timestamp:
        properties["created"] = Converter.DatetimeToStringConverter().convert(
            d.timestamp
        )
    if d.timezone:
        properties["timezone"] = d.timezone
    if d.timezone_offset:
        properties["timezone_offset"] = d.timezone_offset

    with h5py.File(str(path), "w") as f:
        dg = f.create_group("data")
        ds = dg.create_dataset("0", data=d.data)
        ds.attrs["properties"] = json.dumps(properties)


def export_dm3(path: pathlib.Path, d: DataAndMetadata.DataAndMetadata) -> None:
    if not HAS_HYPERSPY:
        raise RuntimeError("hyperspy is not installed.")
    data = d.data
    cals = d.dimensional_calibrations
    if data.ndim == 3:
        sig = hs.signals.Signal1D(data)
    elif data.ndim == 2:
        sig = hs.signals.Signal2D(data)
    elif data.ndim == 1:
        sig = hs.signals.Signal1D(data)
    else:
        sig = hs.signals.BaseSignal(data)
    all_axes = list(sig.axes_manager._axes)
    for i, ax in enumerate(all_axes):
        if cals and i < len(cals):
            c = cals[i]
            ax.offset = c.offset if c.offset else 0.0
            ax.scale = c.scale if c.scale else 1.0
            ax.units = c.units if c.units else ""
            if i < data.ndim - 1:
                ax.name = f"Dim {i} (spatial)"
            else:
                ax.name = "Energy"
    sig.save(str(path), overwrite=True)


def format_info(d: DataAndMetadata.DataAndMetadata) -> str:
    lines = [
        f"Shape: {d.data.shape}",
        f"Dtype: {d.data.dtype}",
        f"Data Descriptor: {d.data_descriptor}",
        f"  is_sequence: {d.data_descriptor.is_sequence}",
        f"  collection_dimension_count: {d.data_descriptor.collection_dimension_count}",
        f"  datum_dimension_count: {d.data_descriptor.datum_dimension_count}",
        f"Intensity Calibration: {d.intensity_calibration}",
        f"Dimensional Calibrations: {d.dimensional_calibrations}",
        f"Timestamp: {d.timestamp}",
        f"Timezone: {d.timezone}",
        f"Timezone Offset: {d.timezone_offset}",
        "",
        "Metadata:",
        pprint.pformat(d.metadata),
    ]
    return "\n".join(lines)


def _get_cal(d: DataAndMetadata.DataAndMetadata, dim: int
             ) -> typing.Tuple[float, float, str]:
    cals = d.dimensional_calibrations
    if cals and dim < len(cals):
        c = cals[dim]
        return (c.offset if c.offset else 0.0,
                c.scale if c.scale else 1.0,
                c.units if c.units else "")
    return 0.0, 1.0, ""


def _get_spectral_cal(d: DataAndMetadata.DataAndMetadata) -> typing.Tuple[numpy.ndarray, str]:
    cals = d.dimensional_calibrations
    if cals and len(cals) >= 3:
        cal = cals[2]
    elif cals and len(cals) >= 1:
        cal = cals[-1]
    else:
        cal = None
    n = d.data.shape[-1]
    if cal is not None:
        offset = cal.offset if cal.offset else 0.0
        scale = cal.scale if cal.scale else 1.0
        units = cal.units if cal.units else ""
        x = offset + scale * numpy.arange(n)
    else:
        x = numpy.arange(n, dtype=float)
        units = ""
    return x, units


def _energy_index_for_zero(d: DataAndMetadata.DataAndMetadata) -> int:
    x, _ = _get_spectral_cal(d)
    return int(numpy.argmin(numpy.abs(x)))


def _compute_fwhm(x: numpy.ndarray, y: numpy.ndarray) -> float:
    peak_val = numpy.max(y)
    half_max = peak_val / 2.0
    above = y >= half_max
    indices = numpy.where(above)[0]
    if len(indices) < 2:
        return 0.0
    return float(x[indices[-1]] - x[indices[0]])


class NHDFApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NHDF Reader / Writer")
        w = 1100
        h = int(w / PHI) + 600
        self.root.geometry(f"{w}x{h}")
        self.current_data: typing.Optional[DataAndMetadata.DataAndMetadata] = None
        self.current_path: typing.Optional[pathlib.Path] = None
        self._x_cal: typing.Optional[numpy.ndarray] = None
        self._spectrum: typing.Optional[numpy.ndarray] = None
        self._spec_units = ""
        self._colorbar = None
        self._cursor_ix = 0
        self._cursor_iy = 0
        self._build_ui()

    def _build_ui(self) -> None:
        # ---- scrollable wrapper ----
        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)
        cvs = tk.Canvas(container)
        vsb = tk.Scrollbar(container, orient=tk.VERTICAL, command=cvs.yview)
        self._inner = tk.Frame(cvs)
        self._inner.bind("<Configure>", lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
        cvs.create_window((0, 0), window=self._inner, anchor="nw")
        cvs.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        def _on_mw(event: tk.Event) -> None:
            cvs.yview_scroll(int(-1 * (event.delta / 120)), "units")
        cvs.bind_all("<MouseWheel>", _on_mw)
        outer = self._inner

        # ---- toolbar ----
        toolbar = tk.Frame(outer)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        tk.Button(toolbar, text="Open NHDF…", command=self._open_file, width=14
                  ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(toolbar, text="Save NHDF…", command=self._save_file, width=14
                  ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(toolbar, text="Export CSV…", command=self._export_csv, width=14
                  ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(toolbar, text="Export Sum Spectrum…", command=self._export_sum_spectrum, width=18
                  ).pack(side=tk.LEFT, padx=(0, 4))
        self.btn_dm3 = tk.Button(toolbar, text="Export DM3…", command=self._export_dm3,
                                 width=12, state=tk.NORMAL if HAS_HYPERSPY else tk.DISABLED)
        self.btn_dm3.pack(side=tk.LEFT, padx=(0, 4))
        if not HAS_HYPERSPY:
            tk.Label(toolbar, text="(dm3: install hyperspy)", fg="red", font=("", 8)
                     ).pack(side=tk.LEFT)

        # ---- file label ----
        self.file_label = tk.Label(outer, text="No file loaded", anchor="w", fg="gray")
        self.file_label.pack(fill=tk.X, padx=10)

        # ---- info boxes ----
        info_row = tk.Frame(outer)
        info_row.pack(fill=tk.X, padx=8, pady=(4, 2))
        info_row.columnconfigure((0, 1, 2, 3), weight=1)

        def make_info_box(parent: tk.Frame, label: str, col: int) -> tk.Label:
            frm = tk.LabelFrame(parent, text=label, padx=6, pady=2)
            frm.grid(row=0, column=col, padx=4, sticky="nsew")
            val = tk.Label(frm, text="—", font=("Courier", 11, "bold"), anchor="center")
            val.pack(fill=tk.X)
            return val

        self.lbl_fwhm = make_info_box(info_row, "FWHM (spectrum)", 0)
        self.lbl_bin_spectrum = make_info_box(info_row, "Bin size (spectrum)", 1)
        self.lbl_bin_x = make_info_box(info_row, "Bin size (x / dim 0)", 2)
        self.lbl_bin_y = make_info_box(info_row, "Bin size (y / dim 1)", 3)

        # ============================================================
        # ROW 1: two summed-spectrum plots side-by-side
        # ============================================================
        row1 = tk.Frame(outer)
        row1.pack(fill=tk.X, padx=8, pady=(4, 0))
        row1.columnconfigure((0, 1), weight=1)

        # -- LEFT: full-range --
        left_col = tk.LabelFrame(row1, text="Full-range plot", padx=4, pady=2)
        left_col.grid(row=0, column=0, padx=(0, 4), sticky="nsew")
        self.fig_full = Figure(figsize=(4.8, 2.6), dpi=100)
        self.ax_full = self.fig_full.add_subplot(111)
        self.canvas_full = FigureCanvasTkAgg(self.fig_full, master=left_col)
        self.canvas_full.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        cl = tk.Frame(left_col); cl.pack(fill=tk.X, pady=(2, 0))
        tk.Label(cl, text="X %").pack(side=tk.LEFT)
        self.sl_full_x = tk.Scale(cl, from_=1, to=100, orient=tk.HORIZONTAL,
                                  command=lambda v: self._on_slider(), length=100)
        self.sl_full_x.set(100); self.sl_full_x.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(cl, text="Y %").pack(side=tk.LEFT)
        self.sl_full_y = tk.Scale(cl, from_=1, to=100, orient=tk.HORIZONTAL,
                                  command=lambda v: self._on_slider(), length=100)
        self.sl_full_y.set(100); self.sl_full_y.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.var_full_scale = tk.StringVar(value="log")
        rl = tk.Frame(left_col); rl.pack(fill=tk.X, pady=(0, 2))
        tk.Label(rl, text="Y scale:").pack(side=tk.LEFT, padx=(0, 4))
        tk.Radiobutton(rl, text="Log", variable=self.var_full_scale,
                       value="log", command=self._on_slider).pack(side=tk.LEFT)
        tk.Radiobutton(rl, text="Linear", variable=self.var_full_scale,
                       value="linear", command=self._on_slider).pack(side=tk.LEFT)

        # -- RIGHT: 1/30-max capped --
        right_col = tk.LabelFrame(row1, text="1/30-max capped plot", padx=4, pady=2)
        right_col.grid(row=0, column=1, padx=(4, 0), sticky="nsew")
        self.fig_zoom = Figure(figsize=(4.8, 2.6), dpi=100)
        self.ax_zoom = self.fig_zoom.add_subplot(111)
        self.canvas_zoom = FigureCanvasTkAgg(self.fig_zoom, master=right_col)
        self.canvas_zoom.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        cr = tk.Frame(right_col); cr.pack(fill=tk.X, pady=(2, 0))
        tk.Label(cr, text="X %").pack(side=tk.LEFT)
        self.sl_zoom_x = tk.Scale(cr, from_=1, to=100, orient=tk.HORIZONTAL,
                                  command=lambda v: self._on_slider(), length=100)
        self.sl_zoom_x.set(100); self.sl_zoom_x.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(cr, text="Y %").pack(side=tk.LEFT)
        self.sl_zoom_y = tk.Scale(cr, from_=1, to=100, orient=tk.HORIZONTAL,
                                  command=lambda v: self._on_slider(), length=100)
        self.sl_zoom_y.set(100); self.sl_zoom_y.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.var_zoom_scale = tk.StringVar(value="linear")
        rr = tk.Frame(right_col); rr.pack(fill=tk.X, pady=(0, 2))
        tk.Label(rr, text="Y scale:").pack(side=tk.LEFT, padx=(0, 4))
        tk.Radiobutton(rr, text="Log", variable=self.var_zoom_scale,
                       value="log", command=self._on_slider).pack(side=tk.LEFT)
        tk.Radiobutton(rr, text="Linear", variable=self.var_zoom_scale,
                       value="linear", command=self._on_slider).pack(side=tk.LEFT)

        # ============================================================
        # ROW 2: 3D spatial image – full width
        # ============================================================
        img_frame = tk.LabelFrame(outer, text="Spatial image (energy slice) — click to place cursor",
                                  padx=6, pady=4)
        img_frame.pack(fill=tk.X, padx=8, pady=(6, 0))

        slice_ctrl = tk.Frame(img_frame)
        slice_ctrl.pack(fill=tk.X, pady=(0, 2))
        tk.Label(slice_ctrl, text="Energy slice:").pack(side=tk.LEFT, padx=(0, 4))
        self.lbl_energy_val = tk.Label(slice_ctrl, text="—", font=("Courier", 10, "bold"),
                                       width=20, anchor="center", relief=tk.GROOVE)
        self.lbl_energy_val.pack(side=tk.LEFT, padx=(0, 8))
        self.sl_energy = tk.Scale(slice_ctrl, from_=0, to=0, orient=tk.HORIZONTAL,
                                  command=self._on_energy_slider, length=500)
        self.sl_energy.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(slice_ctrl, text="Cursor:").pack(side=tk.LEFT, padx=(12, 4))
        self.lbl_cursor_pos = tk.Label(slice_ctrl, text="—", font=("Courier", 10, "bold"),
                                       anchor="center", relief=tk.GROOVE, width=34)
        self.lbl_cursor_pos.pack(side=tk.LEFT)

        self.fig_img = Figure(figsize=(9.5, 3.5), dpi=100)
        self.ax_img = self.fig_img.add_subplot(111)
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=img_frame)
        self.canvas_img.get_tk_widget().pack(fill=tk.X, expand=False)
        self.canvas_img.mpl_connect("button_press_event", self._on_img_click)

        # ============================================================
        # ROW 3: point spectrum – full width, with sliders + radio
        # ============================================================
        pt_frame = tk.LabelFrame(outer, text="Point spectrum (from cursor)", padx=6, pady=4)
        pt_frame.pack(fill=tk.X, padx=8, pady=(6, 0))

        self.fig_pt = Figure(figsize=(9.5, 2.8), dpi=100)
        self.ax_pt = self.fig_pt.add_subplot(111)
        self.canvas_pt = FigureCanvasTkAgg(self.fig_pt, master=pt_frame)
        self.canvas_pt.get_tk_widget().pack(fill=tk.X, expand=False)

        cpt = tk.Frame(pt_frame); cpt.pack(fill=tk.X, pady=(2, 0))
        tk.Label(cpt, text="X %").pack(side=tk.LEFT)
        self.sl_pt_x = tk.Scale(cpt, from_=1, to=100, orient=tk.HORIZONTAL,
                                command=lambda v: self._draw_point_spectrum(), length=140)
        self.sl_pt_x.set(100); self.sl_pt_x.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(cpt, text="Y %").pack(side=tk.LEFT)
        self.sl_pt_y = tk.Scale(cpt, from_=1, to=100, orient=tk.HORIZONTAL,
                                command=lambda v: self._draw_point_spectrum(), length=140)
        self.sl_pt_y.set(100); self.sl_pt_y.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.var_pt_scale = tk.StringVar(value="linear")
        rpt = tk.Frame(pt_frame); rpt.pack(fill=tk.X, pady=(0, 2))
        tk.Label(rpt, text="Y scale:").pack(side=tk.LEFT, padx=(0, 4))
        tk.Radiobutton(rpt, text="Log", variable=self.var_pt_scale,
                       value="log", command=self._draw_point_spectrum).pack(side=tk.LEFT)
        tk.Radiobutton(rpt, text="Linear", variable=self.var_pt_scale,
                       value="linear", command=self._draw_point_spectrum).pack(side=tk.LEFT)

        # ---- metadata text ----
        self.text = scrolledtext.ScrolledText(
            outer, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 10), height=8
        )
        self.text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        # ---- status bar ----
        self.status = tk.Label(outer, text="Ready", anchor="w", relief=tk.SUNKEN, bd=1)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ================================================================
    # helpers
    # ================================================================
    def _sum_spectrum(self) -> typing.Optional[typing.Tuple[numpy.ndarray, numpy.ndarray, str]]:
        if self.current_data is None:
            return None
        data = self.current_data.data
        axes_to_sum = tuple(range(data.ndim - 1))
        spectrum = data.sum(axis=axes_to_sum)
        x_cal, units = _get_spectral_cal(self.current_data)
        return x_cal, spectrum, units

    def _update_info_boxes(self) -> None:
        if self.current_data is None or self._x_cal is None or self._spectrum is None:
            return
        fwhm = _compute_fwhm(self._x_cal, self._spectrum.astype(float))
        self.lbl_fwhm.config(text=f"{fwhm:.4g} {self._spec_units}")
        cals = self.current_data.dimensional_calibrations
        if cals and len(cals) >= 3:
            sc = cals[2].scale if cals[2].scale else 1.0
            su = cals[2].units if cals[2].units else ""
            self.lbl_bin_spectrum.config(text=f"{sc:.4g} {su}")
        else:
            self.lbl_bin_spectrum.config(text="—")
        if cals and len(cals) >= 1:
            sx = cals[0].scale if cals[0].scale else 1.0
            ux = cals[0].units if cals[0].units else ""
            self.lbl_bin_x.config(text=f"{sx:.4g} {ux}")
        else:
            self.lbl_bin_x.config(text="—")
        if cals and len(cals) >= 2:
            sy = cals[1].scale if cals[1].scale else 1.0
            uy = cals[1].units if cals[1].units else ""
            self.lbl_bin_y.config(text=f"{sy:.4g} {uy}")
        else:
            self.lbl_bin_y.config(text="—")

    # ================================================================
    # summed spectrum plots (row 1)
    # ================================================================
    def _draw_plots(self) -> None:
        if self._x_cal is None or self._spectrum is None:
            return
        x = self._x_cal; sp = self._spectrum.astype(float)
        u = self._spec_units; ymax = float(numpy.max(sp))
        xlabel = f"Energy ({u})" if u else "Channel index"
        rng = float(x[-1] - x[0])

        fx = self.sl_full_x.get() / 100.0; fy = self.sl_full_y.get() / 100.0
        zx = self.sl_zoom_x.get() / 100.0; zy = self.sl_zoom_y.get() / 100.0

        # full
        self.ax_full.clear()
        self.ax_full.plot(x, sp, linewidth=0.8)
        self.ax_full.set_xlim(-(rng * fx) / 2, (rng * fx) / 2)
        if self.var_full_scale.get() == "log":
            self.ax_full.set_yscale("log")
            ymin = max(1.0, float(sp[sp > 0].min())) if numpy.any(sp > 0) else 1.0
            self.ax_full.set_ylim(ymin, ymax * fy * 1.05)
        else:
            self.ax_full.set_yscale("linear"); self.ax_full.set_ylim(0, ymax * fy)
        self.ax_full.set_xlabel(xlabel); self.ax_full.set_ylabel("Counts")
        self.ax_full.set_title("Full range", fontsize=9)
        self.fig_full.tight_layout(); self.canvas_full.draw_idle()

        # capped
        cap = ymax / 30.0
        self.ax_zoom.clear()
        self.ax_zoom.plot(x, sp, linewidth=0.8)
        self.ax_zoom.set_xlim(-(rng * zx) / 2, (rng * zx) / 2)
        if self.var_zoom_scale.get() == "log":
            self.ax_zoom.set_yscale("log")
            ymin = max(1.0, float(sp[sp > 0].min())) if numpy.any(sp > 0) else 1.0
            self.ax_zoom.set_ylim(ymin, cap * zy * 1.05)
        else:
            self.ax_zoom.set_yscale("linear"); self.ax_zoom.set_ylim(0, cap * zy)
        self.ax_zoom.set_xlabel(xlabel); self.ax_zoom.set_ylabel("Counts")
        self.ax_zoom.set_title(f"Capped at 1/30 max ({cap:.3g})", fontsize=9)
        self.fig_zoom.tight_layout(); self.canvas_zoom.draw_idle()

    def _update_plot(self) -> None:
        result = self._sum_spectrum()
        if result is None:
            return
        self._x_cal, self._spectrum, self._spec_units = result
        self.sl_full_x.set(100); self.sl_full_y.set(100)
        self.sl_zoom_x.set(100); self.sl_zoom_y.set(100)
        self._draw_plots()

    def _on_slider(self) -> None:
        self._draw_plots()

    # ================================================================
    # 3D spatial image (row 2)
    # ================================================================
    def _setup_energy_slider(self) -> None:
        if self.current_data is None:
            return
        data = self.current_data.data
        if data.ndim < 3:
            self.sl_energy.config(from_=0, to=0); return
        self.sl_energy.config(from_=0, to=data.shape[2] - 1)
        zero_idx = _energy_index_for_zero(self.current_data)
        self.sl_energy.set(zero_idx)
        self._cursor_ix = data.shape[0] // 2
        self._cursor_iy = data.shape[1] // 2
        self._draw_image(zero_idx)
        self._draw_point_spectrum()

    def _on_energy_slider(self, val: str) -> None:
        self._draw_image(int(val))

    def _draw_image(self, energy_idx: int) -> None:
        if self.current_data is None:
            return
        data = self.current_data.data
        if data.ndim < 3:
            return
        slc = data[:, :, energy_idx].astype(float)
        off_x, sc_x, u_x = _get_cal(self.current_data, 0)
        off_y, sc_y, u_y = _get_cal(self.current_data, 1)
        off_e, sc_e, u_e = _get_cal(self.current_data, 2)
        nx, ny = data.shape[0], data.shape[1]
        extent = [off_y, off_y + sc_y * ny, off_x + sc_x * nx, off_x]
        energy_val = off_e + sc_e * energy_idx
        self.lbl_energy_val.config(text=f"{energy_val:.4g} {u_e}")

        self.ax_img.clear()
        if self._colorbar is not None:
            self._colorbar.remove(); self._colorbar = None
        im = self.ax_img.imshow(slc, aspect="auto", origin="upper",
                                extent=extent, cmap="inferno")
        self.ax_img.set_xlabel(f"y ({u_y})" if u_y else "y (pixels)")
        self.ax_img.set_ylabel(f"x ({u_x})" if u_x else "x (pixels)")
        self.ax_img.set_title(f"Energy = {energy_val:.4g} {u_e}  [slice {energy_idx}]", fontsize=9)
        self._colorbar = self.fig_img.colorbar(im, ax=self.ax_img, fraction=0.046, pad=0.04)

        # cursor crosshair
        cx = off_y + sc_y * (self._cursor_iy + 0.5)
        cy = off_x + sc_x * (self._cursor_ix + 0.5)
        self.ax_img.plot(cx, cy, marker="+", color="cyan", markersize=14, markeredgewidth=2)
        self.lbl_cursor_pos.config(
            text=f"ix={self._cursor_ix}  iy={self._cursor_iy}  "
                 f"({cy:.4g} {u_x}, {cx:.4g} {u_y})")

        self.fig_img.tight_layout(); self.canvas_img.draw_idle()

    def _on_img_click(self, event: typing.Any) -> None:
        if event.inaxes is not self.ax_img:
            return
        if self.current_data is None or self.current_data.data.ndim < 3:
            return
        off_x, sc_x, _ = _get_cal(self.current_data, 0)
        off_y, sc_y, _ = _get_cal(self.current_data, 1)
        nx, ny = self.current_data.data.shape[0], self.current_data.data.shape[1]
        iy = int((event.xdata - off_y) / sc_y) if sc_y != 0 else 0
        ix = int((event.ydata - off_x) / sc_x) if sc_x != 0 else 0
        self._cursor_ix = max(0, min(nx - 1, ix))
        self._cursor_iy = max(0, min(ny - 1, iy))
        self._draw_image(self.sl_energy.get())
        self._draw_point_spectrum()

    # ================================================================
    # point spectrum (row 3)
    # ================================================================
    def _draw_point_spectrum(self) -> None:
        if self.current_data is None:
            return
        data = self.current_data.data
        if data.ndim < 3:
            return
        x_cal, units = _get_spectral_cal(self.current_data)
        spec = data[self._cursor_ix, self._cursor_iy, :].astype(float)
        off_x, sc_x, u_x = _get_cal(self.current_data, 0)
        off_y, sc_y, u_y = _get_cal(self.current_data, 1)
        px = off_x + sc_x * (self._cursor_ix + 0.5)
        py = off_y + sc_y * (self._cursor_iy + 0.5)

        ymax = float(numpy.max(spec)) if numpy.max(spec) > 0 else 1.0
        rng = float(x_cal[-1] - x_cal[0])
        sx = self.sl_pt_x.get() / 100.0
        sy = self.sl_pt_y.get() / 100.0
        xlabel = f"Energy ({units})" if units else "Channel index"

        self.ax_pt.clear()
        self.ax_pt.plot(x_cal, spec, linewidth=0.8, color="tab:orange")
        self.ax_pt.set_xlim(-(rng * sx) / 2, (rng * sx) / 2)

        if self.var_pt_scale.get() == "log":
            self.ax_pt.set_yscale("log")
            ymin = max(1.0, float(spec[spec > 0].min())) if numpy.any(spec > 0) else 1.0
            self.ax_pt.set_ylim(ymin, ymax * sy * 1.05)
        else:
            self.ax_pt.set_yscale("linear")
            self.ax_pt.set_ylim(0, ymax * sy)

        self.ax_pt.set_xlabel(xlabel)
        self.ax_pt.set_ylabel("Counts")
        self.ax_pt.set_title(
            f"Pixel ({self._cursor_ix}, {self._cursor_iy})  "
            f"[{px:.3g} {u_x}, {py:.3g} {u_y}]", fontsize=9)
        self.fig_pt.tight_layout(); self.canvas_pt.draw_idle()

    # ================================================================
    # file actions
    # ================================================================
    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open NHDF file",
            filetypes=[("NHDF files", "*.nhdf *.h5 *.hdf5"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.current_path = pathlib.Path(path)
            self.current_data = read_data_and_metadata(self.current_path)
            self.file_label.config(text=str(self.current_path), fg="black")
            info = format_info(self.current_data)
            self.text.config(state=tk.NORMAL)
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, info)
            self.text.config(state=tk.DISABLED)
            self.status.config(text=f"Loaded: {self.current_path.name}")
            self._update_plot()
            self._update_info_boxes()
            self._setup_energy_slider()
        except Exception as e:
            messagebox.showerror("Error reading file", str(e))

    def _save_file(self) -> None:
        if self.current_data is None:
            messagebox.showwarning("Nothing to save", "Open a file first.")
            return
        default_name = (
            self.current_path.stem + "_copy.nhdf" if self.current_path else "output.nhdf"
        )
        path = filedialog.asksaveasfilename(
            title="Save NHDF file", defaultextension=".nhdf", initialfile=default_name,
            filetypes=[("NHDF files", "*.nhdf"), ("HDF5 files", "*.h5 *.hdf5")],
        )
        if not path:
            return
        try:
            save_data_and_metadata(pathlib.Path(path), self.current_data)
            self.status.config(text=f"Saved: {pathlib.Path(path).name}")
            messagebox.showinfo("Saved", f"File saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error saving file", str(e))

    def _export_csv(self) -> None:
        if self.current_data is None:
            messagebox.showwarning("Nothing to export", "Open a file first.")
            return
        data = self.current_data.data
        if data.ndim <= 2:
            path = filedialog.asksaveasfilename(
                title="Export CSV", defaultextension=".csv",
                initialfile=(self.current_path.stem + ".csv" if self.current_path else "output.csv"),
                filetypes=[("CSV files", "*.csv")],
            )
            if not path:
                return
            try:
                numpy.savetxt(path, data if data.ndim == 2 else data.reshape(1, -1),
                              delimiter=",", fmt="%.8g")
                self.status.config(text=f"Exported: {pathlib.Path(path).name}")
                messagebox.showinfo("Exported", f"CSV saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Export error", str(e))
        else:
            folder = filedialog.askdirectory(title="Choose folder for CSV export")
            if not folder:
                return
            try:
                base = self.current_path.stem if self.current_path else "data"
                out = pathlib.Path(folder)
                n = data.shape[0]
                for i in range(n):
                    s = data[i]
                    if s.ndim > 2:
                        s = s.reshape(-1, s.shape[-1])
                    numpy.savetxt(str(out / f"{base}_slice{i:04d}.csv"), s,
                                  delimiter=",", fmt="%.8g")
                self.status.config(text=f"Exported {n} CSV files to {folder}")
                messagebox.showinfo("Exported", f"Saved {n} CSV files to:\n{folder}")
            except Exception as e:
                messagebox.showerror("Export error", str(e))

    def _export_sum_spectrum(self) -> None:
        if self.current_data is None:
            messagebox.showwarning("Nothing to export", "Open a file first.")
            return
        result = self._sum_spectrum()
        if result is None:
            return
        x_cal, spectrum, units = result
        default_name = (
            (self.current_path.stem + "_sum_spectrum.csv")
            if self.current_path else "sum_spectrum.csv"
        )
        path = filedialog.asksaveasfilename(
            title="Export summed spectrum CSV", defaultextension=".csv",
            initialfile=default_name, filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        try:
            hdr = f"{'energy (' + units + ')' if units else 'channel'},counts"
            out = numpy.column_stack((x_cal, spectrum))
            numpy.savetxt(path, out, delimiter=",", fmt="%.8g", header=hdr, comments="")
            self.status.config(text=f"Exported sum spectrum: {pathlib.Path(path).name}")
            messagebox.showinfo(
                "Exported",
                f"Summed spectrum ({spectrum.size} ch).\n"
                f"x: {x_cal[0]:.4g} … {x_cal[-1]:.4g} {units}\nSaved to:\n{path}",
            )
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _export_dm3(self) -> None:
        if self.current_data is None:
            messagebox.showwarning("Nothing to export", "Open a file first.")
            return
        if not HAS_HYPERSPY:
            messagebox.showerror("hyperspy not installed",
                                 "pip install hyperspy  or  conda install -c conda-forge hyperspy")
            return
        default_name = (
            (self.current_path.stem + ".dm3") if self.current_path else "output.dm3"
        )
        path = filedialog.asksaveasfilename(
            title="Export DM3 file", defaultextension=".dm3", initialfile=default_name,
            filetypes=[("Digital Micrograph", "*.dm3"), ("DM4 files", "*.dm4")],
        )
        if not path:
            return
        try:
            export_dm3(pathlib.Path(path), self.current_data)
            self.status.config(text=f"Exported DM3: {pathlib.Path(path).name}")
            messagebox.showinfo("Exported", f"DM3 saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("DM3 export error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    NHDFApp(root)
    root.mainloop()