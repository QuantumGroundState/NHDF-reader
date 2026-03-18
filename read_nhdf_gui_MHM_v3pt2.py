"""
GUI application to read and save NHDF (Nion HDF5) data and metadata files.

Required packages:
  pip install numpy h5py niondata matplotlib
"""

import h5py
import json
import numpy
import numpy.typing
import os
import pathlib
import pprint
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import traceback
import typing
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import lambertw, erf
import datetime

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.path import Path as MplPath

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.utils import Converter

_NDArray = numpy.typing.NDArray[typing.Any]
PHI = 1.618033988749895
_FNT = ("Arial", 8)
_FNT_M = ("Arial", 8)
_FNT_MB = ("Arial", 8, "bold")
_FP = 7
_CMAPS = ["inferno", "viridis", "gray", "cubehelix", "magma", "plasma", "hot", "coolwarm",
          "cividis", "turbo", "bone", "copper", "twilight"]


# ================================================================
# NHDF read / write
# ================================================================
def read_data_and_metadata(path: pathlib.Path) -> DataAndMetadata.DataAndMetadata:
    with h5py.File(str(path), "r") as f:
        dg = f["data"]; key0 = list(sorted(dg.keys()))[0]; ds = dg[key0]
        jp = json.loads(ds.attrs["properties"]); data = numpy.array(ds)
        dd = DataAndMetadata.DataDescriptor(
            is_sequence=jp.get("is_sequence", False),
            collection_dimension_count=jp.get("collection_dimension_count", 0),
            datum_dimension_count=jp.get("datum_dimension_count", 0))
        dm = DataAndMetadata.DataMetadata(
            data_shape_and_dtype=(data.shape, data.dtype),
            intensity_calibration=Calibration.Calibration.from_rpc_dict(jp.get("intensity_calibration", {})),
            dimensional_calibrations=[
                typing.cast(Calibration.Calibration, Calibration.Calibration.from_rpc_dict(d))
                for d in jp.get("dimensional_calibrations", [])],
            metadata=jp.get("metadata", {}),
            timestamp=Converter.DatetimeToStringConverter().convert_back(jp.get("created", "")),
            data_descriptor=dd, timezone=jp.get("timezone", None),
            timezone_offset=jp.get("timezone_offset", None))
        data_fn = lambda: data
        data_fn.shape = data.shape
        data_fn.dtype = data.dtype
        return DataAndMetadata.DataAndMetadata(
            data_fn, data_shape_and_dtype=dm.data_shape_and_dtype,
            intensity_calibration=dm.intensity_calibration,
            dimensional_calibrations=dm.dimensional_calibrations,
            metadata=dm.metadata, timestamp=dm.timestamp,
            data_descriptor=dm.data_descriptor,
            timezone=dm.timezone, timezone_offset=dm.timezone_offset)


def save_data_and_metadata(path: pathlib.Path, d: DataAndMetadata.DataAndMetadata) -> None:
    props: typing.Dict[str, typing.Any] = {}
    props["is_sequence"] = d.data_descriptor.is_sequence
    props["collection_dimension_count"] = d.data_descriptor.collection_dimension_count
    props["datum_dimension_count"] = d.data_descriptor.datum_dimension_count
    if d.intensity_calibration: props["intensity_calibration"] = d.intensity_calibration.rpc_dict
    if d.dimensional_calibrations:
        props["dimensional_calibrations"] = [c.rpc_dict for c in d.dimensional_calibrations]
    if d.metadata: props["metadata"] = d.metadata
    if d.timestamp: props["created"] = Converter.DatetimeToStringConverter().convert(d.timestamp)
    if d.timezone: props["timezone"] = d.timezone
    if d.timezone_offset: props["timezone_offset"] = d.timezone_offset
    with h5py.File(str(path), "w") as f:
        dg = f.create_group("data"); ds = dg.create_dataset("0", data=d.data)
        ds.attrs["properties"] = json.dumps(props)


# ================================================================
# Helpers
# ================================================================
def format_info(d: DataAndMetadata.DataAndMetadata) -> str:
    return "\n".join([
        f"Shape: {d.data.shape}", f"Dtype: {d.data.dtype}",
        f"Descriptor: {d.data_descriptor}",
        f"  is_seq: {d.data_descriptor.is_sequence}",
        f"  coll_dim: {d.data_descriptor.collection_dimension_count}",
        f"  datum_dim: {d.data_descriptor.datum_dimension_count}",
        f"Intensity Cal: {d.intensity_calibration}",
        f"Dim Cals: {d.dimensional_calibrations}",
        f"Time: {d.timestamp}  TZ: {d.timezone}  Off: {d.timezone_offset}",
        "", "Metadata:", pprint.pformat(d.metadata)])


def _gc(d, dim):
    c = d.dimensional_calibrations
    if c and dim < len(c):
        v = c[dim]; return (v.offset or 0., v.scale or 1., v.units or "")
    return 0., 1., ""


def _spec_cal(d):
    c = d.dimensional_calibrations
    cal = c[2] if c and len(c) >= 3 else (c[-1] if c else None)
    n = d.data.shape[-1]
    if cal:
        o = cal.offset or 0.; s = cal.scale or 1.; u = cal.units or ""
        return o + s * numpy.arange(n), u
    return numpy.arange(n, dtype=float), ""


def _e0idx(d):
    x, _ = _spec_cal(d); return int(numpy.argmin(numpy.abs(x)))


def _fwhm(x, y):
    hm = numpy.max(y) / 2.; i = numpy.where(y >= hm)[0]
    return float(x[i[-1]] - x[i[0]]) if len(i) >= 2 else 0.


# ================================================================
# Fit functions
# ================================================================
def _gauss_fn(x, A, x0, sigma):
    return A * numpy.exp(-(x - x0)**2 / (2 * sigma**2))

def _lorentz_fn(x, A, x0, gamma):
    return A / (1 + ((x - x0) / gamma)**2)

def _gausslor_fn(x, A, x0, sigma):
    return A * numpy.exp(-(x - x0)**2 / (2 * sigma**2)) / (1 + ((x - x0) / sigma)**2)

def _fwhm_gauss(sigma):
    return 2 * abs(sigma) * numpy.sqrt(2 * numpy.log(2))

def _fwhm_lorentz(gamma):
    return 2 * abs(gamma)

def _fwhm_gausslor(sigma):
    w = float(numpy.real(lambertw(numpy.sqrt(numpy.e))))
    return 2 * abs(sigma) * numpy.sqrt(2 * w - 1)

def _arr(dam):
    v = dam.data
    return v() if callable(v) else v

def _date_prefix():
    return datetime.datetime.now().strftime("%Y%m%d_")


# ================================================================
# GUI
# ================================================================
class NHDFApp:
    def __init__(self, root):
        self.root = root; self.root.title("NHDF Reader / Writer")
        self.root.geometry("1100x920")
        self.current_data = None; self.current_path = None
        self._xc = None; self._sp = None; self._su = ""
        self._cb = None; self._cix = 0; self._ciy = 0
        self._roi = None; self._roi_drawing = False; self._roi_start = None
        self._roi_moving = False; self._roi_drag_start = None; self._roi_anchor = None
        self._roi2 = None; self._roi2_moving = False
        self._roi2_drag_start = None; self._roi2_anchor = None
        self._roi_drag_corner = None; self._roi2_drag_corner = None
        self._pt_vline_energy = None
        self._build_ui()

    def _build_ui(self):
        ct = tk.Frame(self.root); ct.pack(fill=tk.BOTH, expand=True)
        cv = tk.Canvas(ct); vs = tk.Scrollbar(ct, orient=tk.VERTICAL, command=cv.yview)
        self._inn = tk.Frame(cv)
        self._inn.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=self._inn, anchor="nw")
        cv.configure(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y); cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-1*(e.delta/120)), "units"))
        o = self._inn

        # toolbar
        tb = tk.Frame(o); tb.pack(fill=tk.X, padx=6, pady=3)
        for txt, cmd, w in [
            ("Open NHDF…", self._open_file, 11), ("Save NHDF…", self._save_file, 11),
            ("Export CSV…", self._export_csv, 11),
            ("Export Sum…", self._export_sum, 11),
            ("Export Cube…", self._export_cube, 11)]:
            tk.Button(tb, text=txt, command=cmd, width=w, font=_FNT).pack(side=tk.LEFT, padx=(0, 3))

        self.file_lbl = tk.Label(o, text="No file loaded", anchor="w", fg="gray", font=_FNT)
        self.file_lbl.pack(fill=tk.X, padx=8)

        # Comment box
        cf = tk.Frame(o); cf.pack(fill=tk.X, padx=6, pady=(2, 0))
        tk.Label(cf, text="Comment:", font=_FNT).pack(side=tk.LEFT)
        self.comment_entry = tk.Entry(cf, font=_FNT)
        self.comment_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # Fit function radio buttons
        fr = tk.Frame(o); fr.pack(fill=tk.X, padx=6, pady=(2, 0))
        tk.Label(fr, text="Fit:", font=_FNT_MB).pack(side=tk.LEFT)
        self.var_fit = tk.StringVar(value="gausslor")
        for lbl, val in [("Gaussian", "gauss"), ("Lorentzian", "lorentz"), ("GaussLor", "gausslor")]:
            tk.Radiobutton(fr, text=lbl, variable=self.var_fit, value=val,
                           command=self._on_fit_change, font=_FNT).pack(side=tk.LEFT, padx=(4, 0))

        # info boxes
        ir = tk.Frame(o); ir.pack(fill=tk.X, padx=6, pady=(2, 1))
        ir.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)
        def mkb(p, l, c):
            f = tk.LabelFrame(p, text=l, padx=3, pady=1, font=_FNT)
            f.grid(row=0, column=c, padx=2, sticky="nsew")
            v = tk.Label(f, text="—", font=_FNT_MB, anchor="center"); v.pack(fill=tk.X); return v
        self.lb_pk = mkb(ir, "Peak Max", 0)
        self.lb_ar = mkb(ir, "Area", 1)
        self.lb_fw = mkb(ir, "FWHM", 2)
        self.lb_bs = mkb(ir, "Bin(spectrum)", 3)
        self.lb_bx = mkb(ir, "Bin(x/dim0)", 4)
        self.lb_by = mkb(ir, "Bin(y/dim1)", 5)

        # ROW 1: two summed-spectrum plots
        r1 = tk.Frame(o); r1.pack(fill=tk.X, padx=6, pady=(2, 0))
        r1.columnconfigure((0, 1), weight=1)
        self.ff, self.af, self.cf, self.sxf, self.syf, self.vsf = \
            self._mksp(r1, 0, "Full-range sum", "log", (4.8, 1.4))
        self.fz, self.az, self.cz, self.sxz, self.syz, self.vsz = \
            self._mksp(r1, 1, "1/30-max capped sum", "linear", (4.8, 1.4))

        # ROW 2: spatial image + ROI + colormap controls
        imf = tk.LabelFrame(o, text="Spatial image — click=cursor, drag ROI when active",
                            padx=3, pady=2, font=_FNT)
        imf.pack(fill=tk.X, padx=6, pady=(3, 0))
        # Slider row with peak arrows overlaid
        sl_frame = tk.Frame(imf); sl_frame.pack(fill=tk.X)
        sc = tk.Frame(sl_frame); sc.pack(fill=tk.X)
        tk.Label(sc, text="E:", font=_FNT).pack(side=tk.LEFT)
        self.lb_e = tk.Label(sc, text="—", font=_FNT_MB, width=14, anchor="center", relief=tk.GROOVE)
        self.lb_e.pack(side=tk.LEFT, padx=2)
        self.sl_e = tk.Scale(sc, from_=0, to=0, orient=tk.HORIZONTAL,
                             command=self._on_esl, length=300, font=_FNT)
        self.sl_e.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(sc, text="Cur:", font=_FNT).pack(side=tk.LEFT, padx=(4, 1))
        self.lb_c = tk.Label(sc, text="—", font=_FNT_MB, width=22, anchor="center", relief=tk.GROOVE)
        self.lb_c.pack(side=tk.LEFT)
        # Peak arrow canvas overlaid on top of slider
        self.peak_canvas = tk.Canvas(sl_frame, height=10, highlightthickness=0)
        self.peak_canvas.place(relx=0, rely=0, relwidth=1.0)
        self.peak_canvas.config(bg=imf.cget("bg"))
        self._peak_indices = []

        # ROI row + colormaps on the right
        rc = tk.Frame(imf); rc.pack(fill=tk.X, pady=(1, 0))
        self.btn_roi = tk.Button(rc, text="Draw ROI", command=self._toggle_roi,
                                 font=_FNT, width=9)
        self.btn_roi.pack(side=tk.LEFT, padx=(0, 3))
        self.btn_mvroi = tk.Button(rc, text="Move ROI", command=self._toggle_move_roi,
                                   font=_FNT, width=9)
        self.btn_mvroi.pack(side=tk.LEFT, padx=(0, 3))
        self.btn_clr = tk.Button(rc, text="Clear ROI", command=self._clear_roi,
                                 font=_FNT, width=9)
        self.btn_clr.pack(side=tk.LEFT, padx=(0, 3))
        self.btn_eroi = tk.Button(rc, text="Export ROI…", command=self._export_roi,
                                  font=_FNT, width=10)
        self.btn_eroi.pack(side=tk.LEFT, padx=(0, 3))
        self.lb_roi = tk.Label(rc, text="ROI: none", font=_FNT_MB, anchor="w")
        self.lb_roi.pack(side=tk.LEFT, padx=(4, 0))
        # Colormaps on the right of ROI row
        self.var_cmap = tk.StringVar(value="inferno")
        for cm in _CMAPS:
            tk.Radiobutton(rc, text=cm, variable=self.var_cmap, value=cm,
                           command=lambda: self._draw_img(self.sl_e.get()),
                           font=_FNT).pack(side=tk.RIGHT)

        # Bkg row
        rc2 = tk.Frame(imf); rc2.pack(fill=tk.X, pady=(1, 0))
        self.btn_roi2 = tk.Button(rc2, text="Place Bkg", command=self._place_roi2,
                                  font=_FNT, width=9)
        self.btn_roi2.pack(side=tk.LEFT, padx=(0, 3))
        self.btn_mvroi2 = tk.Button(rc2, text="Move Bkg", command=self._toggle_move_roi2,
                                    font=_FNT, width=9)
        self.btn_mvroi2.pack(side=tk.LEFT, padx=(0, 3))
        self.btn_clr2 = tk.Button(rc2, text="Clear Bkg", command=self._clear_roi2,
                                  font=_FNT, width=9)
        self.btn_clr2.pack(side=tk.LEFT, padx=(0, 3))
        self.btn_ebkg = tk.Button(rc2, text="Export Bkg…", command=self._export_bkg,
                                  font=_FNT, width=10)
        self.btn_ebkg.pack(side=tk.LEFT, padx=(0, 3))
        self.lb_roi2 = tk.Label(rc2, text="Bkg: none", font=_FNT_MB, anchor="w")
        self.lb_roi2.pack(side=tk.LEFT, padx=(4, 0))
        # Contour overlay toggle (right side of Bkg row)
        tk.Label(rc2, text="  Contour:", font=_FNT).pack(side=tk.RIGHT, padx=(0, 2))
        self.var_contour = tk.StringVar(value="off")
        tk.Radiobutton(rc2, text="On", variable=self.var_contour, value="on",
                       command=lambda: self._draw_img(self.sl_e.get()),
                       font=_FNT).pack(side=tk.RIGHT)
        tk.Radiobutton(rc2, text="Off", variable=self.var_contour, value="off",
                       command=lambda: self._draw_img(self.sl_e.get()),
                       font=_FNT).pack(side=tk.RIGHT)
        # Energy binning radio buttons (bin along z / energy axis)
        tk.Label(rc2, text="  BinE:", font=_FNT).pack(side=tk.RIGHT, padx=(8, 2))
        self.var_ebin = tk.StringVar(value="1")
        for _b in ["8", "4", "2", "1"]:
            tk.Radiobutton(rc2, text=_b, variable=self.var_ebin, value=_b,
                           command=lambda: self._draw_img(self.sl_e.get()),
                           font=_FNT).pack(side=tk.RIGHT)

        self.fi = Figure(figsize=(10, 2.0), dpi=100)
        self.ai = self.fi.add_subplot(111)
        self.ci = FigureCanvasTkAgg(self.fi, master=imf)
        self.ci.get_tk_widget().pack(fill=tk.X)
        self.ci.mpl_connect("button_press_event", self._on_img_press)
        self.ci.mpl_connect("button_release_event", self._on_img_release)
        self.ci.mpl_connect("motion_notify_event", self._on_img_motion)
        self._img_widget = self.ci.get_tk_widget()
        self._img_widget.configure(takefocus=True)
        self._img_widget.bind("<Left>", self._on_arrow_key)
        self._img_widget.bind("<Right>", self._on_arrow_key)
        self._img_widget.bind("<Up>", self._on_arrow_key)
        self._img_widget.bind("<Down>", self._on_arrow_key)
        self._img_widget.bind("<Shift-Left>", self._on_arrow_key)
        self._img_widget.bind("<Shift-Right>", self._on_arrow_key)
        self._img_widget.bind("<Shift-Up>", self._on_arrow_key)
        self._img_widget.bind("<Shift-Down>", self._on_arrow_key)

        # ROW 3: point spectrum + ROI overlay
        pf = tk.LabelFrame(o, text="Point spectrum (cursor) + ROI (green) + Bkg (cyan)", padx=3, pady=2, font=_FNT)
        pf.pack(fill=tk.X, padx=6, pady=(3, 0))
        self.fp = Figure(figsize=(10, 1.6), dpi=100)
        self.ap = self.fp.add_subplot(111)
        self.cp = FigureCanvasTkAgg(self.fp, master=pf)
        self.cp.get_tk_widget().pack(fill=tk.X)
        self.cp.mpl_connect("button_press_event", self._on_pt_click)
        cp = tk.Frame(pf); cp.pack(fill=tk.X, pady=(1, 0))
        tk.Label(cp, text="X%", font=_FNT).pack(side=tk.LEFT)
        self.sxp = tk.Scale(cp, from_=1, to=100, orient=tk.HORIZONTAL,
                            command=lambda v: self._draw_pt(), length=100, font=_FNT)
        self.sxp.set(100); self.sxp.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(cp, text="Y%", font=_FNT).pack(side=tk.LEFT)
        self.syp = tk.Scale(cp, from_=1, to=500, orient=tk.HORIZONTAL,
                            command=lambda v: self._draw_pt(), length=100, font=_FNT)
        self.syp.set(100); self.syp.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.vsp = tk.StringVar(value="linear")
        tk.Label(cp, text=" Y:", font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp, text="Log", variable=self.vsp, value="log",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp, text="Lin", variable=self.vsp, value="linear",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Label(cp, text="  Norm:", font=_FNT).pack(side=tk.LEFT, padx=(6, 2))
        self.var_norm = tk.StringVar(value="off")
        tk.Radiobutton(cp, text="Off", variable=self.var_norm, value="off",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp, text="Max→1", variable=self.var_norm, value="max",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp, text="Area→1", variable=self.var_norm, value="area",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)

        # Subtract mode row
        cp2 = tk.Frame(pf); cp2.pack(fill=tk.X, pady=(1, 0))
        tk.Label(cp2, text="Subtract:", font=_FNT).pack(side=tk.LEFT, padx=(0, 2))
        self.var_sub = tk.StringVar(value="off")
        tk.Radiobutton(cp2, text="Off", variable=self.var_sub, value="off",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="Cursor − ROI", variable=self.var_sub, value="cur_roi",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="ROI − Cursor", variable=self.var_sub, value="roi_cur",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="Cursor \u2212 Bkg", variable=self.var_sub, value="cur_roi2",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="ROI \u2212 Bkg", variable=self.var_sub, value="roi_roi2",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="Bkg \u2212 ROI", variable=self.var_sub, value="roi2_roi",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="Bkg \u2212 Cursor", variable=self.var_sub, value="roi2_cur",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Button(cp2, text="Export Sub\u2026", command=self._export_sub,
                  font=_FNT, width=10).pack(side=tk.RIGHT, padx=(6, 0))

        self.text = scrolledtext.ScrolledText(o, wrap=tk.WORD, state=tk.DISABLED,
                                              font=_FNT_M, height=5)
        self.text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(3, 3))
        self.status = tk.Label(o, text="Ready", anchor="w", relief=tk.SUNKEN, bd=1, font=_FNT)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _mksp(self, par, col, title, dflt, fsz):
        f = tk.LabelFrame(par, text=title, padx=3, pady=1, font=_FNT)
        f.grid(row=0, column=col, padx=2, sticky="nsew")
        fig = Figure(figsize=fsz, dpi=100); ax = fig.add_subplot(111)
        c = FigureCanvasTkAgg(fig, master=f); c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        cf = tk.Frame(f); cf.pack(fill=tk.X, pady=(1, 0))
        tk.Label(cf, text="X%", font=_FNT).pack(side=tk.LEFT)
        sx = tk.Scale(cf, from_=1, to=100, orient=tk.HORIZONTAL,
                      command=lambda v: self._on_sl(), length=80, font=_FNT)
        sx.set(100); sx.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(cf, text="Y%", font=_FNT).pack(side=tk.LEFT)
        sy = tk.Scale(cf, from_=1, to=100, orient=tk.HORIZONTAL,
                      command=lambda v: self._on_sl(), length=80, font=_FNT)
        sy.set(100); sy.pack(side=tk.LEFT, fill=tk.X, expand=True)
        vs = tk.StringVar(value=dflt)
        tk.Label(cf, text=" Y:", font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cf, text="Log", variable=vs, value="log",
                       command=self._on_sl, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cf, text="Lin", variable=vs, value="linear",
                       command=self._on_sl, font=_FNT).pack(side=tk.LEFT)
        return fig, ax, c, sx, sy, vs

    # ---- data helpers ----
    def _sum_sp(self):
        if not self.current_data: return None
        d = _arr(self.current_data)
        sp = d.sum(axis=tuple(range(d.ndim - 1)))
        x, u = _spec_cal(self.current_data)
        return x, u, sp

    def _fit_peak(self, x, y):
        """Fit the selected model to spectrum. Returns (peak_max, area, fwhm, x0, fitted_y) or None."""
        mode = self.var_fit.get()
        y_f = y.astype(float)
        if numpy.max(y_f) <= 0:
            return None
        idx_max = numpy.argmax(y_f)
        A0 = float(y_f[idx_max])
        x0_0 = float(x[idx_max])
        hm = A0 / 2
        above = numpy.where(y_f >= hm)[0]
        if len(above) >= 2:
            sig0 = float(x[above[-1]] - x[above[0]]) / 2.355
        else:
            sig0 = float(x[-1] - x[0]) / 10
        sig0 = max(sig0, abs(float(x[1] - x[0])) if len(x) > 1 else 1.0)
        try:
            if mode == "gauss":
                popt, _ = curve_fit(_gauss_fn, x, y_f, p0=[A0, x0_0, sig0], maxfev=5000)
                fwhm = _fwhm_gauss(popt[2])
                fitted = _gauss_fn(x, *popt)
            elif mode == "lorentz":
                popt, _ = curve_fit(_lorentz_fn, x, y_f, p0=[A0, x0_0, sig0], maxfev=5000)
                fwhm = _fwhm_lorentz(popt[2])
                fitted = _lorentz_fn(x, *popt)
            else:
                popt, _ = curve_fit(_gausslor_fn, x, y_f, p0=[A0, x0_0, sig0], maxfev=5000)
                fwhm = _fwhm_gausslor(popt[2])
                fitted = _gausslor_fn(x, *popt)
            peak_max = float(numpy.max(fitted))
            area = float(numpy.trapz(fitted, x))
            return peak_max, area, fwhm, float(popt[1]), fitted
        except Exception:
            return None

    def _on_fit_change(self):
        self._upd_info()
        self._draw_pt()

    def _upd_info(self):
        if self._xc is None: return
        result = self._fit_peak(self._xc, self._sp)
        if result:
            peak_max, area, fw, x0, fitted = result
            self.lb_pk.config(text=f"{peak_max:.4g}")
            self.lb_ar.config(text=f"{area:.4g}")
            self.lb_fw.config(text=f"{fw:.4g} {self._su}")
        else:
            fw = _fwhm(self._xc, self._sp.astype(float))
            self.lb_pk.config(text="—")
            self.lb_ar.config(text="—")
            self.lb_fw.config(text=f"{fw:.4g} {self._su}")
        cs = self.current_data.dimensional_calibrations if self.current_data else []
        for lb, i in [(self.lb_bs, 2), (self.lb_bx, 0), (self.lb_by, 1)]:
            if cs and i < len(cs):
                c = cs[i]; lb.config(text=f"{c.scale or 1:.4g} {c.units or ''}")
            else: lb.config(text="—")

    # ---- spectrum plots ----
    def _d1(self, ax, fig, cv, sx, sy, vs, cap=None):
        if self._xc is None: return
        x = self._xc; sp = self._sp.astype(float); u = self._su
        ym = float(numpy.max(sp)); r = float(x[-1]-x[0])
        fx = sx.get()/100.; fy = sy.get()/100.
        ax.clear(); ax.plot(x, sp, lw=0.6)
        ax.set_xlim(-(r*fx)/2, (r*fx)/2)
        top = (ym/cap if cap else ym)*fy
        if vs.get() == "log":
            ax.set_yscale("log")
            mn = max(1., float(sp[sp > 0].min())) if numpy.any(sp > 0) else 1.
            ax.set_ylim(mn, top*1.05)
        else: ax.set_yscale("linear"); ax.set_ylim(0, top)
        ax.set_xlabel(f"Energy ({u})" if u else "Ch", fontsize=_FP)
        ax.set_ylabel("Cts", fontsize=_FP); ax.tick_params(labelsize=_FP)
        fig.tight_layout(); cv.draw_idle()

    def _draw_plots(self):
        self._d1(self.af, self.ff, self.cf, self.sxf, self.syf, self.vsf)
        self._d1(self.az, self.fz, self.cz, self.sxz, self.syz, self.vsz, cap=30)

    def _upd_plot(self):
        r = self._sum_sp()
        if not r: return
        self._xc, self._su, self._sp = r
        for s in [self.sxf, self.syf, self.sxz, self.syz]: s.set(100)
        self._draw_plots()

    def _on_sl(self): self._draw_plots()

    # ---- spatial image ----
    def _setup_esl(self):
        if not self.current_data: return
        d = _arr(self.current_data)
        if d.ndim < 3: self.sl_e.config(from_=0, to=0); return
        self.sl_e.config(from_=0, to=d.shape[2]-1)
        # Default to Fuchs-Kliewer phonon mode at 0.084 eV; fall back to E=0 index
        _xfk, _ = _spec_cal(self.current_data)
        zi = int(numpy.argmin(numpy.abs(_xfk - 0.06746)))
        self.sl_e.set(zi)
        self._cix = d.shape[0]//2; self._ciy = d.shape[1]//2
        self._roi = None; self._roi_drawing = False
        self._roi_moving = False; self._roi_drag_start = None; self._roi_anchor = None
        self._roi_drag_corner = None
        self._roi2 = None; self._roi2_moving = False
        self._roi2_drag_start = None; self._roi2_anchor = None
        self._roi2_drag_corner = None
        self.lb_roi.config(text="ROI: none")
        self.lb_roi2.config(text="Bkg: none")
        self.btn_mvroi.config(relief=tk.RAISED, text="Move ROI")
        self.btn_mvroi2.config(relief=tk.RAISED, text="Move Bkg")
        self._detect_peaks()
        self._draw_img(zi); self._draw_pt()

    def _detect_peaks(self):
        """Find peaks in the summed spectrum using both linear and log scale."""
        self.peak_canvas.delete("all")
        self._peak_indices = []
        if not self.current_data: return
        d = _arr(self.current_data)
        if d.ndim < 3: return
        sp = d.sum(axis=(0, 1)).astype(float)
        nch = len(sp)
        if nch < 5: return
        from scipy.ndimage import uniform_filter1d
        kern = max(3, nch // 100)
        dist = max(3, nch // 50)
        # Linear-scale peaks
        smooth = uniform_filter1d(sp, size=kern)
        prom_lin = max(smooth.max() * 0.02, 1.)
        peaks_lin, _ = find_peaks(smooth, prominence=prom_lin, distance=dist)
        # Log-scale peaks (catches smaller features)
        sp_pos = numpy.clip(sp, 1e-30, None)
        log_sp = numpy.log10(sp_pos)
        log_smooth = uniform_filter1d(log_sp, size=kern)
        prom_log = max((log_smooth.max() - log_smooth.min()) * 0.03, 0.05)
        peaks_log, _ = find_peaks(log_smooth, prominence=prom_log, distance=dist)
        # Merge and deduplicate
        all_peaks = numpy.unique(numpy.concatenate([peaks_lin, peaks_log]))
        self._peak_indices = all_peaks
        # Draw after canvas has laid out
        self.peak_canvas.after(50, self._draw_peak_arrows)

    def _draw_peak_arrows(self):
        self.peak_canvas.delete("all")
        if not self._peak_indices.size or not self.current_data: return
        d = _arr(self.current_data)
        if d.ndim < 3: return
        nch = d.shape[2]
        self.peak_canvas.update_idletasks()
        w = self.peak_canvas.winfo_width()
        # The slider sits between the E-label/lb_e on the left and Cur-label on the right.
        # Estimate the slider trough region: use sl_e widget geometry relative to peak_canvas.
        try:
            sl_x = self.sl_e.winfo_x()
            sl_w = self.sl_e.winfo_width()
        except Exception:
            sl_x = 60; sl_w = w - 180
        pad = 14  # slider internal trough padding
        x0 = sl_x + pad
        x1 = sl_x + sl_w - pad
        span = x1 - x0
        if span <= 0: return
        h = self.peak_canvas.winfo_height()
        for idx in self._peak_indices:
            frac = idx / max(nch - 1, 1)
            cx = x0 + frac * span
            # Small downward-pointing triangle
            self.peak_canvas.create_polygon(
                cx - 3, 0, cx + 3, 0, cx, h - 1,
                fill="red", outline="red")

    def _on_esl(self, v): self._draw_img(int(v))

    def _draw_img(self, ei):
        if not self.current_data: return
        d = _arr(self.current_data)
        if d.ndim < 3: return
        nz = d.shape[2]
        ebin = int(self.var_ebin.get())
        if ebin > 1:
            lo = max(0, ei - ebin // 2)
            hi = min(nz, lo + ebin)
            lo = max(0, hi - ebin)   # re-clamp lo if hi hit the ceiling
            slc = d[:, :, lo:hi].mean(axis=2).astype(float)
        else:
            slc = d[:, :, ei].astype(float)
        _, sx, ux = _gc(self.current_data, 0); ox = 0.
        _, sy, uy = _gc(self.current_data, 1); oy = 0.
        oe, se, ue = _gc(self.current_data, 2)
        nx, ny = d.shape[0], d.shape[1]
        ext = [0, sy*ny, 0, sx*nx]
        ev = oe+se*ei
        self.lb_e.config(text=f"{ev:.4g} {ue}")
        self.fi.clear(); self.ai = self.fi.add_subplot(111)
        self._cb = None
        cmap = self.var_cmap.get()
        im = self.ai.imshow(slc, aspect="auto", origin="lower", extent=ext, cmap=cmap)
        self.ai.set_xlabel(f"y ({uy})" if uy else "y", fontsize=_FP)
        self.ai.set_ylabel(f"x ({ux})" if ux else "x", fontsize=_FP)
        bin_tag = f" ×{ebin}" if ebin > 1 else ""
        self.ai.set_title(f"E={ev:.4g} {ue} [slice {ei}{bin_tag}]", fontsize=_FP)
        self.ai.tick_params(labelsize=_FP)
        self._cb = self.fi.colorbar(im, ax=self.ai, fraction=0.025, pad=0.015)
        self._cb.ax.tick_params(labelsize=_FP)
        cy_cal = oy + sy * (self._ciy + 0.5)
        cx_cal = ox + sx * (self._cix + 0.5)
        self.ai.plot(cy_cal, cx_cal, "+", color="cyan", ms=10, mew=2)
        self.lb_c.config(text=f"({self._cix},{self._ciy}) {cx_cal:.3g}{ux},{cy_cal:.3g}{uy}")
        if self._roi is not None:
            verts = [(sy * (iy + 0.5), sx * (ix + 0.5)) for ix, iy in self._roi]
            poly = MplPolygon(verts, closed=True, linewidth=1.5,
                              edgecolor="lime", facecolor="lime", alpha=0.2)
            self.ai.add_patch(poly)
            for px, py in verts:
                self.ai.plot(px, py, "s", color="lime", ms=5, mew=1.5)
        if self._roi2 is not None:
            verts2 = [(sy * (iy + 0.5), sx * (ix + 0.5)) for ix, iy in self._roi2]
            poly2 = MplPolygon(verts2, closed=True, linewidth=1.5,
                               edgecolor="cyan", facecolor="cyan", alpha=0.2)
            self.ai.add_patch(poly2)
            for px, py in verts2:
                self.ai.plot(px, py, "s", color="cyan", ms=5, mew=1.5)
        # Contour overlay of equal intensity
        if self.var_contour.get() == "on":
            vmin, vmax = float(slc.min()), float(slc.max())
            if vmax > vmin:
                levels = numpy.linspace(vmin + (vmax - vmin) * 0.1, vmax * 0.95, 8)
                self.ai.contour(slc, levels=levels, origin="lower", extent=ext,
                                colors="white", linewidths=0.6, alpha=0.7)
        self.fi.tight_layout(); self.ci.draw_idle()

    # ---- ROI interaction ----
    def _cal_to_pix(self, xdata, ydata):
        _, sx, _ = _gc(self.current_data, 0)
        _, sy, _ = _gc(self.current_data, 1)
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        iy = int(numpy.floor(xdata / sy)) if sy else 0
        ix = int(numpy.floor(ydata / sx)) if sx else 0
        return max(0, min(nx-1, ix)), max(0, min(ny-1, iy))

    def _poly_mask(self, corners):
        d = _arr(self.current_data)
        nx, ny = d.shape[0], d.shape[1]
        fc = numpy.vstack([corners, corners[0:1]]).astype(float)
        path = MplPath(fc)
        ix_min = max(0, int(corners[:, 0].min()))
        ix_max = min(nx - 1, int(corners[:, 0].max()))
        iy_min = max(0, int(corners[:, 1].min()))
        iy_max = min(ny - 1, int(corners[:, 1].max()))
        mask = numpy.zeros((nx, ny), dtype=bool)
        if ix_min > ix_max or iy_min > iy_max:
            return mask
        sub_ix, sub_iy = numpy.mgrid[ix_min:ix_max+1, iy_min:iy_max+1]
        pts = numpy.column_stack((sub_ix.ravel(), sub_iy.ravel())).astype(float)
        sub_mask = path.contains_points(pts).reshape(sub_ix.shape)
        mask[ix_min:ix_max+1, iy_min:iy_max+1] = sub_mask
        return mask

    def _find_corner(self, corners, ix, iy, threshold=3):
        dists = numpy.sqrt((corners[:, 0].astype(float) - ix)**2 +
                           (corners[:, 1].astype(float) - iy)**2)
        ci = int(numpy.argmin(dists))
        return ci if dists[ci] <= threshold else -1

    def _roi_label_text(self, corners, prefix):
        xs = corners[:, 0]; ys = corners[:, 1]
        return f"{prefix}: x[{xs.min()}-{xs.max()}] y[{ys.min()}-{ys.max()}]"

    def _toggle_roi(self):
        self._roi_drawing = not self._roi_drawing
        if self._roi_drawing:
            self.btn_roi.config(relief=tk.SUNKEN, text="ROI active")
        else:
            self.btn_roi.config(relief=tk.RAISED, text="Draw ROI")

    def _clear_roi(self):
        self._roi = None; self._roi_drawing = False; self._roi_moving = False
        self._roi_drag_corner = None
        self.btn_roi.config(relief=tk.RAISED, text="Draw ROI")
        self.btn_mvroi.config(relief=tk.RAISED, text="Move ROI")
        self.lb_roi.config(text="ROI: none")
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _toggle_move_roi(self):
        if self._roi is None:
            messagebox.showwarning("No ROI", "Draw ROI first.")
            return
        self._roi_moving = not self._roi_moving
        if self._roi_moving:
            self._roi2_moving = False
            self.btn_mvroi2.config(relief=tk.RAISED, text="Move Bkg")
            self.btn_mvroi.config(relief=tk.SUNKEN, text="Moving\u2026")
        else:
            self.btn_mvroi.config(relief=tk.RAISED, text="Move ROI")

    def _move_roi(self, dx, dy):
        if self._roi_anchor is None: return
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        if self._roi_drag_corner is not None:
            ci = self._roi_drag_corner
            new_corners = self._roi_anchor.copy()
            new_corners[ci, 0] = max(0, min(nx - 1, int(self._roi_anchor[ci, 0] + dx)))
            new_corners[ci, 1] = max(0, min(ny - 1, int(self._roi_anchor[ci, 1] + dy)))
            self._roi = new_corners
        else:
            new_corners = self._roi_anchor + numpy.array([[dx, dy]])
            new_corners[:, 0] = numpy.clip(new_corners[:, 0], 0, nx - 1)
            new_corners[:, 1] = numpy.clip(new_corners[:, 1], 0, ny - 1)
            self._roi = new_corners
        self.lb_roi.config(text=self._roi_label_text(self._roi, "ROI"))

    def _on_img_press(self, ev):
        self._img_widget.focus_set()
        if ev.inaxes is not self.ai: return
        if not self.current_data or _arr(self.current_data).ndim < 3: return
        if self._roi_moving and self._roi is not None:
            ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
            ci = self._find_corner(self._roi, ix, iy)
            self._roi_drag_corner = ci if ci >= 0 else None
            self._roi_drag_start = (ix, iy)
            self._roi_anchor = self._roi.copy()
        elif self._roi2_moving and self._roi2 is not None:
            ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
            ci = self._find_corner(self._roi2, ix, iy)
            self._roi2_drag_corner = ci if ci >= 0 else None
            self._roi2_drag_start = (ix, iy)
            self._roi2_anchor = self._roi2.copy()
        elif self._roi_drawing:
            self._roi_start = (ev.xdata, ev.ydata)
        else:
            ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
            self._cix = ix; self._ciy = iy
            self._draw_img(self.sl_e.get()); self._draw_pt()

    def _on_img_motion(self, ev):
        if ev.inaxes is not self.ai: return
        if self._roi_moving and self._roi_drag_start is not None:
            ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
            dx = ix - self._roi_drag_start[0]
            dy = iy - self._roi_drag_start[1]
            self._move_roi(dx, dy)
            self._draw_img(self.sl_e.get())
            return
        if self._roi2_moving and self._roi2_drag_start is not None:
            ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
            dx = ix - self._roi2_drag_start[0]
            dy = iy - self._roi2_drag_start[1]
            self._move_roi2(dx, dy)
            self._draw_img(self.sl_e.get())
            return
        if not self._roi_drawing or self._roi_start is None: return
        ix0, iy0 = self._cal_to_pix(self._roi_start[0], self._roi_start[1])
        ix1, iy1 = self._cal_to_pix(ev.xdata, ev.ydata)
        lo_x, hi_x = min(ix0, ix1), max(ix0, ix1)
        lo_y, hi_y = min(iy0, iy1), max(iy0, iy1)
        mid_y = (lo_y + hi_y) // 2
        self._roi = numpy.array([[lo_x, lo_y], [lo_x, mid_y], [lo_x, hi_y],
                                 [hi_x, hi_y], [hi_x, mid_y], [hi_x, lo_y]])
        self.lb_roi.config(text=self._roi_label_text(self._roi, "ROI"))
        self._draw_img(self.sl_e.get())

    def _on_img_release(self, ev):
        if self._roi_moving and self._roi_drag_start is not None:
            if ev.inaxes is self.ai:
                ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
                dx = ix - self._roi_drag_start[0]
                dy = iy - self._roi_drag_start[1]
                self._move_roi(dx, dy)
            self._roi_drag_start = None; self._roi_anchor = None
            self._roi_drag_corner = None
            self._draw_img(self.sl_e.get()); self._draw_pt()
            return
        if self._roi2_moving and self._roi2_drag_start is not None:
            if ev.inaxes is self.ai:
                ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
                dx = ix - self._roi2_drag_start[0]
                dy = iy - self._roi2_drag_start[1]
                self._move_roi2(dx, dy)
            self._roi2_drag_start = None; self._roi2_anchor = None
            self._roi2_drag_corner = None
            self._draw_img(self.sl_e.get()); self._draw_pt()
            return
        if not self._roi_drawing or self._roi_start is None: return
        if ev.inaxes is not self.ai: return
        ix0, iy0 = self._cal_to_pix(self._roi_start[0], self._roi_start[1])
        ix1, iy1 = self._cal_to_pix(ev.xdata, ev.ydata)
        lo_x, hi_x = min(ix0, ix1), max(ix0, ix1)
        lo_y, hi_y = min(iy0, iy1), max(iy0, iy1)
        mid_y = (lo_y + hi_y) // 2
        self._roi = numpy.array([[lo_x, lo_y], [lo_x, mid_y], [lo_x, hi_y],
                                 [hi_x, hi_y], [hi_x, mid_y], [hi_x, lo_y]])
        self._roi_start = None; self._roi_drawing = False
        self.btn_roi.config(relief=tk.RAISED, text="Draw ROI")
        self.lb_roi.config(text=self._roi_label_text(self._roi, "ROI"))
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _roi_spectrum(self):
        if not self.current_data or self._roi is None: return None
        d = _arr(self.current_data)
        if d.ndim < 3: return None
        mask = self._poly_mask(self._roi)
        if not mask.any(): return None
        sp = d[mask, :].sum(axis=0)
        xc, u = _spec_cal(self.current_data)
        return xc, u, sp

    # ---- Background ROI interaction ----
    def _place_roi2(self):
        if self._roi is None:
            messagebox.showwarning("No ROI", "Draw ROI first, then place Bkg.")
            return
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        xs = self._roi[:, 0]; ys = self._roi[:, 1]
        w = int(xs.max() - xs.min())
        off = max(3, w // 2)
        dx = min(off, nx - 1 - int(xs.max()))
        dy = min(off, ny - 1 - int(ys.max()))
        dx = max(dx, -int(xs.min()))
        dy = max(dy, -int(ys.min()))
        self._roi2 = self._roi.copy() + numpy.array([[dx, dy]])
        self.lb_roi2.config(text=self._roi_label_text(self._roi2, "Bkg"))
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _toggle_move_roi2(self):
        if self._roi2 is None:
            messagebox.showwarning("No Bkg", "Place Bkg first.")
            return
        self._roi2_moving = not self._roi2_moving
        if self._roi2_moving:
            self._roi_moving = False
            self.btn_mvroi.config(relief=tk.RAISED, text="Move ROI")
            self.btn_mvroi2.config(relief=tk.SUNKEN, text="Moving\u2026")
        else:
            self.btn_mvroi2.config(relief=tk.RAISED, text="Move Bkg")

    def _clear_roi2(self):
        self._roi2 = None; self._roi2_moving = False
        self._roi2_drag_start = None; self._roi2_anchor = None
        self._roi2_drag_corner = None
        self.btn_mvroi2.config(relief=tk.RAISED, text="Move Bkg")
        self.lb_roi2.config(text="Bkg: none")
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _move_roi2(self, dx, dy):
        if self._roi2_anchor is None: return
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        if self._roi2_drag_corner is not None:
            ci = self._roi2_drag_corner
            new_corners = self._roi2_anchor.copy()
            new_corners[ci, 0] = max(0, min(nx - 1, int(self._roi2_anchor[ci, 0] + dx)))
            new_corners[ci, 1] = max(0, min(ny - 1, int(self._roi2_anchor[ci, 1] + dy)))
            self._roi2 = new_corners
        else:
            new_corners = self._roi2_anchor + numpy.array([[dx, dy]])
            new_corners[:, 0] = numpy.clip(new_corners[:, 0], 0, nx - 1)
            new_corners[:, 1] = numpy.clip(new_corners[:, 1], 0, ny - 1)
            self._roi2 = new_corners
        self.lb_roi2.config(text=self._roi_label_text(self._roi2, "Bkg"))

    def _roi2_spectrum(self):
        if not self.current_data or self._roi2 is None: return None
        d = _arr(self.current_data)
        if d.ndim < 3: return None
        mask = self._poly_mask(self._roi2)
        if not mask.any(): return None
        sp = d[mask, :].sum(axis=0)
        xc, u = _spec_cal(self.current_data)
        return xc, u, sp

    # ---- arrow-key nudge for ROI / Bkg ----
    def _nudge_roi(self, dix, diy):
        if self._roi is None or not self.current_data: return
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        new_corners = self._roi + numpy.array([[dix, diy]])
        new_corners[:, 0] = numpy.clip(new_corners[:, 0], 0, nx - 1)
        new_corners[:, 1] = numpy.clip(new_corners[:, 1], 0, ny - 1)
        self._roi = new_corners
        self.lb_roi.config(text=self._roi_label_text(self._roi, "ROI"))
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _nudge_roi2(self, dix, diy):
        if self._roi2 is None or not self.current_data: return
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        new_corners = self._roi2 + numpy.array([[dix, diy]])
        new_corners[:, 0] = numpy.clip(new_corners[:, 0], 0, nx - 1)
        new_corners[:, 1] = numpy.clip(new_corners[:, 1], 0, ny - 1)
        self._roi2 = new_corners
        self.lb_roi2.config(text=self._roi_label_text(self._roi2, "Bkg"))
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _on_arrow_key(self, event):
        if not self.current_data or _arr(self.current_data).ndim < 3: return
        key_map = {"Up": (1, 0), "Down": (-1, 0), "Right": (0, 1), "Left": (0, -1)}
        delta = key_map.get(event.keysym)
        if delta is None: return
        dix, diy = delta
        if self._roi2_moving:
            self._nudge_roi2(dix, diy)
        else:
            self._nudge_roi(dix, diy)

    # ---- point spectrum + ROI overlay ----
    def _normalize(self, arr):
        """Normalize a spectrum array based on current setting."""
        a = arr.astype(float)
        mode = self.var_norm.get()
        if mode == "max":
            mx = numpy.max(numpy.abs(a))
            return a / mx if mx > 0 else a
        elif mode == "area":
            area = numpy.sum(numpy.abs(a))
            return a / area if area > 0 else a
        return a

    def _draw_pt(self):
        if not self.current_data: return
        d = _arr(self.current_data)
        if d.ndim < 3: return
        xc, u = _spec_cal(self.current_data)
        sp_cur = self._normalize(d[self._cix, self._ciy, :])

        # ROI spectrum
        roi_sp = None
        roi_r = self._roi_spectrum()
        if roi_r is not None:
            _, _, roi_raw = roi_r
            roi_sp = self._normalize(roi_raw)

        # ROI2 spectrum
        roi2_sp = None
        roi2_r = self._roi2_spectrum()
        if roi2_r is not None:
            _, _, roi2_raw = roi2_r
            roi2_sp = self._normalize(roi2_raw)

        # Subtract mode
        sub_mode = self.var_sub.get()
        diff_sp = None; lbl_sub = ""
        if sub_mode != "off":
            if sub_mode == "cur_roi" and roi_sp is not None:
                diff_sp = sp_cur - roi_sp; lbl_sub = "cur\u2212ROI"
            elif sub_mode == "roi_cur" and roi_sp is not None:
                diff_sp = roi_sp - sp_cur; lbl_sub = "ROI\u2212cur"
            elif sub_mode == "cur_roi2" and roi2_sp is not None:
                diff_sp = sp_cur - roi2_sp; lbl_sub = "cur\u2212Bkg"
            elif sub_mode == "roi_roi2" and roi_sp is not None and roi2_sp is not None:
                diff_sp = roi_sp - roi2_sp; lbl_sub = "ROI\u2212Bkg"
            elif sub_mode == "roi2_roi" and roi_sp is not None and roi2_sp is not None:
                diff_sp = roi2_sp - roi_sp; lbl_sub = "Bkg\u2212ROI"
            elif sub_mode == "roi2_cur" and roi2_sp is not None:
                diff_sp = roi2_sp - sp_cur; lbl_sub = "Bkg\u2212cur"

        # Compute fitted peak/area/FWHM values
        fit_cur = self._fit_peak(xc, sp_cur)
        if fit_cur:
            pk_cur, area_cur, fw_cur, _, fitted_cur = fit_cur
        else:
            fw_cur = _fwhm(xc, sp_cur.astype(float))
            pk_cur = float(numpy.max(sp_cur)); area_cur = float(numpy.trapz(sp_cur.astype(float), xc))
            fitted_cur = None

        pk_roi = area_roi = fw_roi = fitted_roi = None
        if roi_sp is not None:
            fit_roi = self._fit_peak(xc, roi_sp)
            if fit_roi:
                pk_roi, area_roi, fw_roi, _, fitted_roi = fit_roi
            else:
                fw_roi = _fwhm(xc, roi_sp.astype(float))
                pk_roi = float(numpy.max(roi_sp))

        pk_roi2 = area_roi2 = fw_roi2 = fitted_roi2 = None
        if roi2_sp is not None:
            fit_roi2 = self._fit_peak(xc, roi2_sp)
            if fit_roi2:
                pk_roi2, area_roi2, fw_roi2, _, fitted_roi2 = fit_roi2
            else:
                fw_roi2 = _fwhm(xc, roi2_sp.astype(float))
                pk_roi2 = float(numpy.max(roi2_sp))

        # Y limits — use fitted peak max instead of raw max pixel
        ym = max(pk_cur, 1e-30)
        if pk_roi is not None:
            ym = max(ym, pk_roi)
        if pk_roi2 is not None:
            ym = max(ym, pk_roi2)
        if diff_sp is not None:
            ym = max(ym, float(numpy.max(numpy.abs(diff_sp))))

        r = float(xc[-1]-xc[0])
        sx = self.sxp.get()/100.; sy = self.syp.get()/100.
        _, scx, ux = _gc(self.current_data, 0)
        _, scy, uy = _gc(self.current_data, 1)
        px = scx*(self._cix+.5); py = scy*(self._ciy+.5)

        norm_mode = self.var_norm.get()
        ylbl = "Cts" if norm_mode == "off" else ("Norm (max)" if norm_mode == "max" else "Norm (area)")

        self.ap.clear()

        # Build annotation parts with fit info
        fit_mode = self.var_fit.get()
        ann_parts = [f"[{fit_mode}] cur: pk={pk_cur:.4g} FWHM={fw_cur:.4g}{u} A={area_cur:.4g}"]
        if fw_roi is not None:
            ann_parts.append(f"ROI: pk={pk_roi:.4g} FWHM={fw_roi:.4g}{u}" +
                             (f" A={area_roi:.4g}" if area_roi is not None else ""))
        if fw_roi2 is not None:
            ann_parts.append(f"Bkg: pk={pk_roi2:.4g} FWHM={fw_roi2:.4g}{u}" +
                             (f" A={area_roi2:.4g}" if area_roi2 is not None else ""))

        # Plot cursor
        self.ap.plot(xc, sp_cur, lw=0.6, color="tab:orange",
                     label=f"cursor (pk={pk_cur:.4g} FWHM={fw_cur:.4g})")
        if fitted_cur is not None:
            self.ap.plot(xc, fitted_cur, lw=0.8, ls="--", color="tab:orange", alpha=0.7)

        # Plot ROI
        if roi_sp is not None:
            self.ap.plot(xc, roi_sp, lw=0.6, color="lime",
                         label=f"ROI (pk={pk_roi:.4g} FWHM={fw_roi:.4g})")
            if fitted_roi is not None:
                self.ap.plot(xc, fitted_roi, lw=0.8, ls="--", color="lime", alpha=0.7)

        # Plot ROI2
        if roi2_sp is not None:
            self.ap.plot(xc, roi2_sp, lw=0.6, color="cyan",
                         label=f"Bkg (pk={pk_roi2:.4g} FWHM={fw_roi2:.4g})")
            if fitted_roi2 is not None:
                self.ap.plot(xc, fitted_roi2, lw=0.8, ls="--", color="cyan", alpha=0.7)

        # Plot difference
        if diff_sp is not None:
            self.ap.plot(xc, diff_sp, lw=0.6, color="tab:red",
                         label=lbl_sub)

        self.ap.legend(fontsize=_FP, loc="upper right")

        self.ap.set_xlim(-(r*sx)/2, (r*sx)/2)
        top = ym*sy
        if self.vsp.get() == "log":
            self.ap.set_yscale("log")
            all_sp = [sp_cur]
            if roi_sp is not None: all_sp.append(roi_sp)
            if roi2_sp is not None: all_sp.append(roi2_sp)
            mn_vals = []
            for s in all_sp:
                if numpy.any(s > 0): mn_vals.append(float(s[s > 0].min()))
            mn = max(1e-30, float(min(mn_vals))) if mn_vals else 1e-30
            self.ap.set_ylim(mn, top*1.05)
        else:
            ylo = 0
            if diff_sp is not None:
                ylo = min(0, float(numpy.min(diff_sp)) * sy)
            self.ap.set_yscale("linear"); self.ap.set_ylim(ylo, top)
        self.ap.set_xlabel(f"Energy ({u})" if u else "Ch", fontsize=_FP)
        self.ap.set_ylabel(ylbl, fontsize=_FP)
        self.ap.set_title(f"Px({self._cix},{self._ciy}) [{px:.3g}{ux},{py:.3g}{uy}]", fontsize=_FP)
        self.ap.tick_params(labelsize=_FP)

        # Text annotation with fit values in the plot
        self.ap.text(0.01, 0.97, "\n".join(ann_parts),
                     transform=self.ap.transAxes, fontsize=_FP,
                     verticalalignment="top", fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Clickable vertical energy marker
        if self._pt_vline_energy is not None:
            ev = self._pt_vline_energy
            self.ap.axvline(ev, color="red", lw=1.0, ls="--", alpha=0.85)
            ylims = self.ap.get_ylim()
            ytxt = ylims[1] - (ylims[1] - ylims[0]) * 0.04
            self.ap.text(ev, ytxt, f"{ev:.4g} {u}", color="red", fontsize=_FP,
                         ha="center", va="top",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

        self.fp.tight_layout(); self.cp.draw_idle()

    def _on_pt_click(self, ev):
        """Place / update the vertical energy marker on the bottom spectrum plot."""
        if ev.inaxes is not self.ap: return
        self._pt_vline_energy = ev.xdata
        self._draw_pt()

    # ---- file actions ----
    def _comment_suffix(self):
        """Return sanitized comment string for filenames, or empty string."""
        import re
        c = self.comment_entry.get().strip()
        if not c: return ""
        c = re.sub(r'[^\w\s\-]', '', c)
        c = re.sub(r'\s+', '_', c)
        return "_" + c if c else ""

    def _open_file(self):
        p = filedialog.askopenfilename(title="Open NHDF",
            filetypes=[("NHDF", "*.nhdf *.h5 *.hdf5"), ("All", "*.*")])
        if not p: return
        try:
            self.current_path = pathlib.Path(p)
            self.current_data = read_data_and_metadata(self.current_path)
            self.file_lbl.config(text=str(self.current_path), fg="black")
            self.text.config(state=tk.NORMAL); self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, format_info(self.current_data))
            self.text.config(state=tk.DISABLED)
            self.status.config(text=f"Loaded: {self.current_path.name}")
            self._upd_plot(); self._upd_info(); self._setup_esl()
        except Exception as e: 
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _save_file(self):
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        cs = self._comment_suffix()
        dp = _date_prefix()
        dn = dp+(self.current_path.stem+cs+"_copy.nhdf" if self.current_path else "out.nhdf")
        p = filedialog.asksaveasfilename(title="Save NHDF", defaultextension=".nhdf",
            initialfile=dn, filetypes=[("NHDF", "*.nhdf")])
        if not p: return
        try:
            save_data_and_metadata(pathlib.Path(p), self.current_data)
            messagebox.showinfo("Saved", f"→ {p}")
        except Exception as e: 
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _export_csv(self):
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        d = _arr(self.current_data); cs = self._comment_suffix()
        if d.ndim <= 2:
            p = filedialog.asksaveasfilename(title="CSV", defaultextension=".csv",
                initialfile=(_date_prefix()+(self.current_path.stem+cs+".csv" if self.current_path else "o.csv")),
                filetypes=[("CSV", "*.csv")])
            if not p: return
            try:
                numpy.savetxt(p, d if d.ndim == 2 else d.reshape(1, -1), delimiter=",", fmt="%.8g")
                messagebox.showinfo("OK", f"→ {p}")
            except Exception as e: 
                traceback.print_exc()
                messagebox.showerror("Error", str(e))
        else:
            fld = filedialog.askdirectory(title="Folder")
            if not fld: return
            try:
                b = _date_prefix() + (self.current_path.stem if self.current_path else "data") + cs
                out = pathlib.Path(fld); n = d.shape[0]
                for i in range(n):
                    s = d[i]
                    if s.ndim > 2: s = s.reshape(-1, s.shape[-1])
                    numpy.savetxt(str(out/f"{b}_{i:04d}.csv"), s, delimiter=",", fmt="%.8g")
                messagebox.showinfo("OK", f"{n} CSVs → {fld}")
            except Exception as e: 
                traceback.print_exc()
                messagebox.showerror("Error", str(e))

    def _export_sum(self):
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        r = self._sum_sp()
        if not r: return
        xc, u, sp = r; cs = self._comment_suffix()
        p = filedialog.asksaveasfilename(title="Sum spectrum", defaultextension=".csv",
            initialfile=(_date_prefix()+(self.current_path.stem+cs+"_sum.csv" if self.current_path else "sum.csv")),
            filetypes=[("CSV", "*.csv")])
        if not p: return
        try:
            h = f"{'energy('+u+')' if u else 'ch'},counts"
            numpy.savetxt(p, numpy.column_stack((xc, sp)), delimiter=",", fmt="%.8g",
                          header=h, comments="")
            messagebox.showinfo("OK", f"→ {p}")
        except Exception as e: 
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _export_cube(self):
        """Export raw datacube as binary .raw — dimensions x,y,z(energy) in filename."""
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        d = _arr(self.current_data)
        if d.ndim < 3:
            messagebox.showinfo("", f"Data is {d.ndim}D, not a 3D cube."); return
        nx, ny, nz = d.shape[0], d.shape[1], d.shape[2]
        dt = d.dtype.name
        base = self.current_path.stem if self.current_path else "data"
        cs = self._comment_suffix()
        default_name = f"{_date_prefix()}{base}{cs}_x{nx}_y{ny}_z{nz}_{dt}.raw"
        p = filedialog.asksaveasfilename(title="Export datacube", defaultextension=".raw",
            initialfile=default_name,
            filetypes=[("Raw binary", "*.raw"), ("All", "*.*")])
        if not p: return
        try:
            out = numpy.ascontiguousarray(d)
            if out.dtype.byteorder not in ("=", "|", "<" if numpy.little_endian else ">"):
                out = out.astype(out.dtype.newbyteorder("="))
            raw_bytes = out.tobytes(order="C")
            dest = os.path.abspath(p)
            with open(dest, "wb") as fp:
                written = fp.write(raw_bytes)
            expected = out.size * out.dtype.itemsize
            if written != expected:
                raise IOError(f"Wrote {written} bytes but expected {expected}")
            self.status.config(text=f"Cube: {os.path.basename(dest)}")
            messagebox.showinfo("Exported",
                f"Raw datacube saved ({written:,} bytes)\n"
                f"Layout: C-order (x, y, z_energy)\n"
                f"x={nx}  y={ny}  z(energy)={nz}\n"
                f"Dtype: {dt}  ({out.dtype.itemsize} bytes/value)\n"
                f"→ {dest}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Export error", f"{type(e).__name__}: {e}")

    def _export_roi(self):
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        r = self._roi_spectrum()
        if r is None:
            messagebox.showwarning("No ROI", "Draw an ROI on the image first.\n"
                                   "Click 'Draw ROI', then drag a rectangle.")
            return
        xc, u, sp = r
        xs = self._roi[:, 0]; ys = self._roi[:, 1]
        base = self.current_path.stem if self.current_path else "data"
        cs = self._comment_suffix()
        default_name = f"{_date_prefix()}{base}{cs}_spectrum_roi_x{xs.min()}-{xs.max()}_y{ys.min()}-{ys.max()}.csv"
        p = filedialog.asksaveasfilename(title="Export ROI spectrum", defaultextension=".csv",
            initialfile=default_name, filetypes=[("CSV", "*.csv")])
        if not p: return
        try:
            h = f"{'energy('+u+')' if u else 'channel'},counts"
            numpy.savetxt(p, numpy.column_stack((xc, sp)), delimiter=",", fmt="%.8g",
                          header=h, comments="")
            npix = int(self._poly_mask(self._roi).sum())
            self.status.config(text=f"ROI: {pathlib.Path(p).name}")
            messagebox.showinfo("Exported",
                f"ROI x[{xs.min()}-{xs.max()}] y[{ys.min()}-{ys.max()}]\n"
                f"Pixels summed: {npix}\n"
                f"Channels: {sp.size}\n→ {p}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _export_bkg(self):
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        r = self._roi2_spectrum()
        if r is None:
            messagebox.showwarning("No Bkg", "Place a background ROI first.\n"
                                   "Click 'Place Bkg' to create one.")
            return
        xc, u, sp = r
        xs = self._roi2[:, 0]; ys = self._roi2[:, 1]
        base = self.current_path.stem if self.current_path else "data"
        cs = self._comment_suffix()
        default_name = f"{_date_prefix()}{base}{cs}_background_bkg_x{xs.min()}-{xs.max()}_y{ys.min()}-{ys.max()}.csv"
        p = filedialog.asksaveasfilename(title="Export Bkg spectrum", defaultextension=".csv",
            initialfile=default_name, filetypes=[("CSV", "*.csv")])
        if not p: return
        try:
            h = f"{'energy('+u+')' if u else 'channel'},counts"
            numpy.savetxt(p, numpy.column_stack((xc, sp)), delimiter=",", fmt="%.8g",
                          header=h, comments="")
            npix = int(self._poly_mask(self._roi2).sum())
            self.status.config(text=f"Bkg: {pathlib.Path(p).name}")
            messagebox.showinfo("Exported",
                f"Bkg x[{xs.min()}-{xs.max()}] y[{ys.min()}-{ys.max()}]\n"
                f"Pixels summed: {npix}\n"
                f"Channels: {sp.size}\n→ {p}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _export_sub(self):
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        d = _arr(self.current_data)
        if d.ndim < 3: return
        xc, u = _spec_cal(self.current_data)
        sp_cur = self._normalize(d[self._cix, self._ciy, :])
        roi_sp = None
        roi_r = self._roi_spectrum()
        if roi_r is not None:
            roi_sp = self._normalize(roi_r[2])
        roi2_sp = None
        roi2_r = self._roi2_spectrum()
        if roi2_r is not None:
            roi2_sp = self._normalize(roi2_r[2])
        sub_mode = self.var_sub.get()
        diff_sp = None; lbl_sub = ""
        if sub_mode == "cur_roi" and roi_sp is not None:
            diff_sp = sp_cur - roi_sp; lbl_sub = "cur-ROI"
        elif sub_mode == "roi_cur" and roi_sp is not None:
            diff_sp = roi_sp - sp_cur; lbl_sub = "ROI-cur"
        elif sub_mode == "cur_roi2" and roi2_sp is not None:
            diff_sp = sp_cur - roi2_sp; lbl_sub = "cur-Bkg"
        elif sub_mode == "roi_roi2" and roi_sp is not None and roi2_sp is not None:
            diff_sp = roi_sp - roi2_sp; lbl_sub = "ROI-Bkg"
        elif sub_mode == "roi2_roi" and roi_sp is not None and roi2_sp is not None:
            diff_sp = roi2_sp - roi_sp; lbl_sub = "Bkg-ROI"
        elif sub_mode == "roi2_cur" and roi2_sp is not None:
            diff_sp = roi2_sp - sp_cur; lbl_sub = "Bkg-cur"
        if diff_sp is None:
            messagebox.showwarning("No subtraction",
                "Select a Subtract mode and ensure the required ROIs exist.")
            return
        norm_mode = self.var_norm.get()
        norm_tag = "_norm_off" if norm_mode == "off" else f"_norm_{norm_mode}"
        base = self.current_path.stem if self.current_path else "data"
        cs = self._comment_suffix()
        default_name = f"{_date_prefix()}{base}{cs}_sub_{lbl_sub}{norm_tag}.csv"
        p = filedialog.asksaveasfilename(title=f"Export {lbl_sub}", defaultextension=".csv",
            initialfile=default_name, filetypes=[("CSV", "*.csv")])
        if not p: return
        try:
            col_name = lbl_sub + (f"({norm_mode})" if norm_mode != "off" else "")
            h = f"{'energy('+u+')' if u else 'channel'},{col_name}"
            numpy.savetxt(p, numpy.column_stack((xc, diff_sp)), delimiter=",", fmt="%.8g",
                          header=h, comments="")
            self.status.config(text=f"Sub: {pathlib.Path(p).name}")
            messagebox.showinfo("Exported",
                f"{lbl_sub} spectrum (norm: {norm_mode})\n→ {p}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    NHDFApp(root)
    root.mainloop()