"""
GUI application to read and save NHDF (Nion HDF5) data and metadata files.
v3pt6 — Temperature map uses fixed 0.04–0.08 eV window; extensive comments throughout.

Required packages:
  pip install numpy h5py niondata matplotlib
"""

# =============================================================================
# Standard library imports
# =============================================================================
import h5py          # HDF5 file reading/writing (NHDF files are HDF5 under the hood)
import json          # Parse / serialise JSON metadata stored in HDF5 attributes
import numpy         # Numerical array operations — the workhorse for all spectrum math
import numpy.typing  # Type hints for numpy arrays (NDArray)
import os            # File path helpers (dirname, abspath, path manipulation)
import pathlib       # Object-oriented path handling; preferred over raw os.path strings
import pprint        # Pretty-print the NHDF metadata dictionary in the info text box
import tkinter as tk                                      # Main GUI toolkit
from tkinter import filedialog, messagebox, scrolledtext  # Standard dialogs & widgets
import traceback     # Print full stack traces on exceptions for easier debugging
import typing        # Python typing extras (cast, Dict, Any, …)
from scipy.signal import find_peaks   # Automatic peak detection in the summed spectrum
from scipy.optimize import curve_fit  # Nonlinear least-squares peak fitting
from scipy.special import lambertw, erf  # Special functions used in GaussLor FWHM formula
import datetime      # Date/time prefix for exported filenames

# =============================================================================
# Matplotlib imports — use the TkAgg backend so plots embed in a Tk window
# =============================================================================
import matplotlib
matplotlib.use("TkAgg")                                   # Must be called BEFORE pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Tk canvas widget
from matplotlib.figure import Figure                       # Matplotlib figure container
from matplotlib.patches import Rectangle, Polygon as MplPolygon  # ROI overlay shapes
from matplotlib.path import Path as MplPath               # Point-in-polygon test

# =============================================================================
# Nion data model imports
# =============================================================================
# nion.data is the core data model used by Nion Swift (the microscope control
# software that produces NHDF files).  It defines:
#   Calibration.Calibration — holds (offset, scale, units) for one axis
#   DataAndMetadata          — wraps a numpy array with calibration metadata
#   Converter                — converts timestamps between Python datetime and
#                              the ISO-8601 strings stored in NHDF attributes
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.utils import Converter

# =============================================================================
# Optional converter import — DAT / DM3 export functions
# =============================================================================
# write_raw()           — exports a 3-D float32 array as a raw binary .dat file
#                         + a human-readable .txt sidecar with calibrations.
# build_dm3_from_nhdf() — assembles a GMS-compatible DM3 spectrum image in
#                         memory and returns the raw bytes.
#
# Both live in nhdf_converter_GUI_v0pt5.py which must be in the same directory.
# We insert this script's directory into sys.path so the import always finds it
# regardless of the working directory the user launches from.
#
# If the import fails (file missing, dependency error) we store the error
# message in _CONVERTER_ERR and set _CONVERTER_OK = False.  Export buttons
# that need the converter check _CONVERTER_OK before proceeding.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from nhdf_converter_GUI_v0pt5 import write_raw, build_dm3_from_nhdf
    _CONVERTER_OK = True
    _CONVERTER_ERR = ""
except ImportError as _ie:
    write_raw = build_dm3_from_nhdf = None
    _CONVERTER_OK = False
    _CONVERTER_ERR = str(_ie)

# =============================================================================
# Module-level constants
# =============================================================================
_NDArray = numpy.typing.NDArray[typing.Any]  # Convenience alias for typed arrays

# Golden ratio — not currently used in layout but reserved for future aspect-
# ratio calculations (e.g., plot proportions).
PHI = 1.618033988749895

# Tkinter font tuples used throughout the GUI.  Keeping them as module-level
# constants means a single change here resizes all labels / buttons at once.
_FNT    = ("Arial", 8)           # Default font for most labels and buttons
_FNT_M  = ("Arial", 8)           # Monospace alternative (currently same size)
_FNT_MB = ("Arial", 8, "bold")   # Bold variant for info readouts

# Font size in *points* for Matplotlib tick labels and axis titles.
# A value of 7 keeps labels legible at the compact figure sizes used here.
_FP = 7

# Ordered list of Matplotlib colormaps offered in the spatial-image panel.
# "inferno" is the default (good contrast for phonon intensity maps).
_CMAPS = ["inferno", "viridis", "gray", "cubehelix", "magma", "plasma", "hot", "coolwarm",
          "cividis", "turbo", "bone", "copper", "twilight"]

# Boltzmann constant in eV/K — used by the detailed balance temperature
# calculation:  T = 1 / (k_B * slope),  slope in units of eV^{-1}.
# NIST 2018 CODATA value: 8.617333262×10^{-5} eV/K.
_kB_eV = 8.617333e-5   # Boltzmann constant in eV/K


# ================================================================
# NHDF read / write
# ================================================================

def read_data_and_metadata(path: pathlib.Path) -> DataAndMetadata.DataAndMetadata:
    """Load an NHDF file and return a fully populated DataAndMetadata object.

    NHDF (Nion HDF5) is the native file format of Nion Swift.  The file
    structure is:

        /data/
            <key>/           ← HDF5 dataset containing the raw numpy array
                attrs["properties"]  ← JSON string with calibrations, descriptor,
                                        timestamp, timezone, etc.

    The JSON "properties" block encodes everything the Nion data model needs
    to reconstruct a DataAndMetadata object:
      - is_sequence / collection_dimension_count / datum_dimension_count
        describe how the axes are *interpreted* (e.g., a 3-D EELS spectrum
        image has collection_dimensions=2 and datum_dimensions=1).
      - intensity_calibration / dimensional_calibrations hold (offset, scale,
        units) tuples for each axis.
      - metadata is an arbitrary dict of acquisition parameters.
      - created / timezone / timezone_offset carry the timestamp.

    The lambda `data_fn` wraps the already-loaded numpy array so that
    DataAndMetadata can be constructed with its expected "callable data"
    interface.  The `.shape` and `.dtype` attributes are patched onto the
    lambda so DataAndMetadata can inspect them without calling it.
    """
    with h5py.File(str(path), "r") as f:
        # Navigate to the first (and normally only) dataset under /data/
        dg = f["data"]; key0 = list(sorted(dg.keys()))[0]; ds = dg[key0]

        # Parse the JSON properties string stored as an HDF5 attribute
        jp = json.loads(ds.attrs["properties"])

        # Load the raw pixel/count data into memory as a numpy array
        data = numpy.array(ds)

        # Reconstruct the DataDescriptor that tells the data model how the
        # axes are logically grouped (sequence, collection, datum)
        dd = DataAndMetadata.DataDescriptor(
            is_sequence=jp.get("is_sequence", False),
            collection_dimension_count=jp.get("collection_dimension_count", 0),
            datum_dimension_count=jp.get("datum_dimension_count", 0))

        # Reconstruct the DataMetadata which holds shape, dtype, and all
        # calibrations.  from_rpc_dict converts the stored dict into a
        # Calibration.Calibration object.
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

        # DataAndMetadata expects a callable that returns the array.
        # We also attach .shape and .dtype directly so the object can
        # introspect the array without calling data_fn().
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
    """Write a DataAndMetadata object back to disk in NHDF (Nion HDF5) format.

    This is the inverse of read_data_and_metadata().  It:
      1. Assembles a "properties" dict from the calibration and descriptor
         fields of the DataAndMetadata object.
      2. Creates a new HDF5 file with a /data/0 dataset.
      3. Stores the properties dict as a JSON string in the dataset attribute.

    The saved file can be re-opened by Nion Swift and by read_data_and_metadata()
    without loss of calibration information.
    """
    props: typing.Dict[str, typing.Any] = {}

    # Encode the data descriptor so the reader knows how to interpret the axes
    props["is_sequence"] = d.data_descriptor.is_sequence
    props["collection_dimension_count"] = d.data_descriptor.collection_dimension_count
    props["datum_dimension_count"] = d.data_descriptor.datum_dimension_count

    # Calibrations are serialised to plain dicts via .rpc_dict (the Nion wire
    # format — a JSON-serialisable representation of each Calibration object)
    if d.intensity_calibration: props["intensity_calibration"] = d.intensity_calibration.rpc_dict
    if d.dimensional_calibrations:
        props["dimensional_calibrations"] = [c.rpc_dict for c in d.dimensional_calibrations]
    if d.metadata: props["metadata"] = d.metadata

    # Timestamp is stored as an ISO-8601 string via the Converter utility
    if d.timestamp: props["created"] = Converter.DatetimeToStringConverter().convert(d.timestamp)
    if d.timezone: props["timezone"] = d.timezone
    if d.timezone_offset: props["timezone_offset"] = d.timezone_offset

    with h5py.File(str(path), "w") as f:
        # Create /data group and a single dataset named "0" containing the array
        dg = f.create_group("data"); ds = dg.create_dataset("0", data=d.data)
        # Serialise the properties dict to JSON and store as an HDF5 attribute
        ds.attrs["properties"] = json.dumps(props)


# ================================================================
# Standalone helper functions
# ================================================================

def format_info(d: DataAndMetadata.DataAndMetadata) -> str:
    """Format a human-readable summary of a DataAndMetadata object.

    Returns a multi-line string shown in the scrolled text box at the
    bottom of the GUI after a file is loaded.  Includes array shape,
    dtype, dimension descriptor flags, calibrations, timestamp, and
    the full metadata dictionary.
    """
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
    """Get calibration tuple (offset, scale, units) for a given dimension index.

    Parameters
    ----------
    d   : DataAndMetadata — the loaded data object
    dim : int             — axis index (0=rows/Y, 1=cols/X, 2=energy for a 3-D SI)

    Returns
    -------
    (offset, scale, units) — floats and string, with sensible defaults (0, 1, "")
    if the calibration list is absent or shorter than `dim`.

    Used throughout the export functions to package calibration tuples
    for write_raw() and build_dm3_from_nhdf().
    """
    c = d.dimensional_calibrations
    if c and dim < len(c):
        v = c[dim]; return (v.offset or 0., v.scale or 1., v.units or "")
    return 0., 1., ""


def _spec_cal(d):
    """Return the calibrated energy axis of a DataAndMetadata object.

    For a 3-D spectrum image (ny, nx, ne) the energy calibration is on
    axis 2.  For 1-D or 2-D data it falls back to the last axis.

    Returns
    -------
    x : 1-D numpy array of length ne — energy values in calibrated units
        (typically eV), computed as:  x[i] = offset + scale * i
    u : str — units string (usually "eV"), empty string if not set.

    If no calibration is found, returns channel indices as floats and "".
    """
    c = d.dimensional_calibrations
    # For a 3-D spectrum image the energy axis is always dimension 2;
    # fall back to the last calibration if fewer than 3 dims are present.
    cal = c[2] if c and len(c) >= 3 else (c[-1] if c else None)
    n = d.data.shape[-1]   # number of energy channels
    if cal:
        o = cal.offset or 0.; s = cal.scale or 1.; u = cal.units or ""
        # Generate calibrated energy values: E[i] = offset + scale * i
        return o + s * numpy.arange(n), u
    # No calibration metadata — return plain channel indices
    return numpy.arange(n, dtype=float), ""


def _e0idx(d):
    """Return the channel index closest to E = 0 (the zero-loss peak centre).

    This is used to find where the ZLP sits in channel space so that
    loss-side (positive E) and gain-side (negative E) channels can be
    identified by symmetric offsets around this index.
    """
    x, _ = _spec_cal(d); return int(numpy.argmin(numpy.abs(x)))


def _fwhm(x, y):
    """Estimate Full-Width at Half-Maximum of a spectrum peak by threshold crossing.

    This is a fast, model-independent estimate used as a fallback when the
    curve_fit peak fit fails.  It finds the leftmost and rightmost channels
    where y >= max(y)/2 and returns their energy difference.

    Returns 0.0 if fewer than 2 channels exceed the half-maximum.
    """
    hm = numpy.max(y) / 2.
    i = numpy.where(y >= hm)[0]
    # Distance between the outermost channels above the half-max threshold
    return float(x[i[-1]] - x[i[0]]) if len(i) >= 2 else 0.


# ================================================================
# Peak fit model functions and FWHM formulae
# ================================================================
# Three lineshape models are supported, selectable via radio buttons:
#   Gaussian  — symmetric, good for ZLP and phonon peaks far from tails
#   Lorentzian — heavier tails, better for broad or lifetime-broadened peaks
#   GaussLor  — product of Gaussian and Lorentzian; empirically best for ZLP
#
# All models share the parameter convention: A = amplitude, x0 = centre,
# sigma/gamma = width parameter (NOT FWHM — FWHM computed separately below).
# ================================================================

def _gauss_fn(x, A, x0, sigma):
    """Gaussian lineshape: A * exp(-0.5 * ((x - x0) / sigma)^2)."""
    return A * numpy.exp(-(x - x0)**2 / (2 * sigma**2))

def _lorentz_fn(x, A, x0, gamma):
    """Lorentzian (Cauchy) lineshape: A / (1 + ((x - x0)/gamma)^2)."""
    return A / (1 + ((x - x0) / gamma)**2)

def _gausslor_fn(x, A, x0, sigma):
    """GaussLor (pseudo-Voigt product) lineshape.

    Defined as Gaussian × Lorentzian with the same width parameter sigma.
    This is not the standard Voigt convolution but is computationally cheap
    and empirically matches the ZLP shape in monochromated STEM-EELS data.
    """
    return A * numpy.exp(-(x - x0)**2 / (2 * sigma**2)) / (1 + ((x - x0) / sigma)**2)


# FWHM formulae — each converts the fitted width parameter to the full
# width at half maximum in the same units as the energy axis (eV).

def _fwhm_gauss(sigma):
    """FWHM of a Gaussian from its sigma parameter: FWHM = 2.355 * |sigma|."""
    return 2 * abs(sigma) * numpy.sqrt(2 * numpy.log(2))

def _fwhm_lorentz(gamma):
    """FWHM of a Lorentzian from its gamma parameter: FWHM = 2 * |gamma|."""
    return 2 * abs(gamma)

def _fwhm_gausslor(sigma):
    """FWHM of the GaussLor product lineshape.

    The half-maximum occurs where exp(-t^2/2) / (1 + t^2) = 0.5
    (with t = (x - x0)/sigma).  This has no closed-form solution; we use
    the Lambert W function to solve it analytically.  The result is:
        FWHM = 2 * |sigma| * sqrt(2*W(sqrt(e)) - 1)
    where W is the principal branch of the Lambert W function.
    """
    w = float(numpy.real(lambertw(numpy.sqrt(numpy.e))))
    return 2 * abs(sigma) * numpy.sqrt(2 * w - 1)


def _arr(dam):
    """Unwrap the data array from a DataAndMetadata object.

    DataAndMetadata stores data as a callable (lambda returning the array)
    to defer loading.  _arr() handles both the callable and the direct-array
    cases so all call sites can use a single pattern.
    """
    v = dam.data
    return v() if callable(v) else v


def _date_prefix():
    """Return the current date as a filename-safe prefix string (YYYYMMDD_).

    Prepended to all exported filenames so files are chronologically sorted
    in the file browser and the export date is always traceable.
    """
    return datetime.datetime.now().strftime("%Y%m%d_")


# ================================================================
# Main application class
# ================================================================

class NHDFApp:
    """Main GUI application for reading, visualising and exporting NHDF data.

    Layout overview (top → bottom inside a vertically scrollable canvas):
      1. Toolbar — Open / Save / Export buttons
      2. File label + Comment entry
      3. Fit model radio buttons (Gaussian / Lorentzian / GaussLor)
      4. Info readout bar (peak max, area, FWHM, bin sizes)
      5. Two summed-spectrum plots (full log-scale + capped linear-scale)
      6. Spatial image panel with energy slider, ROI/Bkg controls, colourmap
      7. Point spectrum panel showing cursor pixel, ROI, Bkg and difference
      8. Scrolled text box with full NHDF metadata
      9. Status bar

    State variables set in __init__:
      current_data — the loaded DataAndMetadata object (None until opened)
      current_path — pathlib.Path to the open file
      _xc, _sp, _su — energy axis array, summed spectrum, units string
      _cb  — Matplotlib colorbar handle (replaced on each _draw_img call)
      _cix/_ciy — cursor pixel coordinates (dim-0 row, dim-1 col indices)
      _roi  — (N,2) int array of polygon corner pixels for the signal ROI
               (None if no ROI drawn)
      _roi_drawing — True while the user is actively dragging a new ROI
      _roi_start   — (xdata, ydata) of the mouse-press that started the drag
      _roi_moving  — True while Move ROI mode is active
      _roi_drag_start, _roi_anchor, _roi_drag_corner — drag-state bookkeeping
      _roi2 and mirror fields — same state for the background (Bkg) ROI
      _pt_vline_energy — energy position of the vertical marker on the
                          point-spectrum plot (None if not set)
    """

    def __init__(self, root):
        # Store a reference to the Tk root window and set the window title / size
        self.root = root; self.root.title("NHDF Reader / Writer")
        self.root.geometry("1100x920")

        # No file is loaded yet — initialise all data-holding attributes to None/zero
        self.current_data = None; self.current_path = None

        # _xc: calibrated energy axis (1-D array, eV)
        # _sp: summed spectrum across all spatial pixels (1-D array, counts)
        # _su: energy units string (usually "eV")
        self._xc = None; self._sp = None; self._su = ""

        # _cb: handle to the current Matplotlib colorbar; kept so it can be
        #      cleared before a new one is added in _draw_img().
        # _cix/_ciy: row/column index of the cursor pixel displayed in the
        #            point-spectrum panel.
        self._cb = None; self._cix = 0; self._ciy = 0

        # Signal ROI state — a polygon defined by N corner pixels.
        # _roi: None (no ROI) or an (N, 2) integer ndarray of (row, col) corners.
        # _roi_drawing: True while the user is mid-drag drawing a new ROI.
        # _roi_start: (xdata, ydata) of the initial mouse press in calibrated units.
        # _roi_moving: True when "Move ROI" mode is active (drag = translate/resize).
        # _roi_drag_start: pixel coordinates where the drag began.
        # _roi_anchor: snapshot of self._roi at the start of the drag (used to
        #              compute incremental displacement without cumulative error).
        # _roi_drag_corner: index into self._roi of the corner being dragged,
        #                   or None if the whole polygon is being translated.
        self._roi = None; self._roi_drawing = False; self._roi_start = None
        self._roi_moving = False; self._roi_drag_start = None; self._roi_anchor = None

        # Background ROI — identical structure to the signal ROI above.
        # Used as the reference background for area-normalised subtraction and
        # for the detailed-balance temperature map.
        self._roi2 = None; self._roi2_moving = False
        self._roi2_drag_start = None; self._roi2_anchor = None
        self._roi_drag_corner = None; self._roi2_drag_corner = None

        # Energy position of the vertical red marker line in the point-spectrum
        # plot (placed by clicking on that plot).  None = not set.
        self._pt_vline_energy = None

        # Build all Tk widgets.  This call creates every label, button,
        # slider, canvas and plot and packs them into self.root.
        self._build_ui()

    def _build_ui(self):
        """Construct and pack all GUI widgets.

        The outermost container is a vertically-scrollable canvas so the full
        GUI fits on screens smaller than ~1100 × 920 pixels.  All widgets are
        packed into self._inn (the inner frame sitting inside that canvas).
        """
        # ---- scrollable outer container ----
        # ct  : outer frame that fills the root window
        # cv  : Tk Canvas — the scrollable viewport
        # vs  : vertical scrollbar linked to cv
        # _inn: inner Frame placed inside cv via create_window(); all widgets
        #       go into _inn so the scrollregion updates when content grows
        ct = tk.Frame(self.root); ct.pack(fill=tk.BOTH, expand=True)
        cv = tk.Canvas(ct); vs = tk.Scrollbar(ct, orient=tk.VERTICAL, command=cv.yview)
        self._inn = tk.Frame(cv)
        # Whenever _inn resizes (e.g., after a new widget is packed), update
        # the canvas scrollregion so the scrollbar thumb moves correctly
        self._inn.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=self._inn, anchor="nw")
        cv.configure(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y); cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Mouse-wheel scrolling — delta/120 gives scroll units on Windows
        cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-1*(e.delta/120)), "units"))
        o = self._inn   # shorthand so all widget packs below use a single letter

        # ---- toolbar — one button per action ----
        # Each tuple is (button label, callback method, button width in chars).
        # Buttons are packed left-to-right with a 3-pixel gap between them.
        tb = tk.Frame(o); tb.pack(fill=tk.X, padx=6, pady=3)
        for txt, cmd, w in [
            ("Open NHDF…",     self._open_file,         11),  # Load an NHDF file
            ("Save NHDF…",     self._save_file,         11),  # Re-save as NHDF
            ("Export CSV…",    self._export_csv,        11),  # Raw data to CSV
            ("Export Sum…",    self._export_sum,        11),  # Spatially summed spectrum
            ("Export DAT…",    self._export_dat,        11),  # Float32 binary + sidecar
            ("Export DM3…",    self._export_dm3,        11),  # Full cube as GMS DM3
            ("Export BkgSub…", self._export_bkgsub_dat, 13),  # Area-norm bkg-sub DAT
            ("BkgSub DM3…",    self._export_bkgsub_dm3, 12),  # Area-norm bkg-sub DM3
            ("Temp Map…",      self._export_temp_dm3,   11)]: # Detailed balance T map
            tk.Button(tb, text=txt, command=cmd, width=w, font=_FNT).pack(side=tk.LEFT, padx=(0, 3))

        # Label showing the path of the currently loaded file (grey until opened)
        self.file_lbl = tk.Label(o, text="No file loaded", anchor="w", fg="gray", font=_FNT)
        self.file_lbl.pack(fill=tk.X, padx=8)

        # ---- comment entry ----
        # The user can type a short description that gets appended to all
        # exported filenames (after sanitisation in _comment_suffix()).
        cf = tk.Frame(o); cf.pack(fill=tk.X, padx=6, pady=(2, 0))
        tk.Label(cf, text="Comment:", font=_FNT).pack(side=tk.LEFT)
        self.comment_entry = tk.Entry(cf, font=_FNT)
        self.comment_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # ---- fit model selector ----
        # var_fit controls which lineshape is used by _fit_peak().
        # Changing the selection immediately re-fits and redraws via _on_fit_change().
        fr = tk.Frame(o); fr.pack(fill=tk.X, padx=6, pady=(2, 0))
        tk.Label(fr, text="Fit:", font=_FNT_MB).pack(side=tk.LEFT)
        self.var_fit = tk.StringVar(value="gausslor")   # default: GaussLor (best for ZLP)
        for lbl, val in [("Gaussian", "gauss"), ("Lorentzian", "lorentz"), ("GaussLor", "gausslor")]:
            tk.Radiobutton(fr, text=lbl, variable=self.var_fit, value=val,
                           command=self._on_fit_change, font=_FNT).pack(side=tk.LEFT, padx=(4, 0))

        # ---- info readout bar ----
        # Six labelled boxes showing fit results and calibration bin sizes.
        # These update via _upd_info() whenever the cursor moves or the fit model changes.
        #   lb_pk : peak maximum of the fitted model at the cursor pixel
        #   lb_ar : integrated area under the fitted curve
        #   lb_fw : Full-Width at Half-Maximum of the fitted peak
        #   lb_bs : energy calibration step (scale) — one energy bin width
        #   lb_bx : spatial calibration for dim-0 (rows) — one pixel width in nm
        #   lb_by : spatial calibration for dim-1 (cols) — one pixel width in nm
        ir = tk.Frame(o); ir.pack(fill=tk.X, padx=6, pady=(2, 1))
        ir.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)  # equal-width columns
        def mkb(p, l, c):
            """Make a single labelled info box and return its value Label."""
            f = tk.LabelFrame(p, text=l, padx=3, pady=1, font=_FNT)
            f.grid(row=0, column=c, padx=2, sticky="nsew")
            v = tk.Label(f, text="—", font=_FNT_MB, anchor="center"); v.pack(fill=tk.X); return v
        self.lb_pk = mkb(ir, "Peak Max", 0)
        self.lb_ar = mkb(ir, "Area", 1)
        self.lb_fw = mkb(ir, "FWHM", 2)
        self.lb_bs = mkb(ir, "Bin(spectrum)", 3)
        self.lb_bx = mkb(ir, "Bin(x/dim0)", 4)
        self.lb_by = mkb(ir, "Bin(y/dim1)", 5)

        # ---- ROW 1: two summed-spectrum plots side by side ----
        # Left panel  ("Full-range sum"):     log Y scale, shows the full dynamic
        #   range from ZLP down to noise floor.
        # Right panel ("1/30-max capped sum"): linear Y scale capped at 1/30 of the
        #   maximum, revealing weak phonon features that would be invisible on log scale.
        # Each panel is built by _mksp() which returns (fig, ax, canvas, sx, sy, yscale_var).
        r1 = tk.Frame(o); r1.pack(fill=tk.X, padx=6, pady=(2, 0))
        r1.columnconfigure((0, 1), weight=1)   # both columns share the available width
        self.ff, self.af, self.cf, self.sxf, self.syf, self.vsf = \
            self._mksp(r1, 0, "Full-range sum", "log", (4.8, 1.4))
        self.fz, self.az, self.cz, self.sxz, self.syz, self.vsz = \
            self._mksp(r1, 1, "1/30-max capped sum", "linear", (4.8, 1.4))

        # ---- ROW 2: spatial image panel ----
        # This is the main image viewer.  It shows a single energy slice of the
        # 3-D spectrum image (selected by the energy slider), rendered as a
        # false-colour map.  The user clicks to move the cursor pixel, and drags
        # to draw/move the signal ROI or background ROI.
        imf = tk.LabelFrame(o, text="Spatial image — click=cursor, drag ROI when active",
                            padx=3, pady=2, font=_FNT)
        imf.pack(fill=tk.X, padx=6, pady=(3, 0))

        # ---- energy slider row ----
        # sl_e: integer slider from 0 to ne-1 — selects which energy channel
        #       (or bin) to display.  Dragging it calls _on_esl().
        # lb_e: shows the calibrated energy value of the current slice in eV.
        # lb_c: shows the (row, col) index and calibrated position of the cursor.
        # peak_canvas: a thin Tk Canvas overlaid on the slider row; red
        #   downward-pointing triangles are drawn at the channel positions of
        #   automatically detected peaks so the user can quickly navigate to them.
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
        # Peak arrow canvas — placed() on top of sl_frame at full width, 10 px tall.
        # Triangles drawn here are purely cosmetic; they do NOT intercept mouse events.
        self.peak_canvas = tk.Canvas(sl_frame, height=10, highlightthickness=0)
        self.peak_canvas.place(relx=0, rely=0, relwidth=1.0)
        self.peak_canvas.config(bg=imf.cget("bg"))
        self._peak_indices = []   # filled by _detect_peaks() after each file load

        # ---- signal ROI control row ----
        # The signal ROI is a polygon defined by clicking and dragging on the image.
        # It is used to extract a summed spectrum from a user-selected area.
        # Buttons:
        #   Draw ROI   — enter draw mode; next click-drag on the image defines the ROI
        #   Move ROI   — enter move/resize mode; drag the whole polygon or a corner
        #   Clear ROI  — delete the current ROI
        #   Export ROI… — save the summed ROI spectrum to CSV
        # lb_roi shows the current ROI extent as x[min-max] y[min-max].
        # Colourmap radio buttons (right side) select the Matplotlib cmap for _draw_img.
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
        # Colourmap selector — packed right-to-left so the first entry in _CMAPS
        # ("inferno") is closest to the right edge (visually leftmost when packed RIGHT)
        self.var_cmap = tk.StringVar(value="inferno")
        for cm in _CMAPS:
            tk.Radiobutton(rc, text=cm, variable=self.var_cmap, value=cm,
                           command=lambda: self._draw_img(self.sl_e.get()),
                           font=_FNT).pack(side=tk.RIGHT)

        # ---- background ROI control row ----
        # The background (Bkg) ROI is a second polygon used as the reference
        # background for area-normalised subtraction and the temperature map.
        # "Place Bkg" clones the signal ROI and offsets it by half its width —
        # a convenient starting position that the user can then fine-tune.
        # Contour overlay and energy binning controls live on the right side.
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
        # Contour overlay: when "On", _draw_img() adds equal-intensity contour
        # lines on top of the false-colour image for better edge visibility
        tk.Label(rc2, text="  Contour:", font=_FNT).pack(side=tk.RIGHT, padx=(0, 2))
        self.var_contour = tk.StringVar(value="off")
        tk.Radiobutton(rc2, text="On", variable=self.var_contour, value="on",
                       command=lambda: self._draw_img(self.sl_e.get()),
                       font=_FNT).pack(side=tk.RIGHT)
        tk.Radiobutton(rc2, text="Off", variable=self.var_contour, value="off",
                       command=lambda: self._draw_img(self.sl_e.get()),
                       font=_FNT).pack(side=tk.RIGHT)
        # BinE: bin adjacent energy channels before display.
        # Choices 1/2/4/8 average that many channels centred on sl_e.get(),
        # which reduces noise in weak phonon maps at the cost of energy resolution.
        tk.Label(rc2, text="  BinE:", font=_FNT).pack(side=tk.RIGHT, padx=(8, 2))
        self.var_ebin = tk.StringVar(value="1")   # default: no binning
        for _b in ["8", "4", "2", "1"]:
            tk.Radiobutton(rc2, text=_b, variable=self.var_ebin, value=_b,
                           command=lambda: self._draw_img(self.sl_e.get()),
                           font=_FNT).pack(side=tk.RIGHT)

        # ---- spatial image Matplotlib canvas ----
        # fi/ai: Matplotlib Figure/Axes holding the imshow plot.
        # ci: FigureCanvasTkAgg embedding fi into the imf LabelFrame.
        # Mouse events on ci are routed to the ROI drawing / cursor methods.
        # Arrow-key events are captured when _img_widget has focus.
        self.fi = Figure(figsize=(10, 2.0), dpi=100)
        self.ai = self.fi.add_subplot(111)
        self.ci = FigureCanvasTkAgg(self.fi, master=imf)
        self.ci.get_tk_widget().pack(fill=tk.X)
        self.ci.mpl_connect("button_press_event",   self._on_img_press)
        self.ci.mpl_connect("button_release_event", self._on_img_release)
        self.ci.mpl_connect("motion_notify_event",  self._on_img_motion)
        self._img_widget = self.ci.get_tk_widget()
        self._img_widget.configure(takefocus=True)   # must have focus for key events
        # Arrow keys nudge the active ROI (or Bkg) by one pixel.
        # Shift-arrow moves the cursor instead (handled in _on_arrow_key).
        self._img_widget.bind("<Left>",        self._on_arrow_key)
        self._img_widget.bind("<Right>",       self._on_arrow_key)
        self._img_widget.bind("<Up>",          self._on_arrow_key)
        self._img_widget.bind("<Down>",        self._on_arrow_key)
        self._img_widget.bind("<Shift-Left>",  self._on_arrow_key)
        self._img_widget.bind("<Shift-Right>", self._on_arrow_key)
        self._img_widget.bind("<Shift-Up>",    self._on_arrow_key)
        self._img_widget.bind("<Shift-Down>",  self._on_arrow_key)

        # ---- ROW 3: point spectrum panel ----
        # This panel always shows three spectra overlaid on the same axes:
        #   orange : cursor pixel spectrum (d[_cix, _ciy, :])
        #   green  : signal ROI spectrum (summed over all ROI pixels)
        #   cyan   : background ROI spectrum
        # Dashed curves of the same colour show the fitted peak model.
        # A red curve is drawn if a subtraction mode is active (var_sub).
        # Clicking on this plot places / updates a vertical red energy marker.
        pf = tk.LabelFrame(o, text="Point spectrum (cursor) + ROI (green) + Bkg (cyan)", padx=3, pady=2, font=_FNT)
        pf.pack(fill=tk.X, padx=6, pady=(3, 0))
        self.fp = Figure(figsize=(10, 1.6), dpi=100)
        self.ap = self.fp.add_subplot(111)
        self.cp = FigureCanvasTkAgg(self.fp, master=pf)
        self.cp.get_tk_widget().pack(fill=tk.X)
        # Clicking on the spectrum plot sets _pt_vline_energy and redraws
        self.cp.mpl_connect("button_press_event", self._on_pt_click)

        # Control row for the point spectrum panel:
        #   sxp : X-range zoom — 100% = full energy range, lower = zoomed in around E=0
        #   syp : Y-scale zoom — 100% = full peak height; values >100 stretch the Y axis
        #         (useful to see weak gain-side phonon peaks)
        #   vsp : Y-axis scale — Log or Linear
        #   var_norm : normalisation mode applied before plotting (off / max→1 / area→1)
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
        tk.Radiobutton(cp, text="Log",    variable=self.vsp, value="log",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp, text="Lin", variable=self.vsp, value="linear",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Label(cp, text="  Norm:", font=_FNT).pack(side=tk.LEFT, padx=(6, 2))
        self.var_norm = tk.StringVar(value="off")
        tk.Radiobutton(cp, text="Off",    variable=self.var_norm, value="off",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp, text="Max→1",  variable=self.var_norm, value="max",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp, text="Area→1", variable=self.var_norm, value="area",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)

        # ---- subtract mode row ----
        # var_sub selects which pair of spectra to subtract and display as
        # a red curve.  The result can be exported with "Export Sub…".
        # Modes (the − symbol is U+2212 MINUS SIGN for clean display):
        #   cur_roi  : cursor pixel  −  signal ROI average
        #   roi_cur  : signal ROI average  −  cursor pixel
        #   cur_roi2 : cursor pixel  −  background ROI
        #   roi_roi2 : signal ROI  −  background ROI  (most common: signal − background)
        #   roi2_roi : background ROI  −  signal ROI
        #   roi2_cur : background ROI  −  cursor pixel
        cp2 = tk.Frame(pf); cp2.pack(fill=tk.X, pady=(1, 0))
        tk.Label(cp2, text="Subtract:", font=_FNT).pack(side=tk.LEFT, padx=(0, 2))
        self.var_sub = tk.StringVar(value="off")
        tk.Radiobutton(cp2, text="Off",           variable=self.var_sub, value="off",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="Cursor \u2212 ROI",    variable=self.var_sub, value="cur_roi",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="ROI \u2212 Cursor",    variable=self.var_sub, value="roi_cur",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="Cursor \u2212 Bkg",    variable=self.var_sub, value="cur_roi2",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="ROI \u2212 Bkg",       variable=self.var_sub, value="roi_roi2",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="Bkg \u2212 ROI",       variable=self.var_sub, value="roi2_roi",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Radiobutton(cp2, text="Bkg \u2212 Cursor",    variable=self.var_sub, value="roi2_cur",
                       command=self._draw_pt, font=_FNT).pack(side=tk.LEFT)
        tk.Button(cp2, text="Export Sub\u2026", command=self._export_sub,
                  font=_FNT, width=10).pack(side=tk.RIGHT, padx=(6, 0))

        # ---- metadata text box ----
        # Shows the formatted NHDF metadata (from format_info()) after loading.
        # Read-only (state=DISABLED) except when being written by _open_file().
        self.text = scrolledtext.ScrolledText(o, wrap=tk.WORD, state=tk.DISABLED,
                                              font=_FNT_M, height=5)
        self.text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(3, 3))

        # ---- status bar ----
        # Single-line label at the bottom that shows the last completed action.
        self.status = tk.Label(o, text="Ready", anchor="w", relief=tk.SUNKEN, bd=1, font=_FNT)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _mksp(self, par, col, title, dflt, fsz):
        """Create one of the two summed-spectrum plot panels (helper factory).

        Builds a LabelFrame containing:
          - a Matplotlib Figure/Axes embedded as a FigureCanvasTkAgg
          - an X% slider (zoom the horizontal energy range: 100% = full width)
          - a Y% slider (zoom the vertical scale: 100% = full height)
          - Log / Lin radio buttons to switch the Y-axis scale

        Parameters
        ----------
        par   : tk widget — parent frame (the two-column row)
        col   : int       — grid column (0 = left, 1 = right)
        title : str       — LabelFrame header text
        dflt  : str       — default Y scale ("log" or "linear")
        fsz   : tuple     — Matplotlib figure size in inches (width, height)

        Returns
        -------
        (fig, ax, canvas, sx_slider, sy_slider, yscale_var) — all kept as
        instance attributes so _d1() can read and redraw them.
        """
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
        """Return the spectrum summed over all spatial pixels.

        Sums d over every axis except the last (energy) axis, so the result
        is a 1-D array of length ne regardless of whether the data is 2-D or
        3-D.  This gives the highest signal-to-noise representation of the
        sample phonon spectrum and is what the two top panels display.

        Returns (xc, units, sp) or None if no data is loaded.
        """
        if not self.current_data: return None
        d = _arr(self.current_data)
        # Sum over all axes except the last (energy) axis
        sp = d.sum(axis=tuple(range(d.ndim - 1)))
        x, u = _spec_cal(self.current_data)
        return x, u, sp

    def _fit_peak(self, x, y):
        """Fit the currently selected lineshape model to a spectrum.

        Uses scipy.optimize.curve_fit (Levenberg-Marquardt) to optimise the
        three parameters (A, x0, width) of the chosen model.  Initial guesses
        are derived directly from the data:
          - A0  = maximum of y_f (amplitude guess)
          - x0_0 = x position of that maximum (centre guess)
          - sig0 = half the threshold-crossing width / 2.355
                   (sigma/gamma guess, clamped to at least one channel width)

        Parameters
        ----------
        x : 1-D array — calibrated energy axis (eV)
        y : 1-D array — spectrum counts at the cursor or ROI position

        Returns
        -------
        (peak_max, area, fwhm, x0, fitted_y) on success, or None if:
          - the spectrum is all non-positive (cannot fit), or
          - curve_fit fails to converge within 5000 function evaluations.
        """
        mode = self.var_fit.get()
        y_f = y.astype(float)
        if numpy.max(y_f) <= 0:
            return None   # can't fit an empty or all-negative spectrum

        # Build initial parameter guesses from the raw data
        idx_max = numpy.argmax(y_f)
        A0   = float(y_f[idx_max])   # peak amplitude guess
        x0_0 = float(x[idx_max])     # peak centre guess

        # Estimate width: half the full width at half max / 2.355 ≈ sigma
        hm = A0 / 2
        above = numpy.where(y_f >= hm)[0]
        if len(above) >= 2:
            sig0 = float(x[above[-1]] - x[above[0]]) / 2.355
        else:
            sig0 = float(x[-1] - x[0]) / 10   # fall back to 1/10 of range

        # Clamp sig0 to at least one channel width (avoids degenerate guesses)
        sig0 = max(sig0, abs(float(x[1] - x[0])) if len(x) > 1 else 1.0)

        try:
            if mode == "gauss":
                popt, _ = curve_fit(_gauss_fn, x, y_f, p0=[A0, x0_0, sig0], maxfev=5000)
                fwhm   = _fwhm_gauss(popt[2])
                fitted = _gauss_fn(x, *popt)
            elif mode == "lorentz":
                popt, _ = curve_fit(_lorentz_fn, x, y_f, p0=[A0, x0_0, sig0], maxfev=5000)
                fwhm   = _fwhm_lorentz(popt[2])
                fitted = _lorentz_fn(x, *popt)
            else:   # "gausslor"
                popt, _ = curve_fit(_gausslor_fn, x, y_f, p0=[A0, x0_0, sig0], maxfev=5000)
                fwhm   = _fwhm_gausslor(popt[2])
                fitted = _gausslor_fn(x, *popt)

            peak_max = float(numpy.max(fitted))
            # Integrate the fitted curve using the trapezoidal rule
            area     = float(numpy.trapz(fitted, x))
            return peak_max, area, fwhm, float(popt[1]), fitted
        except Exception:
            # curve_fit can raise RuntimeError (max iterations) or ValueError;
            # return None so callers fall back to the threshold-crossing FWHM.
            return None

    def _on_fit_change(self):
        """Called when the user switches the fit model radio button.

        Re-runs the peak fit on the current spectrum and refreshes both
        the info readout bar and the point-spectrum plot.
        """
        self._upd_info()
        self._draw_pt()

    def _upd_info(self):
        """Refresh the six info readout labels in the top bar.

        Runs _fit_peak() on the current summed spectrum (_sp) and populates:
          lb_pk : fitted peak maximum
          lb_ar : fitted integrated area
          lb_fw : FWHM (fitted if curve_fit succeeded, threshold otherwise)
          lb_bs : energy bin width (scale of the energy calibration)
          lb_bx : spatial pixel size along dim-0 (rows)
          lb_by : spatial pixel size along dim-1 (cols)
        """
        if self._xc is None: return
        result = self._fit_peak(self._xc, self._sp)
        if result:
            peak_max, area, fw, x0, fitted = result
            self.lb_pk.config(text=f"{peak_max:.4g}")
            self.lb_ar.config(text=f"{area:.4g}")
            self.lb_fw.config(text=f"{fw:.4g} {self._su}")
        else:
            # Fit failed — show threshold FWHM; leave peak/area as "—"
            fw = _fwhm(self._xc, self._sp.astype(float))
            self.lb_pk.config(text="—")
            self.lb_ar.config(text="—")
            self.lb_fw.config(text=f"{fw:.4g} {self._su}")
        # Calibration bin sizes from the dimensional_calibrations list
        cs = self.current_data.dimensional_calibrations if self.current_data else []
        for lb, i in [(self.lb_bs, 2), (self.lb_bx, 0), (self.lb_by, 1)]:
            if cs and i < len(cs):
                c = cs[i]; lb.config(text=f"{c.scale or 1:.4g} {c.units or ''}")
            else: lb.config(text="—")

    # ---- summed-spectrum plot rendering ----

    def _d1(self, ax, fig, cv, sx, sy, vs, cap=None):
        """Render one summed-spectrum plot panel (_sp on _xc energy axis).

        Parameters
        ----------
        ax, fig, cv : Matplotlib Axes, Figure, FigureCanvasTkAgg for this panel
        sx, sy      : X% and Y% Tk Scale sliders for zoom control
        vs          : Tk StringVar holding "log" or "linear"
        cap         : if not None, cap the displayed Y maximum at ym/cap
                      (used by the right panel to cap at 1/30 of the peak)
        """
        if self._xc is None: return
        x = self._xc; sp = self._sp.astype(float); u = self._su
        ym = float(numpy.max(sp))     # true maximum of the summed spectrum
        r  = float(x[-1] - x[0])     # total energy range (eV)
        fx = sx.get() / 100.          # X zoom fraction: 1.0 = full range
        fy = sy.get() / 100.          # Y zoom fraction: 1.0 = full height

        ax.clear()
        ax.plot(x, sp, lw=0.6)

        # Symmetric X limits centred on E=0
        ax.set_xlim(-(r * fx) / 2, (r * fx) / 2)

        # Apply optional Y cap (e.g., cap=30 means show up to 1/30 of max)
        top = (ym / cap if cap else ym) * fy

        if vs.get() == "log":
            ax.set_yscale("log")
            # Lower log limit: smallest positive count (at least 1 to avoid log(0))
            mn = max(1., float(sp[sp > 0].min())) if numpy.any(sp > 0) else 1.
            ax.set_ylim(mn, top * 1.05)
        else:
            ax.set_yscale("linear"); ax.set_ylim(0, top)

        ax.set_xlabel(f"Energy ({u})" if u else "Ch", fontsize=_FP)
        ax.set_ylabel("Cts", fontsize=_FP)
        ax.tick_params(labelsize=_FP)
        fig.tight_layout(); cv.draw_idle()

    def _draw_plots(self):
        """Redraw both summed-spectrum panels (full-range log and capped linear)."""
        self._d1(self.af, self.ff, self.cf, self.sxf, self.syf, self.vsf)           # left
        self._d1(self.az, self.fz, self.cz, self.sxz, self.syz, self.vsz, cap=30)  # right

    def _upd_plot(self):
        """Recompute the summed spectrum and redraw both panels.

        Called after a new file is loaded.  Resets all four X%/Y% sliders
        to 100% before drawing so the full spectrum is visible.
        """
        r = self._sum_sp()
        if not r: return
        self._xc, self._su, self._sp = r
        for s in [self.sxf, self.syf, self.sxz, self.syz]: s.set(100)
        self._draw_plots()

    def _on_sl(self):
        """Callback for any X%/Y% slider in the summed-spectrum panels.

        Re-renders both panels at the new zoom level without recomputing
        the summed spectrum.
        """
        self._draw_plots()

    # ---- spatial image ----

    def _setup_esl(self):
        """Initialise the energy slider and spatial image after loading a new file.

        Sets the slider range to [0, ne-1], tries to start at the channel
        closest to the MgO Fuchs-Kliewer phonon peak (~0.067 eV), resets
        both ROIs and the cursor to the image centre, runs peak detection,
        then draws the image and point spectrum for the first time.
        """
        if not self.current_data: return
        d = _arr(self.current_data)
        if d.ndim < 3: self.sl_e.config(from_=0, to=0); return

        # Set slider range to cover all energy channels
        self.sl_e.config(from_=0, to=d.shape[2] - 1)

        # Default display channel: closest to the MgO Fuchs-Kliewer phonon at
        # ~0.067 eV (K point of MgO, the dominant vibrational mode visible in
        # monochromated STEM-EELS).  Falls back gracefully if the calibration
        # puts this outside the range.
        _xfk, _ = _spec_cal(self.current_data)
        zi = int(numpy.argmin(numpy.abs(_xfk - 0.06746)))
        self.sl_e.set(zi)

        # Reset cursor to the centre of the spatial image
        self._cix = d.shape[0] // 2; self._ciy = d.shape[1] // 2

        # Clear any leftover ROI and Bkg state from the previously loaded file
        self._roi = None; self._roi_drawing = False
        self._roi_moving = False; self._roi_drag_start = None; self._roi_anchor = None
        self._roi_drag_corner = None
        self._roi2 = None; self._roi2_moving = False
        self._roi2_drag_start = None; self._roi2_anchor = None
        self._roi2_drag_corner = None

        # Reset ROI / Bkg status labels and button reliefs
        self.lb_roi.config(text="ROI: none")
        self.lb_roi2.config(text="Bkg: none")
        self.btn_mvroi.config(relief=tk.RAISED, text="Move ROI")
        self.btn_mvroi2.config(relief=tk.RAISED, text="Move Bkg")

        # Find peaks in the summed spectrum and draw red arrows on the slider
        self._detect_peaks()

        # Draw the initial image and point spectrum
        self._draw_img(zi); self._draw_pt()

    def _detect_peaks(self):
        """Detect spectral peaks in the summed spectrum and mark them on the slider.

        Runs scipy find_peaks twice — once on the linearly smoothed spectrum
        (catches strong features like the ZLP) and once on the log-smoothed
        spectrum (catches weaker phonon peaks that are invisible on linear scale).
        The two peak lists are merged and deduplicated.

        The detected channel indices are stored in self._peak_indices and then
        rendered as small red triangles on peak_canvas by _draw_peak_arrows().
        The after(50) delay gives the canvas time to lay out before we query
        its widget geometry (needed for accurate triangle placement).
        """
        self.peak_canvas.delete("all")
        self._peak_indices = []
        if not self.current_data: return
        d = _arr(self.current_data)
        if d.ndim < 3: return

        # Summed spectrum over all spatial pixels for best SNR
        sp = d.sum(axis=(0, 1)).astype(float)
        nch = len(sp)
        if nch < 5: return   # too few channels to find meaningful peaks

        from scipy.ndimage import uniform_filter1d

        # Smoothing kernel and minimum peak separation scaled to spectrum length
        kern = max(3, nch // 100)   # smoothing window: ~1% of total channels
        dist = max(3, nch // 50)    # minimum inter-peak distance: ~2% of channels

        # ---- Linear-scale peak detection ----
        # Catches dominant peaks (ZLP, strong phonon overtones)
        smooth     = uniform_filter1d(sp, size=kern)
        prom_lin   = max(smooth.max() * 0.02, 1.)   # 2% of max, at least 1 count
        peaks_lin, _ = find_peaks(smooth, prominence=prom_lin, distance=dist)

        # ---- Log-scale peak detection ----
        # Catches weaker features that are compressed on a linear scale
        sp_pos     = numpy.clip(sp, 1e-30, None)    # avoid log(0)
        log_sp     = numpy.log10(sp_pos)
        log_smooth = uniform_filter1d(log_sp, size=kern)
        prom_log   = max((log_smooth.max() - log_smooth.min()) * 0.03, 0.05)
        peaks_log, _ = find_peaks(log_smooth, prominence=prom_log, distance=dist)

        # Merge both lists and remove duplicates (numpy.unique sorts and deduplicates)
        all_peaks = numpy.unique(numpy.concatenate([peaks_lin, peaks_log]))
        self._peak_indices = all_peaks

        # Draw the arrows after a short delay so widget geometry is finalised
        self.peak_canvas.after(50, self._draw_peak_arrows)

    def _draw_peak_arrows(self):
        """Draw small red downward-pointing triangles above detected peaks on the slider.

        This is called ~50 ms after _detect_peaks() so that all widget sizes
        have been resolved by the Tk geometry manager.

        The triangles are drawn on peak_canvas which is placed() (absolute
        positioning) directly over the slider row.  We query the slider widget's
        x position and width to map each channel index to a pixel x-coordinate
        within the canvas.

        The 14-pixel `pad` accounts for the rounded end-caps of the Tk Scale
        widget's internal trough — the thumb cannot move all the way to the
        widget edges, so the active range is slightly inset.
        """
        self.peak_canvas.delete("all")
        if not self._peak_indices.size or not self.current_data: return
        d = _arr(self.current_data)
        if d.ndim < 3: return
        nch = d.shape[2]   # total number of energy channels

        # Force pending geometry updates so winfo_* returns correct sizes
        self.peak_canvas.update_idletasks()
        w = self.peak_canvas.winfo_width()

        # Get the slider's pixel position and width within its parent frame
        try:
            sl_x = self.sl_e.winfo_x()
            sl_w = self.sl_e.winfo_width()
        except Exception:
            sl_x = 60; sl_w = w - 180   # fallback estimates

        pad = 14   # Tk Scale trough padding (pixels inset from widget edge)
        x0 = sl_x + pad          # left pixel of usable trough
        x1 = sl_x + sl_w - pad   # right pixel of usable trough
        span = x1 - x0
        if span <= 0: return   # degenerate geometry — skip

        h = self.peak_canvas.winfo_height()

        for idx in self._peak_indices:
            # Map channel index linearly to pixel x within the trough
            frac = idx / max(nch - 1, 1)
            cx = x0 + frac * span
            # Draw a small filled downward-pointing triangle (3 px base, h-1 px tall)
            self.peak_canvas.create_polygon(
                cx - 3, 0, cx + 3, 0, cx, h - 1,
                fill="red", outline="red")

    def _on_esl(self, v):
        """Energy slider callback — redraws the spatial image at the new channel."""
        self._draw_img(int(v))

    def _draw_img(self, ei):
        """Render the spatial image at energy channel index `ei`.

        Extracts a 2-D intensity slice from the 3-D datacube, applies optional
        energy binning (averaging `ebin` adjacent channels centred on `ei`),
        renders it as a false-colour imshow, overlays the cursor crosshair and
        any active ROI/Bkg polygons, and optionally draws contour lines.

        Parameters
        ----------
        ei : int — energy channel index to display (0-based, from sl_e)
        """
        if not self.current_data: return
        d = _arr(self.current_data)
        if d.ndim < 3: return
        nz = d.shape[2]   # total number of energy channels

        # ---- energy binning ----
        # When ebin > 1, average `ebin` consecutive channels centred on `ei`.
        # This reduces photon-counting noise at the cost of energy resolution,
        # which is useful for imaging weak phonon modes.
        ebin = int(self.var_ebin.get())
        if ebin > 1:
            lo = max(0, ei - ebin // 2)    # first channel to include
            hi = min(nz, lo + ebin)        # one past the last channel
            lo = max(0, hi - ebin)         # re-clamp lo if hi hit the upper boundary
            slc = d[:, :, lo:hi].mean(axis=2).astype(float)
        else:
            slc = d[:, :, ei].astype(float)   # single-channel slice, no binning

        # ---- spatial and energy calibrations ----
        # _gc returns (offset, scale, units) for each axis.
        # Axis 0 = rows (displayed as Y on the imshow), scale sx in nm/pixel.
        # Axis 1 = cols (displayed as X on the imshow), scale sy in nm/pixel.
        # Axis 2 = energy; we just need the calibrated energy value `ev` of slice `ei`.
        _, sx, ux = _gc(self.current_data, 0); ox = 0.
        _, sy, uy = _gc(self.current_data, 1); oy = 0.
        oe, se, ue = _gc(self.current_data, 2)
        nx, ny = d.shape[0], d.shape[1]

        # imshow extent: [xmin, xmax, ymin, ymax] in calibrated units.
        # We map col→x (horizontal) and row→y (vertical), with lower origin.
        ext = [0, sy * ny, 0, sx * nx]
        ev  = oe + se * ei   # calibrated energy of the displayed slice

        # Update the energy label to show the calibrated value
        self.lb_e.config(text=f"{ev:.4g} {ue}")

        # ---- draw the image ----
        # Clear the figure completely (removes old colorbar too) then re-add axes
        self.fi.clear(); self.ai = self.fi.add_subplot(111)
        self._cb = None   # old colorbar handle is now invalid
        cmap = self.var_cmap.get()
        im = self.ai.imshow(slc, aspect="auto", origin="lower", extent=ext, cmap=cmap)
        self.ai.set_xlabel(f"y ({uy})" if uy else "y", fontsize=_FP)
        self.ai.set_ylabel(f"x ({ux})" if ux else "x", fontsize=_FP)
        bin_tag = f" ×{ebin}" if ebin > 1 else ""
        self.ai.set_title(f"E={ev:.4g} {ue} [slice {ei}{bin_tag}]", fontsize=_FP)
        self.ai.tick_params(labelsize=_FP)
        self._cb = self.fi.colorbar(im, ax=self.ai, fraction=0.025, pad=0.015)
        self._cb.ax.tick_params(labelsize=_FP)

        # ---- cursor crosshair ----
        # The cursor pixel (_cix, _ciy) is shown as a cyan "+" marker.
        # +0.5 converts from pixel corner to pixel centre in calibrated units.
        cy_cal = oy + sy * (self._ciy + 0.5)   # column → calibrated x
        cx_cal = ox + sx * (self._cix + 0.5)   # row    → calibrated y
        self.ai.plot(cy_cal, cx_cal, "+", color="cyan", ms=10, mew=2)
        self.lb_c.config(text=f"({self._cix},{self._ciy}) {cx_cal:.3g}{ux},{cy_cal:.3g}{uy}")

        # ---- signal ROI overlay (green) ----
        if self._roi is not None:
            # Convert corner pixel indices to calibrated coordinates for imshow
            verts = [(sy * (iy + 0.5), sx * (ix + 0.5)) for ix, iy in self._roi]
            poly  = MplPolygon(verts, closed=True, linewidth=1.5,
                               edgecolor="lime", facecolor="lime", alpha=0.2)
            self.ai.add_patch(poly)
            # Also draw small squares at each corner for drag handles
            for px, py in verts:
                self.ai.plot(px, py, "s", color="lime", ms=5, mew=1.5)

        # ---- background ROI overlay (cyan) ----
        if self._roi2 is not None:
            verts2 = [(sy * (iy + 0.5), sx * (ix + 0.5)) for ix, iy in self._roi2]
            poly2  = MplPolygon(verts2, closed=True, linewidth=1.5,
                                edgecolor="cyan", facecolor="cyan", alpha=0.2)
            self.ai.add_patch(poly2)
            for px, py in verts2:
                self.ai.plot(px, py, "s", color="cyan", ms=5, mew=1.5)

        # ---- optional contour overlay ----
        # Draws 8 white iso-intensity contours between 10% and 95% of the
        # slice maximum.  Useful for identifying domain boundaries or gradients.
        if self.var_contour.get() == "on":
            vmin, vmax = float(slc.min()), float(slc.max())
            if vmax > vmin:
                levels = numpy.linspace(vmin + (vmax - vmin) * 0.1, vmax * 0.95, 8)
                self.ai.contour(slc, levels=levels, origin="lower", extent=ext,
                                colors="white", linewidths=0.6, alpha=0.7)

        self.fi.tight_layout(); self.ci.draw_idle()

    # ---- ROI interaction helpers ----

    def _cal_to_pix(self, xdata, ydata):
        """Convert Matplotlib event coordinates (calibrated units) to pixel indices.

        Matplotlib imshow events report xdata/ydata in the same units as the
        `extent` parameter — i.e., in nanometres (or whatever the calibration
        units are).  We invert the calibration to get integer pixel indices.

        Parameters
        ----------
        xdata : float — horizontal imshow coordinate (col direction, in cal units)
        ydata : float — vertical imshow coordinate   (row direction, in cal units)

        Returns
        -------
        (ix, iy) : (int, int) — row index and column index, clamped to [0, nx/ny-1]
        """
        _, sx, _ = _gc(self.current_data, 0)   # row scale (nm / pixel)
        _, sy, _ = _gc(self.current_data, 1)   # col scale (nm / pixel)
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        # xdata is in the column direction (y in Matplotlib → col in array)
        iy = int(numpy.floor(xdata / sy)) if sy else 0
        # ydata is in the row direction (x in Matplotlib → row in array)
        ix = int(numpy.floor(ydata / sx)) if sx else 0
        return max(0, min(nx - 1, ix)), max(0, min(ny - 1, iy))

    def _poly_mask(self, corners):
        """Rasterise a polygon (given as corner pixel indices) into a boolean mask.

        Uses matplotlib.path.Path.contains_points() for an exact point-in-polygon
        test, applied only within the bounding box of the polygon for efficiency.

        Parameters
        ----------
        corners : (N, 2) int ndarray — [row, col] pixel coordinates of the polygon
                  corners (in order; the polygon is closed automatically).

        Returns
        -------
        mask : (nx, ny) bool ndarray — True for pixels inside the polygon.
        """
        d = _arr(self.current_data)
        nx, ny = d.shape[0], d.shape[1]

        # Close the polygon by appending the first corner at the end
        fc   = numpy.vstack([corners, corners[0:1]]).astype(float)
        path = MplPath(fc)

        # Bounding box of the polygon (clamped to array bounds)
        ix_min = max(0, int(corners[:, 0].min()))
        ix_max = min(nx - 1, int(corners[:, 0].max()))
        iy_min = max(0, int(corners[:, 1].min()))
        iy_max = min(ny - 1, int(corners[:, 1].max()))

        mask = numpy.zeros((nx, ny), dtype=bool)
        if ix_min > ix_max or iy_min > iy_max:
            return mask   # degenerate polygon — return empty mask

        # Build a grid of (row, col) points within the bounding box and test each
        sub_ix, sub_iy = numpy.mgrid[ix_min:ix_max + 1, iy_min:iy_max + 1]
        pts      = numpy.column_stack((sub_ix.ravel(), sub_iy.ravel())).astype(float)
        sub_mask = path.contains_points(pts).reshape(sub_ix.shape)
        mask[ix_min:ix_max + 1, iy_min:iy_max + 1] = sub_mask
        return mask

    def _find_corner(self, corners, ix, iy, threshold=3):
        """Return the index of the corner pixel nearest to (ix, iy), or -1.

        Used during Move ROI to detect whether the user clicked near a corner
        (resize handle) or elsewhere in the polygon (translate the whole shape).

        Parameters
        ----------
        corners   : (N, 2) int ndarray — corner pixels
        ix, iy    : int — clicked pixel coordinates
        threshold : int — max pixel distance to count as a corner hit

        Returns
        -------
        Index of the nearest corner if within threshold, else -1.
        """
        dists = numpy.sqrt((corners[:, 0].astype(float) - ix)**2 +
                           (corners[:, 1].astype(float) - iy)**2)
        ci = int(numpy.argmin(dists))
        return ci if dists[ci] <= threshold else -1

    def _roi_label_text(self, corners, prefix):
        """Build a status-bar string showing the bounding box of a ROI polygon.

        Example output: "ROI: x[10-30] y[5-25]"
        """
        xs = corners[:, 0]; ys = corners[:, 1]
        return f"{prefix}: x[{xs.min()}-{xs.max()}] y[{ys.min()}-{ys.max()}]"

    def _toggle_roi(self):
        """Toggle draw-ROI mode on/off.

        When active the next mouse drag on the image will define a new
        rectangular (hexagonal) ROI.  The button relief sinks to give
        visual feedback that the mode is active.
        """
        self._roi_drawing = not self._roi_drawing
        if self._roi_drawing:
            self.btn_roi.config(relief=tk.SUNKEN, text="ROI active")
        else:
            self.btn_roi.config(relief=tk.RAISED, text="Draw ROI")

    def _clear_roi(self):
        """Delete the signal ROI and reset all related state."""
        self._roi = None; self._roi_drawing = False; self._roi_moving = False
        self._roi_drag_corner = None
        self.btn_roi.config(relief=tk.RAISED, text="Draw ROI")
        self.btn_mvroi.config(relief=tk.RAISED, text="Move ROI")
        self.lb_roi.config(text="ROI: none")
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _toggle_move_roi(self):
        """Toggle Move ROI mode on/off.

        In move mode, dragging on the image translates the entire polygon.
        Dragging near a corner resizes only that corner.
        Only one ROI can be in move mode at a time (deactivates Move Bkg).
        """
        if self._roi is None:
            messagebox.showwarning("No ROI", "Draw ROI first.")
            return
        self._roi_moving = not self._roi_moving
        if self._roi_moving:
            # Deactivate Bkg move mode to avoid conflicting drag operations
            self._roi2_moving = False
            self.btn_mvroi2.config(relief=tk.RAISED, text="Move Bkg")
            self.btn_mvroi.config(relief=tk.SUNKEN, text="Moving\u2026")
        else:
            self.btn_mvroi.config(relief=tk.RAISED, text="Move ROI")

    def _move_roi(self, dx, dy):
        """Apply a pixel displacement (dx, dy) to the signal ROI.

        If _roi_drag_corner is set, only that corner moves (resize).
        Otherwise, the entire polygon is translated.

        `_roi_anchor` is a snapshot of the corner positions at the start
        of the drag, so all movements are relative to that snapshot and
        do not accumulate floating-point error over many mouse events.
        """
        if self._roi_anchor is None: return
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        if self._roi_drag_corner is not None:
            # Resize: move only the dragged corner, clamp to array bounds
            ci = self._roi_drag_corner
            new_corners = self._roi_anchor.copy()
            new_corners[ci, 0] = max(0, min(nx - 1, int(self._roi_anchor[ci, 0] + dx)))
            new_corners[ci, 1] = max(0, min(ny - 1, int(self._roi_anchor[ci, 1] + dy)))
            self._roi = new_corners
        else:
            # Translate: shift all corners by the same (dx, dy)
            new_corners = self._roi_anchor + numpy.array([[dx, dy]])
            new_corners[:, 0] = numpy.clip(new_corners[:, 0], 0, nx - 1)
            new_corners[:, 1] = numpy.clip(new_corners[:, 1], 0, ny - 1)
            self._roi = new_corners
        self.lb_roi.config(text=self._roi_label_text(self._roi, "ROI"))

    def _on_img_press(self, ev):
        """Handle mouse button press on the spatial image.

        Dispatch logic (priority order):
          1. If Move ROI is active: start a translate/resize drag.
             Check for a corner hit (resize) vs interior click (translate).
          2. Elif Move Bkg is active: same for the background ROI.
          3. Elif Draw ROI mode is active: record the drag start point.
          4. Else: move the cursor pixel to the clicked location and redraw.
        """
        # Give keyboard focus to the image widget so arrow keys work
        self._img_widget.focus_set()
        if ev.inaxes is not self.ai: return
        if not self.current_data or _arr(self.current_data).ndim < 3: return

        if self._roi_moving and self._roi is not None:
            # Convert click to pixel coords; check if near a corner (resize) or not
            ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
            ci = self._find_corner(self._roi, ix, iy)
            self._roi_drag_corner = ci if ci >= 0 else None   # None → translate
            self._roi_drag_start  = (ix, iy)
            self._roi_anchor      = self._roi.copy()   # snapshot for relative dragging
        elif self._roi2_moving and self._roi2 is not None:
            ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
            ci = self._find_corner(self._roi2, ix, iy)
            self._roi2_drag_corner = ci if ci >= 0 else None
            self._roi2_drag_start  = (ix, iy)
            self._roi2_anchor      = self._roi2.copy()
        elif self._roi_drawing:
            # Record drag start in calibrated (Matplotlib) coordinates;
            # we convert to pixels during motion/release
            self._roi_start = (ev.xdata, ev.ydata)
        else:
            # Default: move cursor to clicked pixel and refresh displays
            ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
            self._cix = ix; self._ciy = iy
            self._draw_img(self.sl_e.get()); self._draw_pt()

    def _on_img_motion(self, ev):
        """Handle mouse drag on the spatial image (live preview during drag).

        Updates the ROI polygon in real time while the mouse button is held
        so the user gets immediate visual feedback.
        """
        if ev.inaxes is not self.ai: return

        # Move/resize ROI drag
        if self._roi_moving and self._roi_drag_start is not None:
            ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
            dx = ix - self._roi_drag_start[0]
            dy = iy - self._roi_drag_start[1]
            self._move_roi(dx, dy)
            self._draw_img(self.sl_e.get())
            return

        # Move/resize Bkg drag
        if self._roi2_moving and self._roi2_drag_start is not None:
            ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
            dx = ix - self._roi2_drag_start[0]
            dy = iy - self._roi2_drag_start[1]
            self._move_roi2(dx, dy)
            self._draw_img(self.sl_e.get())
            return

        # Draw ROI drag — build a 6-corner hexagonal approximation of the
        # rectangle as the user sweeps the mouse.  The three left-edge corners
        # share the same x; the three right-edge corners share the other x.
        # The midpoint corners ensure the polygon has enough vertices for
        # smooth resizing and mask computation.
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
        """Handle mouse button release on the spatial image.

        Finalises whichever operation is in progress:
          - ROI/Bkg drag: apply the last displacement, clear drag state,
            redraw image and point spectrum.
          - Draw ROI: fix the polygon at the release position, exit draw mode.
        """
        # Finalise ROI translate/resize
        if self._roi_moving and self._roi_drag_start is not None:
            if ev.inaxes is self.ai:
                ix, iy = self._cal_to_pix(ev.xdata, ev.ydata)
                dx = ix - self._roi_drag_start[0]
                dy = iy - self._roi_drag_start[1]
                self._move_roi(dx, dy)
            # Clear drag state so the next click is treated as a new press
            self._roi_drag_start = None; self._roi_anchor = None
            self._roi_drag_corner = None
            self._draw_img(self.sl_e.get()); self._draw_pt()
            return

        # Finalise Bkg translate/resize
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

        # Finalise ROI drawing
        if not self._roi_drawing or self._roi_start is None: return
        if ev.inaxes is not self.ai: return
        ix0, iy0 = self._cal_to_pix(self._roi_start[0], self._roi_start[1])
        ix1, iy1 = self._cal_to_pix(ev.xdata, ev.ydata)
        lo_x, hi_x = min(ix0, ix1), max(ix0, ix1)
        lo_y, hi_y = min(iy0, iy1), max(iy0, iy1)
        mid_y = (lo_y + hi_y) // 2
        self._roi = numpy.array([[lo_x, lo_y], [lo_x, mid_y], [lo_x, hi_y],
                                 [hi_x, hi_y], [hi_x, mid_y], [hi_x, lo_y]])
        # Exit draw mode and restore the button appearance
        self._roi_start = None; self._roi_drawing = False
        self.btn_roi.config(relief=tk.RAISED, text="Draw ROI")
        self.lb_roi.config(text=self._roi_label_text(self._roi, "ROI"))
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _roi_spectrum(self):
        """Return the summed spectrum for all pixels inside the signal ROI.

        Uses _poly_mask() to rasterise the polygon into a boolean mask, then
        sums all masked pixels along the spatial axes.

        Returns
        -------
        (xc, units, spectrum_array) or None if no ROI is set or the mask is empty.
        """
        if not self.current_data or self._roi is None: return None
        d = _arr(self.current_data)
        if d.ndim < 3: return None
        mask = self._poly_mask(self._roi)
        if not mask.any(): return None
        # d[mask, :] selects all masked pixels; sum along axis 0 collapses spatial dims
        sp = d[mask, :].sum(axis=0)
        xc, u = _spec_cal(self.current_data)
        return xc, u, sp

    # ---- Background ROI interaction ----

    def _place_roi2(self):
        """Create the background ROI as a copy of the signal ROI, offset by ~half its width.

        The offset is chosen to place the Bkg ROI immediately next to the
        signal ROI on the image.  It is clamped so both edges stay within
        the array bounds.  The user can then fine-tune the placement with
        Move Bkg or arrow keys.

        Requires a signal ROI to already exist (enforced by the warning dialog).
        """
        if self._roi is None:
            messagebox.showwarning("No ROI", "Draw ROI first, then place Bkg.")
            return
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        xs = self._roi[:, 0]; ys = self._roi[:, 1]
        w   = int(xs.max() - xs.min())      # ROI width in pixels (row direction)
        off = max(3, w // 2)                # desired offset: half the width, min 3 px

        # Clamp offsets so the copied ROI stays within the image bounds
        dx = min(off, nx - 1 - int(xs.max()))   # max rightward shift before hitting edge
        dy = min(off, ny - 1 - int(ys.max()))   # max downward shift before hitting edge
        dx = max(dx, -int(xs.min()))             # max leftward shift before hitting edge
        dy = max(dy, -int(ys.min()))             # max upward shift before hitting edge

        self._roi2 = self._roi.copy() + numpy.array([[dx, dy]])
        self.lb_roi2.config(text=self._roi_label_text(self._roi2, "Bkg"))
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _toggle_move_roi2(self):
        """Toggle Move Bkg mode on/off.

        Mirrors _toggle_move_roi() for the background ROI.  Deactivates
        Move ROI mode if it is currently active (only one can move at a time).
        """
        if self._roi2 is None:
            messagebox.showwarning("No Bkg", "Place Bkg first.")
            return
        self._roi2_moving = not self._roi2_moving
        if self._roi2_moving:
            self._roi_moving = False   # deactivate signal ROI move mode
            self.btn_mvroi.config(relief=tk.RAISED, text="Move ROI")
            self.btn_mvroi2.config(relief=tk.SUNKEN, text="Moving\u2026")
        else:
            self.btn_mvroi2.config(relief=tk.RAISED, text="Move Bkg")

    def _clear_roi2(self):
        """Delete the background ROI and reset all related state."""
        self._roi2 = None; self._roi2_moving = False
        self._roi2_drag_start = None; self._roi2_anchor = None
        self._roi2_drag_corner = None
        self.btn_mvroi2.config(relief=tk.RAISED, text="Move Bkg")
        self.lb_roi2.config(text="Bkg: none")
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _move_roi2(self, dx, dy):
        """Apply a pixel displacement (dx, dy) to the background ROI.

        Identical logic to _move_roi() but operates on self._roi2 /
        self._roi2_anchor / self._roi2_drag_corner.
        See _move_roi() for a detailed explanation of the algorithm.
        """
        if self._roi2_anchor is None: return
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        if self._roi2_drag_corner is not None:
            # Resize: move only the dragged corner
            ci = self._roi2_drag_corner
            new_corners = self._roi2_anchor.copy()
            new_corners[ci, 0] = max(0, min(nx - 1, int(self._roi2_anchor[ci, 0] + dx)))
            new_corners[ci, 1] = max(0, min(ny - 1, int(self._roi2_anchor[ci, 1] + dy)))
            self._roi2 = new_corners
        else:
            # Translate: shift all corners uniformly
            new_corners = self._roi2_anchor + numpy.array([[dx, dy]])
            new_corners[:, 0] = numpy.clip(new_corners[:, 0], 0, nx - 1)
            new_corners[:, 1] = numpy.clip(new_corners[:, 1], 0, ny - 1)
            self._roi2 = new_corners
        self.lb_roi2.config(text=self._roi_label_text(self._roi2, "Bkg"))

    def _roi2_spectrum(self):
        """Return the summed spectrum for all pixels inside the background ROI.

        Identical to _roi_spectrum() but for self._roi2.
        Returns (xc, units, spectrum_array) or None.
        """
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
        """Shift the signal ROI by one pixel in the (dix, diy) direction.

        Called by _on_arrow_key() when Move Bkg is NOT active.
        All corners are shifted uniformly; corners are clamped to stay within
        the array bounds so the ROI cannot drift off the edge of the image.
        """
        if self._roi is None or not self.current_data: return
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        new_corners = self._roi + numpy.array([[dix, diy]])
        new_corners[:, 0] = numpy.clip(new_corners[:, 0], 0, nx - 1)
        new_corners[:, 1] = numpy.clip(new_corners[:, 1], 0, ny - 1)
        self._roi = new_corners
        self.lb_roi.config(text=self._roi_label_text(self._roi, "ROI"))
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _nudge_roi2(self, dix, diy):
        """Shift the background ROI by one pixel in the (dix, diy) direction.

        Called by _on_arrow_key() when Move Bkg IS active.
        Identical logic to _nudge_roi() but operates on _roi2.
        """
        if self._roi2 is None or not self.current_data: return
        nx, ny = _arr(self.current_data).shape[0], _arr(self.current_data).shape[1]
        new_corners = self._roi2 + numpy.array([[dix, diy]])
        new_corners[:, 0] = numpy.clip(new_corners[:, 0], 0, nx - 1)
        new_corners[:, 1] = numpy.clip(new_corners[:, 1], 0, ny - 1)
        self._roi2 = new_corners
        self.lb_roi2.config(text=self._roi_label_text(self._roi2, "Bkg"))
        self._draw_img(self.sl_e.get()); self._draw_pt()

    def _on_arrow_key(self, event):
        """Handle arrow-key events on the spatial image canvas.

        Arrow keys nudge the active ROI by one pixel in the pressed direction.
        The key-to-delta mapping uses the array axis convention:
          Up    → row+1  (higher row index = toward top of image)
          Down  → row-1
          Right → col+1
          Left  → col-1

        If Move Bkg mode is active, the background ROI is nudged;
        otherwise the signal ROI is nudged.
        """
        if not self.current_data or _arr(self.current_data).ndim < 3: return
        # Map key name to (Δrow, Δcol)
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
        """Normalise a 1-D spectrum according to the current var_norm setting.

        Three modes:
          "off"  : return the array unchanged (raw counts)
          "max"  : divide by the maximum absolute value → peak height = 1.0
          "area" : divide by the sum of absolute values → spectral area = 1.0

        The area normalisation is the same operation used per-pixel in
        _compute_bkgsub(), so spectra normalised this way are directly
        comparable to the background-subtracted cube.

        Returns a float64 array.  If the normalisation denominator is zero
        (blank pixel) the input array is returned unchanged to avoid NaN.
        """
        a = arr.astype(float)
        mode = self.var_norm.get()
        if mode == "max":
            mx = numpy.max(numpy.abs(a))
            return a / mx if mx > 0 else a
        elif mode == "area":
            area = numpy.sum(numpy.abs(a))
            return a / area if area > 0 else a
        return a   # "off" — no normalisation

    def _draw_pt(self):
        """Render the point-spectrum panel.

        Collects spectra from the cursor pixel, signal ROI, and background
        ROI; applies normalisation and any active subtraction mode; fits each
        spectrum with the selected peak model; then plots all curves on self.ap.

        Also:
          - annotates the plot with fit statistics (peak, FWHM, area)
          - draws a vertical red dashed line at _pt_vline_energy (if set)
          - uses var_norm, vsp, sxp, syp, var_sub to control display options

        This method is called whenever the cursor moves, the energy slider
        changes, normalisation changes, or the fit model changes.
        """
        if not self.current_data: return
        d = _arr(self.current_data)
        if d.ndim < 3: return
        xc, u = _spec_cal(self.current_data)
        # Extract the spectrum at the current cursor pixel and normalise it
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
        """Build a filename-safe suffix from the comment entry box.

        Returns "_<sanitised_comment>" if the entry is non-empty, or "" if empty.
        Sanitisation:
          1. Strip leading/trailing whitespace
          2. Remove any character that is not alphanumeric, space, underscore or hyphen
          3. Replace runs of whitespace with a single underscore
        This ensures the suffix can be used in a filename on all major OSes.
        """
        import re
        c = self.comment_entry.get().strip()
        if not c: return ""
        c = re.sub(r'[^\w\s\-]', '', c)   # remove special characters
        c = re.sub(r'\s+', '_', c)         # spaces → underscores
        return "_" + c if c else ""

    def _open_file(self):
        """Open an NHDF file and populate all GUI panels.

        Shows a file-open dialog, reads the file with read_data_and_metadata(),
        updates the info text box, triggers the summed-spectrum and info-bar
        refresh, and initialises the spatial image panel.
        """
        p = filedialog.askopenfilename(title="Open NHDF",
            filetypes=[("NHDF", "*.nhdf *.h5 *.hdf5"), ("All", "*.*")])
        if not p: return
        try:
            self.current_path = pathlib.Path(p)
            self.current_data = read_data_and_metadata(self.current_path)
            self.file_lbl.config(text=str(self.current_path), fg="black")
            # Write formatted metadata into the scrolled text box
            self.text.config(state=tk.NORMAL); self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, format_info(self.current_data))
            self.text.config(state=tk.DISABLED)
            self.status.config(text=f"Loaded: {self.current_path.name}")
            # Refresh plots and info bar, then initialise the image panel
            self._upd_plot(); self._upd_info(); self._setup_esl()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _save_file(self):
        """Save the currently loaded data back to disk in NHDF format.

        Proposes a filename of the form  YYYYMMDD_<stem><comment>_copy.nhdf
        and opens a save-as dialog.  All calibration metadata is preserved
        by save_data_and_metadata().
        """
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        cs = self._comment_suffix()
        dp = _date_prefix()
        dn = dp + (self.current_path.stem + cs + "_copy.nhdf" if self.current_path else "out.nhdf")
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
        """Export the raw data array as CSV file(s).

        For 1-D or 2-D data: saves a single .csv file.
        For 3-D data: saves one .csv per slice along axis 0 into a user-chosen
        folder, with filenames  YYYYMMDD_<stem><comment>_NNNN.csv.
        Each 2-D slice is written as a plain comma-separated matrix.
        This is a convenience export — for large datasets prefer Export DAT or DM3.
        """
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        d = _arr(self.current_data); cs = self._comment_suffix()
        if d.ndim <= 2:
            # Single CSV file for 1-D or 2-D data
            p = filedialog.asksaveasfilename(title="CSV", defaultextension=".csv",
                initialfile=(_date_prefix() + (self.current_path.stem + cs + ".csv"
                             if self.current_path else "o.csv")),
                filetypes=[("CSV", "*.csv")])
            if not p: return
            try:
                numpy.savetxt(p, d if d.ndim == 2 else d.reshape(1, -1), delimiter=",", fmt="%.8g")
                messagebox.showinfo("OK", f"→ {p}")
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Error", str(e))
        else:
            # For 3-D+ data: one CSV per slice along the first axis
            fld = filedialog.askdirectory(title="Folder for CSV slices")
            if not fld: return
            try:
                b   = _date_prefix() + (self.current_path.stem if self.current_path else "data") + cs
                out = pathlib.Path(fld)
                n   = d.shape[0]   # number of slices
                for i in range(n):
                    s = d[i]
                    if s.ndim > 2: s = s.reshape(-1, s.shape[-1])   # collapse higher dims
                    numpy.savetxt(str(out / f"{b}_{i:04d}.csv"), s, delimiter=",", fmt="%.8g")
                messagebox.showinfo("OK", f"{n} CSVs → {fld}")
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Error", str(e))

    def _export_sum(self):
        """Export the spatially summed spectrum as a two-column CSV.

        Columns: calibrated energy (eV) and total summed counts.
        The header row labels the energy column with the correct units string.
        This is useful for quick QC of the phonon spectrum and for fitting
        peak positions in external software.
        """
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        r = self._sum_sp()
        if not r: return
        xc, u, sp = r; cs = self._comment_suffix()
        p = filedialog.asksaveasfilename(title="Sum spectrum", defaultextension=".csv",
            initialfile=(_date_prefix() + (self.current_path.stem + cs + "_sum.csv"
                         if self.current_path else "sum.csv")),
            filetypes=[("CSV", "*.csv")])
        if not p: return
        try:
            # Write a header line then two columns: energy, counts
            h = f"{'energy(' + u + ')' if u else 'ch'},counts"
            numpy.savetxt(p, numpy.column_stack((xc, sp)), delimiter=",", fmt="%.8g",
                          header=h, comments="")
            messagebox.showinfo("OK", f"→ {p}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _export_dat(self):
        """Export datacube as float32 .dat + .txt sidecar (nhdf_converter_GUI_v0pt5 format).

        Layout on disk: C-order, shape (ne, nd0, nd1) — ne slowest, nd1 fastest.
        A matching .txt sidecar with calibrations is written alongside the .dat.
        Delegates to write_raw() imported from nhdf_converter_GUI_v0pt5.
        """
        if not _CONVERTER_OK:
            messagebox.showerror("Import error",
                f"nhdf_converter_GUI_v0pt5 could not be imported:\n{_CONVERTER_ERR}")
            return
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        d = _arr(self.current_data)
        if d.ndim < 3:
            messagebox.showinfo("", f"Data is {d.ndim}D, not a 3D cube."); return
        nd0, nd1, ne = d.shape
        base = self.current_path.stem if self.current_path else "data"
        cs = self._comment_suffix()
        default_name = f"{_date_prefix()}{base}{cs}.dat"
        p = filedialog.asksaveasfilename(title="Export DAT", defaultextension=".dat",
            initialfile=default_name,
            filetypes=[("Raw binary float32", "*.dat"), ("All", "*.*")])
        if not p: return
        try:
            # Axis 0 = rows (Y), axis 1 = cols (X), axis 2 = energy — matches
            # the (ny, nx, ne) convention expected by write_raw.
            cal_y = _gc(self.current_data, 0)
            cal_x = _gc(self.current_data, 1)
            cal_e = _gc(self.current_data, 2)
            ic = self.current_data.intensity_calibration
            cal_i = (ic.offset or 0., ic.scale or 1., ic.units or '')
            nbytes, txt_path = write_raw(p, d,
                                         cal_x=cal_x, cal_y=cal_y,
                                         cal_e=cal_e, cal_i=cal_i)
            self.status.config(text=f"DAT: {pathlib.Path(p).name}")
            messagebox.showinfo("Exported",
                f"Raw float32 datacube\n"
                f"Shape: ({ne}, {nd0}, {nd1})  [ne × dim0 × dim1]\n"
                f"Written: {nbytes:,} bytes\n→ {p}\n→ {txt_path}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Export error", f"{type(e).__name__}: {e}")

    def _export_dm3(self):
        """Export datacube as a GMS-compatible DM3 Spectrum Image.

        Delegates to build_dm3_from_nhdf() imported from nhdf_converter_GUI_v0pt5,
        which uses the embedded DM3 template to produce a file that opens directly
        in Gatan Microscopy Suite as an EELS Spectrum Image.
        """
        if not _CONVERTER_OK:
            messagebox.showerror("Import error",
                f"nhdf_converter_GUI_v0pt5 could not be imported:\n{_CONVERTER_ERR}")
            return
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        d = _arr(self.current_data)
        if d.ndim < 3:
            messagebox.showinfo("", f"Data is {d.ndim}D, not a 3D cube."); return
        nd0, nd1, ne = d.shape
        base = self.current_path.stem if self.current_path else "data"
        cs = self._comment_suffix()
        default_name = f"{_date_prefix()}{base}{cs}.dm3"
        p = filedialog.asksaveasfilename(title="Export DM3", defaultextension=".dm3",
            initialfile=default_name,
            filetypes=[("Gatan DM3", "*.dm3"), ("All", "*.*")])
        if not p: return
        try:
            # Axis 0 = rows (Y), axis 1 = cols (X), axis 2 = energy — matches
            # the (ny, nx, ne) convention expected by build_dm3_from_nhdf.
            cal_y = _gc(self.current_data, 0)
            cal_x = _gc(self.current_data, 1)
            cal_e = _gc(self.current_data, 2)
            ic = self.current_data.intensity_calibration
            cal_i = (ic.offset or 0., ic.scale or 1., ic.units or '')
            title = self.current_path.stem if self.current_path else "Spectrum Image"
            out_bytes = build_dm3_from_nhdf(d,
                                            cal_x=cal_x, cal_y=cal_y,
                                            cal_e=cal_e, cal_i=cal_i,
                                            title=title)
            dest = os.path.abspath(p)
            with open(dest, 'wb') as fh:
                fh.write(out_bytes)
            self.status.config(text=f"DM3: {pathlib.Path(p).name}")
            messagebox.showinfo("Exported",
                f"GMS-compatible DM3 Spectrum Image\n"
                f"Shape: ({nd0}, {nd1}, {ne})  [dim0 × dim1 × ne]\n"
                f"Written: {len(out_bytes):,} bytes\n→ {dest}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Export error", f"{type(e).__name__}: {e}")

    def _export_roi(self):
        """Export the signal ROI summed spectrum as a two-column CSV.

        Summing all pixels inside the polygon gives the highest SNR spectrum
        from the region of interest.  The filename encodes the ROI bounding
        box so the source region is always traceable from the filename alone.
        """
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
        # Encode the ROI bounding box in the filename for traceability
        default_name = (f"{_date_prefix()}{base}{cs}"
                        f"_spectrum_roi_x{xs.min()}-{xs.max()}_y{ys.min()}-{ys.max()}.csv")
        p = filedialog.asksaveasfilename(title="Export ROI spectrum", defaultextension=".csv",
            initialfile=default_name, filetypes=[("CSV", "*.csv")])
        if not p: return
        try:
            h = f"{'energy(' + u + ')' if u else 'channel'},counts"
            numpy.savetxt(p, numpy.column_stack((xc, sp)), delimiter=",", fmt="%.8g",
                          header=h, comments="")
            npix = int(self._poly_mask(self._roi).sum())   # count of summed pixels
            self.status.config(text=f"ROI: {pathlib.Path(p).name}")
            messagebox.showinfo("Exported",
                f"ROI x[{xs.min()}-{xs.max()}] y[{ys.min()}-{ys.max()}]\n"
                f"Pixels summed: {npix}\n"
                f"Channels: {sp.size}\n→ {p}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _export_bkg(self):
        """Export the background ROI summed spectrum as a two-column CSV.

        Identical structure to _export_roi() but operates on self._roi2.
        The filename encodes "background_bkg" and the ROI extent for clarity.
        """
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
        default_name = (f"{_date_prefix()}{base}{cs}"
                        f"_background_bkg_x{xs.min()}-{xs.max()}_y{ys.min()}-{ys.max()}.csv")
        p = filedialog.asksaveasfilename(title="Export Bkg spectrum", defaultextension=".csv",
            initialfile=default_name, filetypes=[("CSV", "*.csv")])
        if not p: return
        try:
            h = f"{'energy(' + u + ')' if u else 'channel'},counts"
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
        """Export the currently displayed subtraction spectrum as a two-column CSV.

        Replicates the same pair of spectra computed by _draw_pt() for the
        active var_sub mode, applies the same var_norm normalisation, and
        writes the difference to a CSV file.

        The filename encodes the subtraction type (e.g., "ROI-Bkg") and the
        normalisation mode so the source of the data is always clear from
        the filename.

        The most physically meaningful mode for phonon EELS background
        removal is "ROI − Bkg" (var_sub = "roi_roi2"), which subtracts the
        area-normalised background spectrum from the area-normalised signal
        spectrum, suppressing the elastic tail and leaving the phonon peaks.
        """
        if not self.current_data: messagebox.showwarning("", "Open first."); return
        d = _arr(self.current_data)
        if d.ndim < 3: return
        xc, u = _spec_cal(self.current_data)

        # Collect and normalise all three spectra
        sp_cur  = self._normalize(d[self._cix, self._ciy, :])
        roi_sp  = None
        roi_r   = self._roi_spectrum()
        if roi_r is not None:
            roi_sp = self._normalize(roi_r[2])
        roi2_sp = None
        roi2_r  = self._roi2_spectrum()
        if roi2_r is not None:
            roi2_sp = self._normalize(roi2_r[2])

        # Compute the requested difference spectrum
        sub_mode = self.var_sub.get()
        diff_sp = None; lbl_sub = ""
        if sub_mode == "cur_roi"  and roi_sp is not None:
            diff_sp = sp_cur - roi_sp;   lbl_sub = "cur-ROI"
        elif sub_mode == "roi_cur"  and roi_sp is not None:
            diff_sp = roi_sp - sp_cur;   lbl_sub = "ROI-cur"
        elif sub_mode == "cur_roi2" and roi2_sp is not None:
            diff_sp = sp_cur - roi2_sp;  lbl_sub = "cur-Bkg"
        elif sub_mode == "roi_roi2" and roi_sp is not None and roi2_sp is not None:
            diff_sp = roi_sp - roi2_sp;  lbl_sub = "ROI-Bkg"
        elif sub_mode == "roi2_roi" and roi_sp is not None and roi2_sp is not None:
            diff_sp = roi2_sp - roi_sp;  lbl_sub = "Bkg-ROI"
        elif sub_mode == "roi2_cur" and roi2_sp is not None:
            diff_sp = roi2_sp - sp_cur;  lbl_sub = "Bkg-cur"

        if diff_sp is None:
            messagebox.showwarning("No subtraction",
                "Select a Subtract mode and ensure the required ROIs exist.")
            return

        # Build filename including subtraction type and normalisation mode
        norm_mode   = self.var_norm.get()
        norm_tag    = "_norm_off" if norm_mode == "off" else f"_norm_{norm_mode}"
        base        = self.current_path.stem if self.current_path else "data"
        cs          = self._comment_suffix()
        default_name = f"{_date_prefix()}{base}{cs}_sub_{lbl_sub}{norm_tag}.csv"
        p = filedialog.asksaveasfilename(title=f"Export {lbl_sub}", defaultextension=".csv",
            initialfile=default_name, filetypes=[("CSV", "*.csv")])
        if not p: return
        try:
            # Column header names the subtraction and normalisation used
            col_name = lbl_sub + (f"({norm_mode})" if norm_mode != "off" else "")
            h = f"{'energy(' + u + ')' if u else 'channel'},{col_name}"
            numpy.savetxt(p, numpy.column_stack((xc, diff_sp)), delimiter=",", fmt="%.8g",
                          header=h, comments="")
            self.status.config(text=f"Sub: {pathlib.Path(p).name}")
            messagebox.showinfo("Exported",
                f"{lbl_sub} spectrum (norm: {norm_mode})\n→ {p}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _compute_bkgsub(self):
        """Compute the area-normalised, background-subtracted datacube.

        Shared by _export_bkgsub_dat and _export_bkgsub_dm3 so the normalization
        logic and its comments live in exactly one place.

        Returns
        -------
        result : ndarray, float64, shape (nd0, nd1, ne)
            Per-pixel area-normalised spectrum minus the area-normalised bkg spectrum.
        xs2 : ndarray   dim-0 corner indices of the bkg ROI (for filename tags).
        ys2 : ndarray   dim-1 corner indices of the bkg ROI.

        Returns None if any pre-condition is not met (error dialogs shown here).
        """
        if not self.current_data:
            messagebox.showwarning("", "Open first."); return None
        d = _arr(self.current_data)
        if d.ndim < 3:
            messagebox.showinfo("", f"Data is {d.ndim}D, not a 3D cube."); return None
        roi2_r = self._roi2_spectrum()
        if roi2_r is None:
            messagebox.showwarning("No Bkg",
                "Place a background ROI first.\nClick 'Place Bkg' to create one.")
            return None
        _, _, bkg_raw = roi2_r
        bkg_area = float(numpy.sum(numpy.abs(bkg_raw.astype(float))))
        if bkg_area == 0:
            messagebox.showwarning("Bkg is zero",
                "Background spectrum sums to zero; cannot area-normalize.")
            return None
        bkg_norm = bkg_raw.astype(float) / bkg_area

        # Area-normalize every pixel then subtract background
        d_f = d.astype(float)                                             # (nd0, nd1, ne)

        # Sum |counts| across all energy channels for every (nd0, nd1) pixel.
        # keepdims=True retains the trailing size-1 axis so the result broadcasts
        # directly against d_f (shape nd0 × nd1 × 1 vs nd0 × nd1 × ne).
        pixel_areas = numpy.sum(numpy.abs(d_f), axis=2, keepdims=True)   # (nd0, nd1, 1)

        # Step 1 of zero-pixel guard: replace any zero-area denominator with 1.0
        # before dividing.  A true zero spectrum divided by 1.0 still gives zero
        # for the numerator term, so this is a safe placeholder — but crucially it
        # prevents NaN / inf from propagating into the result array.  Without this,
        # numpy would raise a divide-by-zero warning and fill those pixels with NaN,
        # which would corrupt the entire output file.
        safe_areas = numpy.where(pixel_areas > 0, pixel_areas, 1.0)

        # Divide every pixel spectrum by its own area (area-normalise per pixel),
        # then subtract the area-normalised background spectrum via broadcasting.
        # For zero-area pixels the division yields 0/1 = 0, but then subtracting
        # bkg_norm would leave a spurious −bkg_norm ghost in those pixels, so a
        # second correction is applied immediately below (Step 2).
        result = d_f / safe_areas - bkg_norm[numpy.newaxis, numpy.newaxis, :]

        # Step 2 of zero-pixel guard: overwrite the pixels that had zero total
        # counts with a flat-zero spectrum.  This discards the −bkg_norm artifact
        # introduced in the line above for those pixels.  The physical meaning is:
        # a pixel with no counts carries no spectral information, so it should not
        # be filled with a negative ghost of the background after subtraction.
        result[pixel_areas[:, :, 0] == 0, :] = 0.0

        xs2 = self._roi2[:, 0]; ys2 = self._roi2[:, 1]
        return result, xs2, ys2

    # ================================================================
    # Temperature map via the detailed balance principle
    # ================================================================
    #
    # PHYSICS BACKGROUND
    # ------------------
    # In vibrational EELS the fast electron can either LOSE energy to the
    # sample (creating a phonon — the "loss side", positive energy) or
    # GAIN energy from the sample (annihilating a thermally-excited phonon
    # — the "gain side", negative energy).  The phonon occupation follows
    # Bose-Einstein statistics with occupation number
    #
    #     n(E) = 1 / [exp(E / k_B T) - 1]
    #
    # so the loss intensity scales as (n+1) and the gain intensity as n.
    # Their ratio at the same |E| cancels the phonon density-of-states
    # and leaves a pure Boltzmann factor:
    #
    #     I(+E) / I(-E)  =  (n+1) / n  =  exp(E / k_B T)
    #
    # Taking the natural logarithm linearises this:
    #
    #     ln[ I(+E) / I(-E) ]  =  E / (k_B T)
    #
    # which is a straight line through the origin with slope 1/(k_B T).
    # A least-squares fit forced through the origin gives:
    #
    #     slope = sum(E_i * lnR_i) / sum(E_i^2)
    #     T     = 1 / (k_B * slope)
    #
    # ENERGY WINDOW
    # -------------
    # The fit is restricted to the energy range  0.04 eV  to  0.08 eV.
    #
    #   E_min = 0.04 eV — avoids the zero-loss peak (ZLP) tails.  The ZLP
    #       dominates the spectrum near E = 0 and its wings contaminate both
    #       the loss and gain sides.  0.04 eV is safely beyond the ZLP for
    #       typical vibrational EELS data (ZLP FWHM ~ 10-15 meV).
    #
    #   E_max = 0.08 eV — avoids noise-dominated channels at higher energy
    #       losses where the gain-side signal is exponentially suppressed by
    #       the Boltzmann factor.  It also stays within the Fuchs-Kliewer
    #       phonon region (~0.067 eV for MgO) where the detailed balance
    #       relationship is physically meaningful.
    #
    # These values are appropriate for MgO vibrational EELS taken on a
    # monochromated STEM at room temperature and above.
    #
    # ================================================================

    # Fixed energy bounds for the detailed balance fit (eV).
    # These are intentionally hardcoded rather than auto-detected, because
    # the physically meaningful phonon region is well-known for MgO and
    # auto-detection can be fooled by artifacts or weak signal.
    _TEMP_E_MIN = 0.04   # lower bound: safely outside ZLP tails (eV)
    _TEMP_E_MAX = 0.08   # upper bound: within FK phonon region, above noise (eV)

    def _compute_temperature_map(self):
        """Compute a per-pixel temperature map from the detailed balance principle.

        Uses the background-subtracted cube (from _compute_bkgsub) and measures
        the loss/gain intensity ratio in the fixed energy window:

            E_min = 0.04 eV   (avoid ZLP contamination)
            E_max = 0.08 eV   (stay within phonon region, avoid noise)

        The ratio at each energy channel is:
            R(E) = I(+E) / I(-E) = exp(E / k_B T)
        and ln(R) vs E is fit with a line through the origin.

        Returns
        -------
        temp_map : ndarray, float64, shape (nd0, nd1)
            Temperature in Kelvin at each spatial pixel.
            NaN where the fit is invalid (negative slope, insufficient
            valid channels, or non-positive signal on either side).
        e_min    : float   Lower energy bound used (eV), always 0.04.
        e_max    : float   Upper energy bound used (eV), always 0.08.
        n_good   : int     Number of pixels that yielded a valid temperature.
        bkg_info : tuple   (xs2, ys2) — background-ROI corner index arrays,
                           carried through for filename tagging.

        Returns None if any pre-condition fails (data not loaded, no bkg ROI,
        ZLP at edge, etc.).  Error dialogs are shown by this method.
        """

        # ==============================================================
        # STEP 1: Obtain the background-subtracted datacube.
        # ==============================================================
        # _compute_bkgsub() returns the area-normalised, background-
        # subtracted cube of shape (nd0, nd1, ne).  At every spatial
        # pixel the spectrum has been divided by its own total area
        # (sum of absolute values) and then the identically-normalised
        # background spectrum has been subtracted.
        #
        # The area normalisation cancels perfectly in the loss/gain
        # ratio because both sides are divided by the same per-pixel
        # scalar, so the detailed balance formula is still valid.
        # The background subtraction isolates the phonon signal of
        # interest from the substrate/support contribution.
        # ==============================================================
        out = self._compute_bkgsub()
        if out is None:
            # _compute_bkgsub already showed the relevant error dialog
            # (no data loaded, no bkg ROI, etc.)
            return None
        result, xs2, ys2 = out
        nd0, nd1, ne = result.shape

        # ==============================================================
        # STEP 2: Build the calibrated energy axis and locate the ZLP.
        # ==============================================================
        # _spec_cal() reads the dimensional calibration from the NHDF
        # metadata and returns:
        #   xc : 1-D array of length ne — calibrated energy in eV
        #        (e.g., [-0.5, -0.499, ..., 0, ..., 1.0])
        #   eu : string — energy units (usually "eV")
        #
        # _e0idx() finds the channel index closest to E = 0 (the ZLP
        # centre).  Everything to the right of e0 is the loss side
        # (+E) and everything to the left is the gain side (-E).
        # ==============================================================
        xc, eu = _spec_cal(self.current_data)
        e0 = _e0idx(self.current_data)

        # Guard: the ZLP must not be at the very first or last channel,
        # otherwise we cannot mirror channels across it.
        if e0 <= 0 or e0 >= ne - 1:
            messagebox.showwarning("ZLP at edge",
                f"The zero-loss channel (index {e0}) is at the edge of the "
                f"spectrum ({ne} channels).  Cannot mirror loss/gain sides.")
            return None

        # Channel spacing in eV — used to convert the energy bounds
        # (0.04 eV, 0.08 eV) into channel-offset integers.
        scale = abs(float(xc[1] - xc[0])) if ne > 1 else 1.0

        # ==============================================================
        # STEP 3: Convert the fixed energy bounds to channel offsets.
        # ==============================================================
        # E_min = 0.04 eV — channels closer to the ZLP than this are
        #     contaminated by the zero-loss peak tails and would bias
        #     the ratio toward 1 (i.e., artificially high temperature).
        #
        # E_max = 0.08 eV — channels beyond this are dominated by noise
        #     on the gain side (the gain signal decays exponentially with
        #     energy via the Boltzmann factor).  Including noisy channels
        #     degrades the fit.  0.08 eV also keeps us within the MgO
        #     Fuchs-Kliewer phonon region (~0.067 eV peak).
        #
        # k_min_ch and k_max are integer offsets from the ZLP centre (e0).
        # For offset k:
        #   loss channel index = e0 + k   (positive energy)
        #   gain channel index = e0 - k   (negative energy, mirrored)
        # ==============================================================
        e_min = self._TEMP_E_MIN   # 0.04 eV
        e_max = self._TEMP_E_MAX   # 0.08 eV

        k_min_ch = int(numpy.ceil(e_min / scale))    # first usable offset
        k_max    = int(numpy.floor(e_max / scale))   # last usable offset

        # Clamp k_max so that both e0+k and e0-k stay within the array.
        # min(e0, ne-1-e0) is the maximum symmetric offset the array allows.
        k_max_possible = min(e0, ne - 1 - e0)
        k_max = min(k_max, k_max_possible)

        # Sanity check: we need at least one usable channel in the window.
        if k_min_ch > k_max:
            messagebox.showwarning("Insufficient range",
                f"The energy window [{e_min:.3f}, {e_max:.3f}] eV "
                f"(channel offsets [{k_min_ch}, {k_max}]) does not contain "
                f"any usable channels.\n\n"
                f"Channel spacing: {scale:.5f} eV/ch\n"
                f"Max symmetric offset: {k_max_possible} channels "
                f"({k_max_possible * scale:.4f} eV)\n\n"
                f"The spectrum may be too narrow or the ZLP too close to "
                f"the edge for this energy window.")
            return None

        # ==============================================================
        # STEP 4: Vectorised detailed-balance fit across all pixels.
        # ==============================================================
        #
        # Instead of looping over every (ix, iy) pixel (which would be
        # extremely slow for a large spectrum image), we loop over channel
        # offsets k and operate on the ENTIRE (nd0, nd1) spatial plane
        # at once using numpy broadcasting.  This is efficient because
        # the number of channels in [k_min, k_max] is typically small
        # (a few tens), while the number of pixels can be thousands.
        #
        # At each offset k we:
        #   1. Read the loss-side slice:  loss = result[:, :, e0 + k]
        #   2. Read the gain-side slice:  gain = result[:, :, e0 - k]
        #   3. Determine which pixels are valid: both loss > 0 AND
        #      gain > 0.  We cannot take the logarithm of zero or
        #      negative values (these arise where the background
        #      subtraction overcorrected or the signal is absent).
        #   4. Compute  lnR = ln(loss / gain)  at valid pixels.
        #   5. Accumulate the weighted least-squares sums:
        #        sum_E_lnR += E_k * lnR   (numerator)
        #        sum_E2    += E_k^2        (denominator)
        #        n_valid   += 1            (count of valid channels)
        #
        # After the loop, the slope at each pixel is:
        #     slope = sum_E_lnR / sum_E2
        #
        # This is the ordinary-least-squares solution for a line through
        # the origin (y = m*x), which minimises sum[(lnR_i - m*E_i)^2].
        #
        # The temperature is then:
        #     T = 1 / (k_B * slope)
        #
        # Only pixels with slope > 0 AND at least 3 valid channels are
        # kept.  Everything else is set to NaN (invalid / unphysical).
        # ==============================================================

        # Accumulators — one value per spatial pixel, initialised to zero.
        sum_E_lnR = numpy.zeros((nd0, nd1))    # numerator:   sum of E_k * ln(R_k)
        sum_E2    = numpy.zeros((nd0, nd1))    # denominator: sum of E_k^2
        n_valid   = numpy.zeros((nd0, nd1), dtype=int)  # count of valid channels

        for k in range(k_min_ch, k_max + 1):
            # E_k is the LOSS-SIDE energy for this offset — always positive.
            # It is the physical energy transfer from the beam to the sample.
            E_k = float(xc[e0 + k])

            # Extract the loss-side and gain-side intensity slices.
            # Each is a 2-D array of shape (nd0, nd1) — one value per pixel.
            loss = result[:, :, e0 + k]    # I(+E) at every pixel
            gain = result[:, :, e0 - k]    # I(-E) at every pixel (mirrored)

            # Validity mask: both sides must be strictly positive so that
            # the logarithm is defined and finite.  Pixels where either
            # side is zero or negative (due to background over-subtraction
            # or absent signal) are excluded from the fit at this channel.
            valid = (loss > 0) & (gain > 0)

            # Compute the Boltzmann log-ratio:  ln(I(+E) / I(-E)).
            # Initialise to zero so that invalid pixels contribute nothing
            # to the accumulator sums.
            lnR = numpy.zeros((nd0, nd1))
            lnR[valid] = numpy.log(loss[valid] / gain[valid])

            # Accumulate into the least-squares sums.
            # The multiplication by `valid` (boolean → 0 or 1) ensures that
            # invalid pixels add exactly zero to the sums.
            sum_E_lnR += E_k * lnR * valid
            sum_E2    += E_k ** 2 * valid
            n_valid   += valid.astype(int)

        # ==============================================================
        # STEP 5: Convert the fitted slope to temperature.
        # ==============================================================
        #
        # A pixel is considered "good" if:
        #   (a) sum_E2 > 0  — at least one channel contributed to the fit
        #   (b) n_valid >= 3 — at least 3 channels, so the fit is not
        #       determined by a single point (robustness against outliers)
        #
        # The slope must also be positive for the temperature to be
        # physically meaningful.  A negative slope would imply the gain
        # side is stronger than the loss side, which violates detailed
        # balance and indicates data problems (e.g., artefacts, wrong
        # background, or the spectrum is not thermal).
        # ==============================================================
        good = (sum_E2 > 0) & (n_valid >= 3)
        slope = numpy.zeros((nd0, nd1))
        slope[good] = sum_E_lnR[good] / sum_E2[good]

        # Initialise temperature map to NaN (= "no valid measurement").
        temp_map = numpy.full((nd0, nd1), numpy.nan)

        # Only compute T where the slope is positive (physical).
        # T = 1 / (k_B * slope),  with k_B = 8.617333e-5 eV/K.
        pos_slope = good & (slope > 0)
        temp_map[pos_slope] = 1.0 / (_kB_eV * slope[pos_slope])

        # Count how many pixels have a valid temperature value.
        n_good = int(numpy.count_nonzero(pos_slope))

        return temp_map, e_min, e_max, n_good, (xs2, ys2)

    def _export_temp_dm3(self):
        """Compute a temperature map and export it as a DM3 file.

        This is the button handler for "Temp Map…".  It:
          1. Calls _compute_temperature_map() to do the physics.
          2. Shows a diagnostics dialog (energy window, T statistics).
          3. Wraps the 2-D temperature map as a pseudo-3-D array with
             a single "energy" channel so it can be passed to
             build_dm3_from_nhdf() (which requires 3-D input).
          4. Writes the resulting DM3 bytes to disk.

        The exported DM3 opens in Gatan Microscopy Suite (GMS) as a
        spectrum image with 1 energy channel.  The spatial "survey"
        view shows the temperature map in Kelvin.
        """

        # ---- guard: make sure the converter module is available ----
        # write_raw and build_dm3_from_nhdf are imported at the top of
        # this file from nhdf_converter_GUI_v0pt5.  If that import
        # failed (file not found, dependency missing, etc.), we show
        # the stored error message.
        if not _CONVERTER_OK:
            messagebox.showerror("Import error",
                f"nhdf_converter_GUI_v0pt5 could not be imported:\n{_CONVERTER_ERR}")
            return

        # ---- run the temperature computation ----
        # _compute_temperature_map() handles all its own error dialogs
        # (no data, no bkg, ZLP at edge, insufficient range).  If any
        # of those fail it returns None and the user has already been
        # informed via a messagebox.
        out = self._compute_temperature_map()
        if out is None:
            return
        temp_map, e_min, e_max, n_good, bkg_info = out
        nd0, nd1 = temp_map.shape
        xs2, ys2 = bkg_info

        # ---- check that at least some pixels are valid ----
        if n_good == 0:
            messagebox.showwarning("No valid pixels",
                "Detailed balance fit failed at every pixel.\n"
                "This typically means the gain-side signal is negative or zero\n"
                "after background subtraction.  Check your Bkg ROI placement.")
            return

        # ---- show diagnostics to the user ----
        # Extract the valid (non-NaN) temperature values for statistics.
        valid_temps = temp_map[~numpy.isnan(temp_map)]
        n_ch = int(numpy.floor(e_max / abs(float(_gc(self.current_data, 2)[1])))
                   - numpy.ceil(e_min / abs(float(_gc(self.current_data, 2)[1])))) + 1
        info = (
            f"Energy window: [{e_min:.4f}, {e_max:.4f}] eV\n"
            f"  (0.04 eV lower bound avoids ZLP tails)\n"
            f"  (0.08 eV upper bound stays within FK phonon region)\n"
            f"Approx. channels in window: ~{max(n_ch, 0)}\n"
            f"Valid pixels: {n_good} / {nd0 * nd1}\n"
            f"\n"
            f"Mean T:   {numpy.mean(valid_temps):.1f} K\n"
            f"Median T: {numpy.median(valid_temps):.1f} K\n"
            f"Std T:    {numpy.std(valid_temps):.1f} K\n"
            f"Min T:    {numpy.min(valid_temps):.1f} K\n"
            f"Max T:    {numpy.max(valid_temps):.1f} K"
        )
        messagebox.showinfo("Temperature Map — Detailed Balance", info)

        # ---- prepare the 2-D map for DM3 export ----
        # NaN values are replaced with 0.0 because GMS (Gatan Microscopy
        # Suite) does not handle IEEE 754 NaN gracefully in float32 images.
        # Pixels that were NaN (= no valid temperature) will appear as 0 K
        # in the exported map — the user should interpret those as "no data".
        temp_out = numpy.where(numpy.isnan(temp_map), 0.0, temp_map)

        # build_dm3_from_nhdf() requires a 3-D input array of shape
        # (ny, nx, ne).  We add a trailing dimension of size 1 to turn
        # our 2-D temperature map into a (nd0, nd1, 1) pseudo-cube.
        # The resulting DM3 will have a single "energy" channel whose
        # value at each pixel is the temperature in Kelvin.
        temp_3d = temp_out[:, :, numpy.newaxis].astype(numpy.float32)

        # ---- build the default filename ----
        # Includes the date prefix, source filename, optional user comment,
        # and the background-ROI coordinates for traceability.
        base = self.current_path.stem if self.current_path else "data"
        cs = self._comment_suffix()
        bkg_tag = f"_bkg_d0{xs2.min()}-{xs2.max()}_d1{ys2.min()}-{ys2.max()}"
        default_name = f"{_date_prefix()}{base}{cs}_tempmap{bkg_tag}.dm3"

        # ---- ask the user where to save ----
        p = filedialog.asksaveasfilename(
            title="Export Temperature Map as DM3",
            defaultextension=".dm3", initialfile=default_name,
            filetypes=[("Gatan DM3", "*.dm3"), ("All", "*.*")])
        if not p:
            return   # user cancelled the dialog

        try:
            # ---- set up calibrations for the DM3 file ----
            # Spatial calibrations (cal_y for dim 0 = rows, cal_x for dim 1 = cols)
            # are copied directly from the original NHDF so that the temperature
            # map has the correct spatial scale and units (typically nm).
            cal_y = _gc(self.current_data, 0)   # dim 0 calibration (rows / Y)
            cal_x = _gc(self.current_data, 1)   # dim 1 calibration (cols / X)

            # The "energy" axis has only 1 channel and is labelled in Kelvin.
            # This is a dummy dimension required by the DM3 format for a
            # spectrum image.  The single channel holds the temperature value.
            cal_e = (0.0, 1.0, 'K')

            # The intensity calibration labels what the pixel values represent.
            # We set units to 'K' (Kelvin) so GMS displays "K" in the data bar.
            cal_i = (0.0, 1.0, 'K')

            # The title appears in the GMS window header.
            title = (self.current_path.stem if self.current_path
                     else "Temperature Map") + " (detailed balance)"

            # ---- call the DM3 builder ----
            # build_dm3_from_nhdf() assembles a complete, GMS-compatible DM3
            # file in memory and returns the raw bytes.  Internally it:
            #   - transposes the data to (ne, ny, nx) axis order
            #   - generates a grayscale survey thumbnail
            #   - patches calibrations and dimensions into the DM3 tag tree
            out_bytes = build_dm3_from_nhdf(temp_3d,
                                            cal_x=cal_x, cal_y=cal_y,
                                            cal_e=cal_e, cal_i=cal_i,
                                            title=title)

            # ---- write the bytes to disk ----
            dest = os.path.abspath(p)
            with open(dest, 'wb') as fh:
                fh.write(out_bytes)

            # ---- update status bar and confirm to the user ----
            self.status.config(text=f"Temp DM3: {pathlib.Path(p).name}")
            messagebox.showinfo("Exported",
                f"Temperature map (detailed balance)\n"
                f"Shape: ({nd0}, {nd1}, 1)\n"
                f"Energy window: [{e_min:.4f}, {e_max:.4f}] eV\n"
                f"Valid pixels: {n_good} / {nd0 * nd1}\n"
                f"Mean T: {numpy.mean(valid_temps):.1f} K\n"
                f"Written: {len(out_bytes):,} bytes\n\u2192 {dest}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Export error", f"{type(e).__name__}: {e}")

    def _export_bkgsub_dat(self):
        """Export area-normalized, background-subtracted datacube as float32 .dat + .txt sidecar.

        Computation delegated to _compute_bkgsub().
        Output layout: float32 C-order, shape (ne, dim0, dim1) — ne slowest, dim1 fastest.
        A .txt sidecar with calibrations is written alongside.
        """
        out = self._compute_bkgsub()
        if out is None: return
        result, xs2, ys2 = out
        nd0, nd1, ne = result.shape
        base = self.current_path.stem if self.current_path else "data"
        cs = self._comment_suffix()
        bkg_tag = f"_bkg_d0{xs2.min()}-{xs2.max()}_d1{ys2.min()}-{ys2.max()}"
        default_name = f"{_date_prefix()}{base}{cs}_areanorm_bkgsub{bkg_tag}.dat"
        p = filedialog.asksaveasfilename(
            title="Export area-norm bkg-subtracted cube",
            defaultextension=".dat", initialfile=default_name,
            filetypes=[("Raw binary float32", "*.dat"), ("All", "*.*")])
        if not p: return
        try:
            # Write .dat: transpose to (ne, nd0, nd1), float32, C-order, no header
            out_arr = numpy.ascontiguousarray(result.transpose(2, 0, 1), dtype=numpy.float32)
            out_arr.tofile(p)

            # Write .txt sidecar matching nhdf_converter_GUI write_raw format
            cal0 = _gc(self.current_data, 0)
            cal1 = _gc(self.current_data, 1)
            cal2 = _gc(self.current_data, 2)
            txt_path = os.path.splitext(p)[0] + '.txt'
            with open(txt_path, 'w') as fh:
                fh.write(f'DAT file: {os.path.basename(p)}\n')
                fh.write(f'Written : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                fh.write(f'Source  : {self.current_path.name if self.current_path else "unknown"}\n')
                fh.write(f'Processing: area-normalized per pixel, background subtracted\n')
                fh.write(f'  Background ROI: dim0 [{xs2.min()}-{xs2.max()}]'
                         f'  dim1 [{ys2.min()}-{ys2.max()}]\n')
                fh.write(f'  Bkg pixels summed: {int(self._poly_mask(self._roi2).sum())}\n')
                fh.write('\n')
                fh.write('Format\n')
                fh.write('------\n')
                fh.write(f'dtype      : float32 (IEEE 754 single precision, little-endian)\n')
                fh.write(f'axis order : C-order  (ne slowest, dim1 fastest)\n')
                fh.write(f'shape      : ({ne}, {nd0}, {nd1})  =  ne x dim0 x dim1\n')
                fh.write(f'values     : {ne * nd0 * nd1:,}\n')
                fh.write(f'file size  : {ne * nd0 * nd1 * 4:,} bytes'
                         f'  ({ne * nd0 * nd1 * 4 / 1024 / 1024:.3f} MB)\n')
                fh.write('\n')
                fh.write('Calibrations\n')
                fh.write('------------\n')
                fh.write(f'{"Axis":<14}{"dim":<6}{"offset (px)":<26}{"scale":<26}{"units"}\n')
                fh.write(f'{"-"*14}{"-"*6}{"-"*26}{"-"*26}{"-"*10}\n')
                for label, dim, cal in [('dim0', '0', cal0), ('dim1', '1', cal1),
                                        ('Energy', '2', cal2)]:
                    _off_px = round(cal[0] / cal[1]) if cal[1] != 0.0 else 0
                    fh.write(f'{label:<14}{dim:<6}{_off_px:<26}{cal[1]:<26.10g}{cal[2]}\n')
                if cal2[1] != 0.0:
                    _zlp_ch = round(-cal2[0] / cal2[1])
                    fh.write(f'  Start channel : 0\n')
                    fh.write(f'  End channel   : {ne - 1}\n')
                    fh.write(f'  ZLP channel   : {_zlp_ch}\n')

            nbytes = os.path.getsize(p)
            self.status.config(text=f"BkgSub DAT: {pathlib.Path(p).name}")
            messagebox.showinfo("Exported",
                f"Area-norm bkg-subtracted cube\n"
                f"Shape: ({ne}, {nd0}, {nd1})  [ne × dim0 × dim1]\n"
                f"Bkg ROI: dim0[{xs2.min()}-{xs2.max()}] dim1[{ys2.min()}-{ys2.max()}]\n"
                f"Written: {nbytes:,} bytes\n→ {p}\n→ {txt_path}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Export error", f"{type(e).__name__}: {e}")

    def _export_bkgsub_dm3(self):
        """Export area-normalized, background-subtracted datacube as a GMS-compatible DM3.

        Computation delegated to _compute_bkgsub().
        Calls build_dm3_from_nhdf() (imported from nhdf_converter_GUI_v0pt5) with the
        processed result array and the calibrations from the loaded file.
        """
        if not _CONVERTER_OK:
            messagebox.showerror("Import error",
                f"nhdf_converter_GUI_v0pt5 could not be imported:\n{_CONVERTER_ERR}")
            return
        out = self._compute_bkgsub()
        if out is None: return
        result, xs2, ys2 = out
        nd0, nd1, ne = result.shape
        base = self.current_path.stem if self.current_path else "data"
        cs = self._comment_suffix()
        bkg_tag = f"_bkg_d0{xs2.min()}-{xs2.max()}_d1{ys2.min()}-{ys2.max()}"
        default_name = f"{_date_prefix()}{base}{cs}_areanorm_bkgsub{bkg_tag}.dm3"
        p = filedialog.asksaveasfilename(
            title="Export area-norm bkg-subtracted cube as DM3",
            defaultextension=".dm3", initialfile=default_name,
            filetypes=[("Gatan DM3", "*.dm3"), ("All", "*.*")])
        if not p: return
        try:
            # Axis 0 = rows (Y), axis 1 = cols (X), axis 2 = energy — matches
            # the (ny, nx, ne) convention expected by build_dm3_from_nhdf.
            cal_y = _gc(self.current_data, 0)
            cal_x = _gc(self.current_data, 1)
            cal_e = _gc(self.current_data, 2)
            ic = self.current_data.intensity_calibration
            cal_i = (ic.offset or 0., ic.scale or 1., ic.units or '')
            title = self.current_path.stem if self.current_path else "Spectrum Image"
            out_bytes = build_dm3_from_nhdf(result,
                                            cal_x=cal_x, cal_y=cal_y,
                                            cal_e=cal_e, cal_i=cal_i,
                                            title=title)
            dest = os.path.abspath(p)
            with open(dest, 'wb') as fh:
                fh.write(out_bytes)
            self.status.config(text=f"BkgSub DM3: {pathlib.Path(p).name}")
            messagebox.showinfo("Exported",
                f"Area-norm bkg-subtracted DM3 Spectrum Image\n"
                f"Shape: ({nd0}, {nd1}, {ne})  [dim0 × dim1 × ne]\n"
                f"Bkg ROI: dim0[{xs2.min()}-{xs2.max()}] dim1[{ys2.min()}-{ys2.max()}]\n"
                f"Written: {len(out_bytes):,} bytes\n→ {dest}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Export error", f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    NHDFApp(root)
    root.mainloop()