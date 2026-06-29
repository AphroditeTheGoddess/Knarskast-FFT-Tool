"""
fft_analyzer_qt.py
==================
Planetary-gearbox vibration / strain FFT analyser - PyQt rewrite.

A clean-utility desktop tool for the HvA Maintenance-Lab planetary gearbox
rig. Loads enDAQ .IDE recordings, PhotonFirst FBG .csv files and generic
CSVs, computes single-sided amplitude / PSD / ASD spectra with configurable
windowing, detrending, zero-phase IIR / notch / Hampel filtering and Welch
averaging, overlays gear-mesh and sideband markers and the Miao AM/FM
expected-signal model, and compares fault stages H0-H5 in a stacked or
overlaid view that scrolls to an unlimited number of graphs.

This is a full port of the original Tkinter program. The numerical core
(gear kinematics, the FFT engine, FBG conversions, feature extraction and
the sideband-vs-severity correlation incl. the brute-force modulation-family
scan) is reused unchanged; only the UI layer is new - built on Qt for a
cleaner, more professional layout, native scroll areas and a native colour
picker with opacity.

Run:
    pip install PyQt6 numpy scipy matplotlib idelib      # (or PySide6)
    python fft_analyzer_qt.py
"""

# Pylance/Pyright: numpy, matplotlib and Qt ship type stubs that don't model
# array element types, Axes.spines, or .item()/.instance() returns, so the
# rules below fire only false positives in this file (it runs correctly).
# Scoped to this file; the rest of your project keeps full type checking.
# pyright: reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportOptionalOperand=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false

from __future__ import annotations

import os
import re
import sys
import gc
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal as sps
from scipy import stats as scistats
from scipy.ndimage import median_filter
from scipy.special import jv

if TYPE_CHECKING:
    # Type-checker view only: bind the Qt names to ONE concrete binding
    # (PyQt6). At runtime this whole branch is skipped and the `else` below
    # picks PyQt6 or PySide6. The else matters: without it, Pylance also
    # analyses the runtime loop and re-merges QtWidgets into an ambiguous
    # type, which makes every QWidget subclass look like it inherits from a
    # non-class ("Argument to class must be a base class").
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt
    QT_BINDING = "PyQt6"
    Signal = QtCore.pyqtSignal
else:
    # --- Qt binding: PyQt6 preferred, PySide6 fallback (scoped enums in both)
    QT_BINDING = None
    _qt_errors = []
    for _cand in ("PyQt6", "PySide6"):
        try:
            if _cand == "PyQt6":
                from PyQt6 import QtCore, QtGui, QtWidgets
                from PyQt6.QtCore import Qt
                Signal = QtCore.pyqtSignal
            else:
                from PySide6 import QtCore, QtGui, QtWidgets
                from PySide6.QtCore import Qt
                Signal = QtCore.Signal
            QT_BINDING = _cand
            os.environ.setdefault("QT_API", _cand.lower())
            break
        except Exception as _e:          # NOT just ImportError: a Windows Qt
            _qt_errors.append(           # DLL-load failure is an ImportError too
                f"  {_cand}: {type(_e).__name__}: {_e}")
            continue
    if QT_BINDING is None:
        raise SystemExit(
            "A Qt binding could not be loaded. Tried:\n"
            + "\n".join(_qt_errors)
            + "\n\nIf a binding IS installed but failed above with a DLL / load "
            "error, this is an environment clash (common in the Anaconda 'base' "
            "env, which ships its own Qt5). Fixes, best first:\n"
            "  1) Run from a clean conda env:\n"
            "       conda create -n thesis python=3.12\n"
            "       conda activate thesis\n"
            "       pip install PyQt6 numpy scipy matplotlib idelib\n"
            "  2) Force-reinstall the wheels in this env:\n"
            "       pip install --force-reinstall PyQt6 PyQt6-Qt6 PyQt6-sip\n"
            "  3) Clear stale Qt plugin variables, then retry:\n"
            "       set QT_PLUGIN_PATH=\n"
            "       set QT_QPA_PLATFORM_PLUGIN_PATH=")

import matplotlib
try:
    matplotlib.use("QtAgg")
except Exception:
    pass
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT

try:
    import idelib
except ImportError:
    idelib = None


# ==========================================================================
# CONFIG - defaults (all editable in the UI at runtime)
# ==========================================================================
CONFIG = {
    "Z_SUN": 21, "Z_RING": 93, "Z_PLANET": 36, "N_PLANETS": 3,
    "INPUT_RPM": 1500.0,
    "WINDOW": "hann", "SPECTRUM_TYPE": "amplitude",
    "DETREND": True, "USE_WELCH": False, "WELCH_SEGMENTS": 8,
    "FILTER_ENABLED": False, "FILTER_TYPE": "bandpass",
    "FILTER_FAMILY": "butter", "FILTER_LOW_HZ": 10.0,
    "FILTER_HIGH_HZ": 2000.0, "FILTER_ORDER": 4,
    "FILTER_RP_DB": 1.0, "FILTER_RS_DB": 40.0,
    "NOTCH_ENABLED": False, "NOTCH_HZ": 50.0, "NOTCH_Q": 30.0,
    "HAMPEL_ENABLED": False, "HAMPEL_WIN": 7, "HAMPEL_SIGMA": 3.0,
    "FS_OVERRIDE": False, "FS_OVERRIDE_HZ": 4000.0,
    "N_GMF_HARMONICS": 3, "N_SIDEBANDS": 3, "SIDEBAND_SPACING": "f_cpl",
    "FREQ_MAX_HZ": 1500.0, "LOG_X": False, "LOG_Y": False,
    "SCROLL_ZOOM_BASE": 1.25, "DARK_MODE": False,
    "BETA_PER_MM": 0.10, "B_OVER_A": 0.25, "TOOTH_HEIGHT_MM": 2.25,
    "STAGE_CUT_PCT": {0: 0, 1: 10, 2: 25, 3: 50, 4: 75, 5: 100},
    "OVERLAY_ALPHA": 0.6, "SB_WINDOW_HZ": 1.0, "GMF_SNAP_PCT": 1.0,
    "FBG_LAMBDA_B_NM": 1527.0, "FBG_PE": 0.21, "FBG_N_EFF": 1.468,
    "FBG_ETA": 8.3e-6, "FBG_ALPHA_NEFF": 0.809e-6, "TREND_LOWPASS_HZ": 1.0,
}

GRAPH_PALETTE = ["#0a84ff", "#ff375f", "#30d158", "#bf5af2", "#ff9f0a",
                 "#64d2ff", "#ff6482", "#ac8e68", "#ffd60a", "#8e8e93"]

# Upper bound on the comparison stack. A stacked view beyond this is
# unreadable anyway, and an unbounded stack makes the compare canvas tall
# enough to hit matplotlib's Agg 2**16-pixel limit and crash. Use overlay
# mode to view many spectra on one axes.
MAX_STACK = 40

# Stored spectra are kept as float32 to roughly halve RAM (important on
# low-memory machines). float32 carries ~7 significant digits, well beyond
# vibration/strain sensor precision, so graphs and analysis are unaffected.
# CSV export regenerates the EXACT float64 frequency axis from df and writes
# every computed bin, so exported data is full-resolution and precise.
# Set MEMORY_SAVER = False to keep float64 in memory.
MEMORY_SAVER = True
SPECTRUM_DTYPE = np.float32 if MEMORY_SAVER else np.float64


def _store(a):
    """Cast a spectrum array to the storage dtype (see MEMORY_SAVER)."""
    return np.asarray(a, dtype=SPECTRUM_DTYPE)


# ==========================================================================
# Numerical core (ported verbatim from the Tk version)
# ==========================================================================
def gear_frequencies(rpm, z_s, z_r, z_p, n_p):
    """Fixed-ring planetary characteristic frequencies (Lei 2014, Miao 2015):
    carrier f_c, gear mesh f_mesh, planet pass f_pp and planet-fault sideband
    spacing f_cpl = f_mesh / Z_p."""
    f_s = rpm / 60.0
    f_c = f_s / (1.0 + z_r / z_s)
    f_mesh = z_r * f_c
    return {"f_s": f_s, "f_c": f_c, "f_mesh": f_mesh,
            "f_pp": n_p * f_c, "f_cpl": f_mesh / z_p}


def stage_from_label(label):
    mt = re.search(r"H(\d+)", label, re.IGNORECASE)
    return int(mt.group(1)) if mt else None


def torque_from_label(label, default=None):
    """Parse a torque tag from a graph label: '<n>Nm' / '<n> Nm', or 'T<n>'
    (1-4 digits so a date like 'H4_20251030' is not mistaken for a torque).
    Returns the torque in Nm, or `default` if no tag is present."""
    m = re.search(r"(?<![A-Za-z0-9])(\d{1,4}(?:\.\d+)?)\s*Nm\b", label,
                  re.IGNORECASE)
    if not m:
        m = re.search(r"(?<![A-Za-z0-9])T[_\-]?(\d{1,4})(?![\d.])", label)
    return float(m.group(1)) if m else default


def theory_line_spectrum(A, B, n_sb):
    """Miao (2015) AM/FM line spectrum |C_j|, j = -n_sb..n_sb (C_0=carrier)."""
    return {j: abs(jv(j, B) + 0.5 * A * (jv(j - 1, B) + jv(j + 1, B)))
            for j in range(-n_sb, n_sb + 1)}


def stage_modulation(stage, beta, b_over_a, torque=None, torque_ref=None,
                     gamma=0.0):
    """Miao AM/FM modulation indices for a fault stage. Optional Bartelmus
    load-susceptibility scaling A -> A*(1 + gamma*T/T_ref) is applied only
    when torque, torque_ref and gamma are all truthy. gamma=0 (default) means
    load has no effect, i.e. the sideband/carrier ratio is load-independent:
    the linear-model null. A non-zero gamma models the faulty-gearbox load
    sensitivity (Bartelmus & Zimroz 2009)."""
    pct = CONFIG["STAGE_CUT_PCT"].get(stage)
    if pct is None:
        return None
    p_c = pct / 100.0 * CONFIG["TOOTH_HEIGHT_MM"]
    A = beta * p_c
    if torque and torque_ref and gamma:
        A = A * (1.0 + gamma * (float(torque) / float(torque_ref)))
    return p_c, A, A * b_over_a


def load_ide_channels(path):
    if idelib is None:
        raise RuntimeError("idelib is not installed (pip install idelib)")
    ds = idelib.importFile(path)
    channels = {}
    for chid, ch in ds.channels.items():
        units = [sc.units[0] for sc in ch.subchannels]
        if not ({"Acceleration", "Rotation", "Temperature"} & set(units)):
            continue
        label = f"[{chid}] {ch.name}"
        channels[label] = (chid, [sc.name for sc in ch.subchannels], units)
    return ds, channels


def extract_axis(dataset, channel_id, axis_index):
    el = dataset.channels[channel_id].getSession()
    arr = el.arraySlice()
    t = arr[0] * 1e-6
    vals = arr[1 + axis_index]
    fs = (len(t) - 1) / (t[-1] - t[0])
    return t, np.asarray(vals, dtype=float), fs


def _fbg_num(s):
    s = s.strip()
    if not s:
        return np.nan
    if s.count(",") == 1 and "." not in s:
        s = s.replace(",", ".")
    if s.count(".") > 1:
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return np.nan if v == 4294967295 else v


def load_fbg_csv(path):
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
    hdr_i = next((i for i, l in enumerate(lines)
                  if "packet_ts" in l.lower()), None)
    if hdr_i is None:
        raise ValueError("No 'packet_ts' header - not a PhotonFirst CSV?")
    hdr = lines[hdr_i]
    delim = ";" if hdr.count(";") >= hdr.count(",") else ","
    cols = [c.strip() for c in hdr.split(delim)]
    ts_i = next(i for i, c in enumerate(cols) if "packet_ts" in c.lower())
    rows = [l.split(delim) for l in lines[hdr_i + 1:] if l.strip()]

    def col(ci):
        return np.array([_fbg_num(r[ci]) if len(r) > ci else np.nan
                         for r in rows])

    ts = col(ts_i)
    data = {}
    for ci, name in enumerate(cols):
        if ci == ts_i or not name:
            continue
        vals = col(ci)
        if np.isfinite(vals).sum() < 10:
            continue
        if np.nanmedian(vals) > 1e5:
            vals = vals / 1000.0
        if not (1000.0 < np.nanmedian(vals) < 2000.0):
            continue
        idx = np.arange(len(vals))
        good = np.isfinite(vals)
        data[name] = np.interp(idx, idx[good], vals[good])
    if not data:
        raise ValueError("No wavelength columns (~1000-2000 nm) found.")
    tg = ts[np.isfinite(ts)]
    span = (tg[-1] - tg[0]) * 1e-6 if tg.size > 1 else 0.0
    fs = (len(tg) - 1) / span if span > 0 else 19230.0
    return data, fs


def load_generic_csv(path, value_cols, time_col=None, delim="auto",
                     fs_fallback=1000.0):
    raw = open(path, "r", errors="replace").read().splitlines()
    lines = [l for l in raw if l.strip()]
    if not lines:
        raise ValueError("empty file")
    want = [v.strip().lower() for v in value_cols if v.strip()]
    if not want:
        raise ValueError("no value columns specified")

    def pick_delim(line):
        if delim and delim != "auto":
            return {"tab": "\t"}.get(delim, delim)
        cand = {d: line.count(d) for d in (";", ",", "\t")}
        return max(cand, key=lambda k: cand[k])

    d = pick_delim(lines[0])
    hdr_i = cols = None
    for i, l in enumerate(lines):
        parts = [p.strip() for p in l.split(d)]
        low = [p.lower() for p in parts]
        if all(w in low for w in want):
            hdr_i, cols = i, parts
            break
    if hdr_i is None or cols is None:
        raise ValueError(f"header containing {value_cols} not found")
    idx = {c.lower(): k for k, c in enumerate(cols)}
    rows = [l.split(d) for l in lines[hdr_i + 1:]]

    def col(name):
        k = idx[name.lower()]
        return np.array([_fbg_num(r[k]) if len(r) > k else np.nan
                         for r in rows])

    data = {}
    for v in value_cols:
        if not v.strip():
            continue
        vals = col(v)
        good = np.isfinite(vals)
        if good.sum() < 2:
            continue
        ii = np.arange(len(vals))
        data[v.strip()] = np.interp(ii, ii[good], vals[good])
    if not data:
        raise ValueError("no usable numeric data in the requested columns")
    fs = fs_fallback
    if time_col and time_col.strip() and time_col.strip().lower() in idx:
        tg = col(time_col)
        tg = tg[np.isfinite(tg)]
        if tg.size > 1 and (tg[-1] - tg[0]) > 0:
            fs = (len(tg) - 1) / (tg[-1] - tg[0])
    return data, fs


def design_iir(fs, ftype, family, order, f_lo, f_hi, rp=1.0, rs=40.0):
    nyq = fs / 2.0
    if ftype in ("bandpass", "bandstop"):
        Wn = [max(f_lo, 1e-6) / nyq, min(f_hi, 0.99 * nyq) / nyq]
    elif ftype == "lowpass":
        Wn = min(f_hi, 0.99 * nyq) / nyq
    elif ftype == "highpass":
        Wn = max(f_lo, 1e-6) / nyq
    else:
        raise ValueError(f"unknown filter type: {ftype}")
    if family == "butter":
        return sps.butter(order, Wn, btype=ftype, output="sos")
    if family == "cheby1":
        return sps.cheby1(order, rp, Wn, btype=ftype, output="sos")
    if family == "cheby2":
        return sps.cheby2(order, rs, Wn, btype=ftype, output="sos")
    if family == "bessel":
        return sps.bessel(order, Wn, btype=ftype, output="sos", norm="phase")
    if family == "ellip":
        return sps.ellip(order, rp, rs, Wn, btype=ftype, output="sos")
    raise ValueError(f"unknown filter family: {family}")


def notch_filter(x, fs, f0, q):
    b, a = sps.iirnotch(f0, q, fs)
    return sps.filtfilt(b, a, x)


def hampel_filter(x, win, n_sigma):
    x = np.asarray(x, dtype=float)
    size = 2 * int(win) + 1
    med = median_filter(x, size=size, mode="nearest")
    mad = 1.4826 * median_filter(np.abs(x - med), size=size, mode="nearest")
    out = x.copy()
    spike = (mad > 0) & (np.abs(x - med) > n_sigma * mad)
    out[spike] = med[spike]
    return out


def infer_input_rpm(f_mesh_meas, z_s, z_r):
    f_s = f_mesh_meas * (z_s + z_r) / (z_r * z_s)
    return 60.0 * f_s, f_s


def amplitude_spectrum(x, fs, window, detrend, use_welch, n_segments,
                       kind="amplitude"):
    x = np.asarray(x, dtype=float)
    if detrend:
        x = x - np.mean(x)
    if kind in ("psd", "asd"):
        if use_welch:
            nperseg = max(256, len(x) // max(1, n_segments))
            f, pxx = sps.welch(x, fs=fs, window=window, nperseg=nperseg,
                               scaling="density")
        else:
            # scipy accepts detrend=False at runtime (already mean-removed
            # above); its type stub only lists str, hence the ignore.
            f, pxx = sps.periodogram(x, fs=fs, window=window,
                                     scaling="density",
                                     detrend=False)  # type: ignore[arg-type]
        return f, (np.sqrt(pxx) if kind == "asd" else pxx)
    if use_welch:
        nperseg = max(256, len(x) // max(1, n_segments))
        f, pxx = sps.welch(x, fs=fs, window=window, nperseg=nperseg,
                           scaling="spectrum")
        return f, np.sqrt(2.0 * pxx)
    n = len(x)
    w = sps.get_window(window, n)
    cg = np.sum(w) / n
    spec = np.fft.rfft(x * w)
    amp = (2.0 / n) * np.abs(spec) / cg
    amp[0] /= 2.0
    return np.fft.rfftfreq(n, d=1.0 / fs), amp


def spec_ylabel(sp):
    u = sp.get("unit", "g")
    return {"amplitude": f"Amp [{u}]", "psd": f"PSD [{u}\u00b2/Hz]",
            "asd": f"ASD [{u}/\u221aHz]",
            "trend": f"T [{u}]"}.get(sp.get("ytype", "amplitude"),
                                     f"Amp [{u}]")


def decimate(f, a, x0, x1, width_px):
    """Min-max decimation: one (min,max) pair per pixel column over [x0,x1]."""
    i0, i1 = np.searchsorted(f, [x0, x1])
    i0, i1 = max(i0 - 1, 0), min(i1 + 1, len(f))
    fs_, as_ = f[i0:i1], a[i0:i1]
    n_bins = max(int(width_px), 2000)
    if len(fs_) <= 2 * n_bins:
        return fs_, as_
    idx = np.linspace(0, len(fs_) - 1, n_bins + 1).astype(int)
    starts = idx[:-1]
    mins = np.minimum.reduceat(as_, starts)
    maxs = np.maximum.reduceat(as_, starts)
    mids = (idx[:-1] + idx[1:]) // 2
    out_f = np.empty(2 * n_bins)
    out_a = np.empty(2 * n_bins)
    out_f[0::2], out_f[1::2] = fs_[starts], fs_[mids]
    out_a[0::2], out_a[1::2] = mins, maxs
    return out_f, out_a


# ==========================================================================
# Parameters - one object captures every analysis setting (UI-independent).
# ==========================================================================
@dataclass
class Params:
    rpm: float = CONFIG["INPUT_RPM"]
    zs: int = CONFIG["Z_SUN"]
    zr: int = CONFIG["Z_RING"]
    zp: int = CONFIG["Z_PLANET"]
    np_: int = CONFIG["N_PLANETS"]
    n_harm: int = CONFIG["N_GMF_HARMONICS"]
    n_sb: int = CONFIG["N_SIDEBANDS"]
    sb_spacing: str = CONFIG["SIDEBAND_SPACING"]
    window: str = CONFIG["WINDOW"]
    spec_type: str = CONFIG["SPECTRUM_TYPE"]
    detrend: bool = CONFIG["DETREND"]
    use_welch: bool = CONFIG["USE_WELCH"]
    welch_segs: int = CONFIG["WELCH_SEGMENTS"]
    filt_on: bool = CONFIG["FILTER_ENABLED"]
    filt_type: str = CONFIG["FILTER_TYPE"]
    filt_family: str = CONFIG["FILTER_FAMILY"]
    f_lo: float = CONFIG["FILTER_LOW_HZ"]
    f_hi: float = CONFIG["FILTER_HIGH_HZ"]
    f_order: int = CONFIG["FILTER_ORDER"]
    f_rp: float = CONFIG["FILTER_RP_DB"]
    f_rs: float = CONFIG["FILTER_RS_DB"]
    notch_on: bool = CONFIG["NOTCH_ENABLED"]
    notch_hz: float = CONFIG["NOTCH_HZ"]
    notch_q: float = CONFIG["NOTCH_Q"]
    hampel_on: bool = CONFIG["HAMPEL_ENABLED"]
    hampel_win: int = CONFIG["HAMPEL_WIN"]
    hampel_sigma: float = CONFIG["HAMPEL_SIGMA"]
    fs_override_on: bool = CONFIG["FS_OVERRIDE"]
    fs_override_hz: float = CONFIG["FS_OVERRIDE_HZ"]
    fbg_quantity: str = "microstrain"
    f_max: float = CONFIG["FREQ_MAX_HZ"]
    log_x: bool = CONFIG["LOG_X"]
    log_y: bool = CONFIG["LOG_Y"]
    beta: float = CONFIG["BETA_PER_MM"]
    boa: float = CONFIG["B_OVER_A"]
    torque: float = 0.0
    torque_ref: float = 10.0
    gamma: float = 0.0
    show_avg: bool = False
    avg_only: bool = False
    sg_window: str = CONFIG["WINDOW"]
    sg_detrend: bool = CONFIG["DETREND"]
    sg_nperseg: int = 1024
    sg_overlap_pct: float = 75.0
    sg_db: bool = True
    sg_cmap: str = "viridis"
    sg_fmax: float = CONFIG["FREQ_MAX_HZ"]
    sg_filt_on: bool = CONFIG["FILTER_ENABLED"]
    sg_filt_type: str = CONFIG["FILTER_TYPE"]
    sg_filt_family: str = CONFIG["FILTER_FAMILY"]
    sg_f_lo: float = CONFIG["FILTER_LOW_HZ"]
    sg_f_hi: float = CONFIG["FILTER_HIGH_HZ"]
    sg_f_order: int = CONFIG["FILTER_ORDER"]
    sg_f_rp: float = CONFIG["FILTER_RP_DB"]
    sg_f_rs: float = CONFIG["FILTER_RS_DB"]
    sg_notch_on: bool = CONFIG["NOTCH_ENABLED"]
    sg_notch_hz: float = CONFIG["NOTCH_HZ"]
    sg_notch_q: float = CONFIG["NOTCH_Q"]
    sg_hampel_on: bool = CONFIG["HAMPEL_ENABLED"]
    sg_hampel_win: int = CONFIG["HAMPEL_WIN"]
    sg_hampel_sigma: float = CONFIG["HAMPEL_SIGMA"]
    show_theory: bool = False
    theory_alpha: float = CONFIG["OVERLAY_ALPHA"]
    theory_stage: int = 3
    gmf_focus_k: int = 1
    focus_custom_hz: float = 0.0
    mk_freq: float = 0.0
    mk_label: str = ""
    mk_lo: float = 0.0
    mk_hi: float = 0.0
    graph_px: int = 220
    sp_left: float = 0.10
    sp_right: float = 0.97
    sp_top_px: int = 46
    sp_bot_px: int = 48
    sp_hspace: float = 0.45


# ==========================================================================
# Analysis built on Params (pure; no Qt dependency)
# ==========================================================================
def open_path(path):
    if path.lower().endswith(".csv"):
        data, fs = load_fbg_csv(path)
        nodes = sorted(data)
        chans = {"[FBG] PhotonFirst": ("FBG", nodes,
                                       ["Wavelength"] * len(nodes))}
        return ("fbg", data, fs), chans
    ds, chans = load_ide_channels(path)
    return ("ide", ds, None), chans


def open_generic(path, spec):
    data, fs = load_generic_csv(path, spec["value_cols"], spec.get("time_col"),
                                spec.get("delim", "auto"), spec.get("fs", 1e3))
    cols = sorted(data)
    chans = {f"[CSV] {os.path.basename(path)}":
             ("CSV", cols, ["a.u."] * len(cols))}
    return ("gen", data, fs), chans


def close_handle(handle):
    if isinstance(handle, tuple) and handle and handle[0] == "ide":
        try:
            handle[1].close()
        except Exception:
            pass


def extract_named(handle, chans, ch_label, axis_name, P):
    if handle[0] == "fbg":
        data, fs = handle[1], handle[2]
        wl = data[axis_name]
        dwl = wl - np.mean(wl)
        lamB = CONFIG["FBG_LAMBDA_B_NM"]
        if P.fbg_quantity == "temperature":
            sens = (lamB / CONFIG["FBG_N_EFF"]) * (
                CONFIG["FBG_ALPHA_NEFF"] + CONFIG["FBG_ETA"])
            T = dwl / sens
            fc = min(CONFIG["TREND_LOWPASS_HZ"], 0.45 * fs)
            sos = sps.butter(4, fc / (fs / 2), "low", output="sos")
            return sps.sosfiltfilt(sos, T), fs, "\u00b0C", "time"
        if P.fbg_quantity == "microstrain":
            eps = dwl / (lamB * (1 - CONFIG["FBG_PE"])) * 1e6
            return eps, fs, "\u00b5\u03b5", "freq"
        return dwl * 1000.0, fs, "pm", "freq"
    if handle[0] == "gen":
        return handle[1][axis_name], handle[2], "a.u.", "freq"
    chid, subs, units = chans[ch_label]
    aidx = subs.index(axis_name)
    _, x, fs_det = extract_axis(handle[1], chid, aidx)
    if units[aidx] == "Temperature":
        return x, fs_det, "\u00b0C", "time"
    return x, fs_det, "g", "freq"


def get_fs(fs_detected, P):
    return float(P.fs_override_hz) if P.fs_override_on else fs_detected


def apply_filter_chain(x, fs, P):
    parts = []
    if P.hampel_on:
        x = hampel_filter(x, int(P.hampel_win), float(P.hampel_sigma))
        parts.append(f"Hampel(\u00b1{int(P.hampel_win)},{P.hampel_sigma:g}\u03c3)")
    if P.filt_on:
        sos = design_iir(fs, P.filt_type, P.filt_family, int(P.f_order),
                         float(P.f_lo), float(P.f_hi), float(P.f_rp),
                         float(P.f_rs))
        x = sps.sosfiltfilt(sos, x)
        if P.filt_type in ("bandpass", "bandstop"):
            band = f"{P.f_lo:g}-{P.f_hi:g}Hz"
        elif P.filt_type == "lowpass":
            band = f"<{P.f_hi:g}Hz"
        else:
            band = f">{P.f_lo:g}Hz"
        parts.append(f"{P.filt_family} {P.filt_type} {band}")
    if P.notch_on:
        x = notch_filter(x, fs, float(P.notch_hz), float(P.notch_q))
        parts.append(f"notch {P.notch_hz:g}Hz")
    return x, ((", " + " + ".join(parts)) if parts else "")


def process_signal(x, fs, label_base, unit, P):
    x, filt_txt = apply_filter_chain(x, fs, P)
    f, amp = amplitude_spectrum(x, fs, P.window, P.detrend, P.use_welch,
                                P.welch_segs, kind=P.spec_type)
    df = float(f[1] - f[0])                     # exact, before any downcast
    pos = amp[amp > 0]
    floor = float(pos.min()) if pos.size else 1e-9
    return {"label": f"{label_base} ({P.window}{filt_txt})",
            "f": _store(f), "amp": _store(amp), "ytype": P.spec_type,
            "unit": unit, "domain": "freq", "df": df,
            "floor": floor, "fs": fs, "n": len(x)}, len(x)


def make_trend_spec(x, fs, label, unit, src=None):
    t = np.arange(len(x)) / fs
    return {"label": f"{label} (T trend)", "f": _store(t),
            "amp": _store(np.asarray(x, float)), "ytype": "trend",
            "unit": unit, "domain": "time", "df": 1.0 / fs, "floor": 1e-9,
            "fs": fs, "n": len(x), "src": src}


def marker_params(P):
    gf = gear_frequencies(P.rpm, P.zs, P.zr, P.zp, P.np_)
    return gf, gf[P.sb_spacing]


def refine_gmf(sp, gf):
    fm = gf["f_mesh"]
    band = fm * CONFIG["GMF_SNAP_PCT"] / 100.0
    i0, i1 = np.searchsorted(sp["f"], [fm - band, fm + band])
    if i1 - i0 < 3:
        return gf, fm, 0.0
    seg = sp["amp"][i0:i1]
    fm_meas = float(sp["f"][i0:i1][int(np.argmax(seg))])
    scale = fm_meas / fm
    return {k: v * scale for k, v in gf.items()}, fm_meas, (scale - 1.0) * 100


def carrier_width(sp, fm_meas):
    f, amp = sp["f"], sp["amp"]
    ex = 0.5 if sp.get("ytype") == "psd" else 1.0
    i0, i1 = np.searchsorted(f, [fm_meas - 3.0, fm_meas + 3.0])
    if i1 - i0 < 5:
        return 0.0
    seg = amp[i0:i1] ** ex
    ip = i0 + int(np.argmax(seg))
    half = (amp[ip] ** ex) / np.sqrt(2.0)
    il = ip
    while il > 0 and amp[il] ** ex > half and f[ip] - f[il] < 5.0:
        il -= 1
    ir, n = ip, len(f)
    while ir < n - 1 and amp[ir] ** ex > half and f[ir] - f[ip] < 5.0:
        ir += 1
    return float(f[ir] - f[il])


def extract_features(sp, gf, n_sb, w=None):
    f, amp = sp["f"], sp["amp"]
    ex = 0.5 if sp.get("ytype") == "psd" else 1.0
    w = w if w is not None else CONFIG["SB_WINDOW_HZ"]
    fm, fcpl = gf["f_mesh"], gf["f_cpl"]

    def peak(fc):
        i0, i1 = np.searchsorted(f, [fc - w, fc + w])
        seg = amp[i0:i1]
        return float(seg.max()) ** ex if seg.size else 0.0

    carrier = peak(fm)
    line_js = [j for j in range(-n_sb, n_sb + 1) if j != 0]
    line_amps = {j: peak(fm + j * fcpl) for j in line_js}
    ratio = sum(line_amps.values()) / carrier if carrier > 0 else np.nan
    lo, hi = fm - (n_sb + 2) * fcpl, fm + (n_sb + 2) * fcpl
    i0, i1 = np.searchsorted(f, [lo, hi])
    fseg, aseg = f[i0:i1], amp[i0:i1] ** ex
    mask = np.ones(len(fseg), dtype=bool)
    for n in range(-n_sb, n_sb + 1):
        fc = fm + n * fcpl
        mask &= ~((fseg > fc - w) & (fseg < fc + w))
    noise_seg = aseg[mask]
    if noise_seg.size:
        med = float(np.median(noise_seg))
        mad = float(np.median(np.abs(noise_seg - med)))
        floor = med + 3.0 * 1.4826 * mad
    else:
        floor = np.nan
    sb1 = max(line_amps.get(-1, 0.0), line_amps.get(1, 0.0))
    return {"carrier": carrier, "ratio": ratio, "floor": floor,
            "sb1": sb1, "detected": bool(sb1 > floor), "line_amps": line_amps}


def sideband_ratio(sp, fm_meas, carrier, spacing, w, n_sb):
    if not (carrier > 0) or spacing <= 0:
        return np.nan
    f, amp = sp["f"], sp["amp"]
    ex = 0.5 if sp.get("ytype") == "psd" else 1.0

    def peak(fc):
        i0, i1 = np.searchsorted(f, [fc - w, fc + w])
        seg = amp[i0:i1]
        return float(seg.max()) ** ex if seg.size else 0.0

    tot = sum(peak(fm_meas + j * spacing)
              for j in range(-n_sb, n_sb + 1) if j != 0)
    return tot / carrier


def correlation_report(spectra, P):
    """Sideband-vs-severity correlation across staged graphs + brute-force
    modulation-family scan. Returns (lines, status) or (None, reason)."""
    if len(spectra) < 3:
        return None, "needs \u22653 graphs with H<n> stage labels"
    gf, _ = marker_params(P)
    n_sb = P.n_sb
    rows, row_specs, skipped = [], [], []
    for sp in spectra:
        if sp.get("domain") != "freq":
            skipped.append(sp["label"])
            continue
        stage = stage_from_label(sp["label"])
        T_i = torque_from_label(sp["label"], P.torque)
        mod = (stage_modulation(stage, P.beta, P.boa, torque=T_i,
                                torque_ref=P.torque_ref, gamma=P.gamma)
               if stage is not None else None)
        if mod is None:
            skipped.append(sp["label"])
            continue
        p_c, A, B = mod
        C = theory_line_spectrum(A, B, n_sb)
        pred = sum(C[j] for j in C if j != 0) / C[0]
        gf_i, fm_meas, dev_pct = refine_gmf(sp, gf)
        w_i = float(np.clip(carrier_width(sp, fm_meas),
                            CONFIG["SB_WINDOW_HZ"], 0.35 * gf_i["f_cpl"]))
        feat = extract_features(sp, gf_i, n_sb, w=w_i)
        js = sorted(j for j in C if j != 0)
        mvec = np.array([feat["line_amps"][j] for j in js])
        pvec = np.array([C[j] / C[0] * feat["carrier"] for j in js])
        fit_r = (float(scistats.pearsonr(pvec, mvec)[0])
                 if len(js) >= 3 and np.std(mvec) > 0 and np.std(pvec) > 0
                 else np.nan)
        rows.append((stage, p_c, pred, feat, sp["label"], fit_r, fm_meas,
                     dev_pct, w_i, T_i))
        row_specs.append(sp)
    if len(rows) < 3:
        return None, "fewer than 3 graphs have parsable H<n> stages"
    paired = sorted(zip(rows, row_specs), key=lambda rs: rs[0][1])
    rows = [r for r, _s in paired]
    row_specs = [s for _r, s in paired]
    pc = np.array([r[1] for r in rows])
    meas = np.array([r[3]["ratio"] for r in rows])
    pred = np.array([r[2] for r in rows])
    rho, p_rho = scistats.spearmanr(pc, meas)
    reg_slope, reg_intercept, reg_r, reg_p, reg_stderr = \
        scistats.linregress(pc, meas)
    ok = pred > 0
    r_mp, p_mp = (scistats.pearsonr(pred[ok], meas[ok])
                  if ok.sum() >= 3 else (np.nan, np.nan))

    families = [("f_cpl", "planet-fault f_cpl"), ("f_pp", "planet-pass  f_pp"),
                ("f_c", "carrier      f_c"), ("f_s", "shaft        f_s")]
    fam_results = []
    for key, name in families:
        mvals = []
        for r, sp_i in zip(rows, row_specs):
            fm_i, carrier_i = r[6], r[3]["carrier"]
            scale_i = (fm_i / gf["f_mesh"]) if gf["f_mesh"] else 1.0
            spacing_i = gf[key] * scale_i
            wf = float(np.clip(carrier_width(sp_i, fm_i),
                               CONFIG["SB_WINDOW_HZ"], 0.40 * spacing_i))
            mvals.append(sideband_ratio(sp_i, fm_i, carrier_i, spacing_i,
                                        wf, n_sb))
        mvals = np.asarray(mvals, float)
        good = np.isfinite(mvals)
        if good.sum() >= 3 and np.std(mvals[good]) > 0:
            rho_f, p_f = scistats.spearmanr(pc[good], mvals[good])
            okp = good & (pred > 0)
            r_f = (float(scistats.pearsonr(pred[okp], mvals[okp])[0])
                   if okp.sum() >= 3 and np.std(mvals[okp]) > 0
                   and np.std(pred[okp]) > 0 else np.nan)
        else:
            rho_f, p_f, r_f = np.nan, np.nan, np.nan
        fam_results.append([name, key, rho_f, p_f, r_f])

    def _score(fr):
        _n, _k, rho_f, p_f, r_f = fr
        if not np.isfinite(rho_f):
            return (-2.0, -2.0, -2.0)
        sig = 1 if (np.isfinite(p_f) and p_f < 0.05 and rho_f > 0) else 0
        return (sig, (r_f if np.isfinite(r_f) else -1.0), rho_f)
    best = max(fam_results, key=_score) if fam_results else None

    L = ["CORRELATION ANALYSIS \u2014 sidebands vs fault severity",
         f"GMF nominal = {gf['f_mesh']:.2f} Hz (snapped per graph, "
         f"\u00b1{CONFIG['GMF_SNAP_PCT']:g}%), f_cpl = {gf['f_cpl']:.2f} Hz, "
         f"n_sb = {n_sb}",
         f"Model: dK/K = \u03b2\u00b7p_c, \u03b2 = {P.beta:g}/mm, "
         f"B/A = {P.boa:g}", "",
         "Per graph (sorted by cut depth):",
         f"{'stage':>5} {'p_c[mm]':>8} {'GMF meas':>9} {'rpm meas':>9} "
         f"{'win\u00b1Hz':>7} {'ratio meas':>11} {'ratio pred':>11} "
         f"{'fit r':>7} {'SB>noise':>9} {'T[Nm]':>7}"]
    for st, p, pr, ft, lab, fit_r, fm_meas, devp, w_i, T_i in rows:
        L.append(f"{'H%d' % st:>5} {p:8.3f} {fm_meas:9.2f} "
                 f"{P.rpm * (1 + devp / 100):9.1f} {w_i:7.2f} "
                 f"{ft['ratio']:11.4f} {pr:11.4f} {fit_r:7.3f} "
                 f"{'YES' if ft['detected'] else 'no':>9} {T_i:7.1f}")
    L += ["", "Trend with damage (monotone increase expected):",
          f"  Spearman \u03c1 = {rho:.3f}, p = {p_rho:.4f}",
          f"  Linear fit: slope = {reg_slope:.4f} \u00b1 {reg_stderr:.4f} "
          f"per mm, R\u00b2 = {reg_r ** 2:.3f}, p = {reg_p:.4f}",
          "", "Agreement with AM/FM model prediction:",
          (f"  Pearson r(meas, pred) = {r_mp:.3f}, p = {p_mp:.4f}"
           if np.isfinite(r_mp) else
           "  (needs \u22653 stages with non-zero prediction)"),
          "", "MODULATION-FAMILY SCAN (brute-force peak search):",
          "Families isolated at the planet-fault, planet-pass, carrier and",
          "shaft spacings; strongest bin per line is taken so speed drift /",
          "smear cannot hide a line.",
          f"{'family':>20} {'spacing Hz':>11} {'Spearman \u03c1':>12} "
          f"{'p':>8} {'r vs model':>11}"]
    for name, key, rho_f, p_f, r_f in fam_results:
        L.append(f"{name:>20} {gf[key]:11.2f} {rho_f:12.3f} {p_f:8.4f} "
                 f"{r_f:11.3f}")
    if best is not None and np.isfinite(best[2]):
        bn, _bk, brho, bp, br = best
        if np.isfinite(bp) and bp < 0.05 and brho > 0:
            L += ["", f"  Best match: {bn.strip()} \u2014 significant rising "
                  f"trend (\u03c1 = {brho:.2f}, p = {bp:.3f})"
                  + (f", model agreement r = {br:.2f}." if np.isfinite(br)
                     else ".")]
        else:
            L += ["", f"  Best (weak): {bn.strip()} (\u03c1 = {brho:.2f}, "
                  f"p = {bp:.3f}); no family shows a significant rising trend."]
    L += ["  Note: the IDE accelerometer is housing-mounted, so it picks up",
          "  every source in the box; a family can read ABOVE the AM/FM",
          "  prediction from unrelated content. Trust the rank trend with",
          "  damage (\u03c1) over the absolute ratio, and compare families.",
          "", "INTERPRETATION:"]
    if p_rho < 0.05 and rho > 0:
        L.append("  + Significant POSITIVE monotone trend: sideband level\n"
                 f"    rises with cut depth (\u03c1 = {rho:.2f}, p = {p_rho:.3f}"
                 ", >95% certainty).")
    elif p_rho < 0.05:
        L.append("  - Significant NEGATIVE trend \u2014 opposite to the model;\n"
                 "    check feature extraction before interpreting.")
    else:
        L.append(f"  - No significant monotone trend (p = {p_rho:.2f}): the\n"
                 "    data does not statistically support a damage-sideband\n"
                 "    relation yet.")
    if any(r[3]["ratio"] > 1 for r in rows):
        L.append("  ! Some measured ratios exceed 1: 'sideband' windows hold\n"
                 "    peaks larger than the carrier \u2014 likely contamination\n"
                 "    or a missed peak. Inspect with the overlay before trust.")
    if n_sb < 2:
        L.append("  ! fit r is nan because n_sb = 1 gives only 2 lines; set\n"
                 "    'Sideband pairs' to 2 or more and re-analyze.")
    L += ["", f"n = {len(rows)} graphs"
          + (f"; skipped (no H<n>): {len(skipped)}" if skipped else ""),
          "Caution: n is small with one graph per stage \u2014 for the thesis",
          "claim, repeat per run (3 runs \u00d7 stages)."]
    return L, f"Spearman \u03c1 = {rho:.3f} (p = {p_rho:.4f})"


def acquisition_report(sp, spectra, P):
    fs = float(sp.get("fs") or 0.0)
    fs_det = float(sp.get("fs_detected", fs))
    nyq = fs / 2.0
    is_freq = sp.get("domain") == "freq"
    L = ["ANALYSIS", "", "Acquisition:",
         f"  fs detected (timestamps) : {fs_det:12.4f} Hz",
         f"  fs used                  : {fs:12.4f} Hz"]
    if P.fs_override_on:
        dev = (fs - fs_det) / fs_det * 100.0 if fs_det else float("nan")
        L.append(f"  override vs detected     : {dev:+12.2f} %")
    L.append(f"  Nyquist                  : {nyq:12.4f} Hz")
    if sp.get("n"):
        L += [f"  samples N                : {sp['n']:12,}",
              f"  duration                 : {sp['n'] / fs:12.3f} s"]
    L.append(f"  resolution \u0394f            : {sp['df']:12.5f} Hz")
    gf, _ = marker_params(P)
    L += ["", "Gear-mesh frequency / speed:",
          f"  set-point RPM            : {P.rpm:12.1f} rpm",
          f"  nominal f_s              : {gf['f_s']:12.4f} Hz",
          f"  nominal GMF (Z_r\u00b7f_c)    : {gf['f_mesh']:12.4f} Hz"]
    if is_freq:
        _gf, fm_meas, dev_pct = refine_gmf(sp, gf)
        rpm_meas, fs_meas = infer_input_rpm(fm_meas, P.zs, P.zr)
        L += [f"  measured GMF peak        : {fm_meas:12.4f} Hz",
              f"  deviation from nominal   : {dev_pct:+12.2f} %",
              f"  inferred f_s             : {fs_meas:12.4f} Hz",
              f"  inferred input RPM       : {rpm_meas:12.1f} rpm"]
        if abs(dev_pct) > 2.0:
            L.append("    ! >2% speed deviation \u2014 motor ran off set point.")
        L += ["", "Harmonic coverage (below Nyquist?):"]
        for k in range(1, P.n_harm + 1):
            fh = k * gf["f_mesh"]
            L.append(f"  GMF \u00d7{k} = {fh:10.2f} Hz : "
                     + ("ok" if fh < nyq else "ABOVE NYQUIST \u2014 aliased"))
    else:
        L.append("  (compute a frequency spectrum to infer RPM)")
    corr, status = correlation_report(spectra, P)
    if corr:
        L += ["", "=" * 72, ""] + corr
    else:
        L += ["", "Correlation across fault stages:", f"  (skipped \u2014 {status})"]
    return "\n".join(L)


# ==========================================================================
# Appearance - Apple-flavoured light / dark palettes + a Qt stylesheet
# ==========================================================================
THEMES = {
    "light": {"bg": "#f5f5f7", "panel": "#ffffff", "fg": "#1d1d1f",
              "muted": "#6e6e73", "border": "#d2d2d7", "accent": "#0a84ff",
              "field": "#ffffff", "hover": "#ececf0",
              "fig": "#ffffff", "ax": "#ffffff", "grid": "#e5e5ea",
              "axfg": "#1d1d1f", "line": "#0a84ff"},
    "dark": {"bg": "#1c1c1e", "panel": "#2c2c2e", "fg": "#f5f5f7",
             "muted": "#98989d", "border": "#3a3a3c", "accent": "#0a84ff",
             "field": "#3a3a3c", "hover": "#48484a",
             "fig": "#1c1c1e", "ax": "#2c2c2e", "grid": "#3a3a3c",
             "axfg": "#f5f5f7", "line": "#0a84ff"},
}

_FONT = ('"SF Pro Text", "Helvetica Neue", "Segoe UI", "Inter", '
         '"Cantarell", system-ui, sans-serif')


def build_qss(t):
    return f"""
    * {{ font-family: {_FONT}; font-size: 13px; color: {t['fg']};
         outline: none; }}
    QMainWindow, QWidget#root {{ background: {t['bg']}; }}
    QLabel {{ background: transparent; }}
    QLabel#muted {{ color: {t['muted']}; }}
    QLabel#h1 {{ font-size: 15px; font-weight: 600; }}
    QLabel#section {{ color: {t['muted']}; font-size: 11px; font-weight: 600;
        letter-spacing: 0.6px; }}
    QFrame#card {{ background: {t['panel']}; border: 1px solid {t['border']};
        border-radius: 12px; }}
    QFrame#sidebar {{ background: {t['panel']};
        border-right: 1px solid {t['border']}; }}
    QFrame#toolbar {{ background: {t['panel']};
        border-bottom: 1px solid {t['border']}; }}
    QFrame#statusbar {{ background: {t['panel']};
        border-top: 1px solid {t['border']}; }}
    QPushButton {{ background: {t['field']}; border: 1px solid {t['border']};
        border-radius: 8px; padding: 6px 12px; }}
    QPushButton:hover {{ background: {t['hover']}; }}
    QPushButton:pressed {{ background: {t['border']}; }}
    QPushButton:disabled {{ color: {t['muted']}; }}
    QPushButton#primary {{ background: {t['accent']}; color: white;
        border: 1px solid {t['accent']}; font-weight: 600; }}
    QPushButton#primary:hover {{ background: #3395ff; }}
    QPushButton#ghost {{ background: transparent;
        border: 1px solid transparent; padding: 5px 9px; font-size: 15px; }}
    QPushButton#ghost:hover {{ background: {t['hover']}; }}
    QPushButton#seg {{ background: transparent; border: none;
        border-radius: 7px; padding: 5px 14px; color: {t['muted']}; }}
    QPushButton#seg:checked {{ background: {t['panel']}; color: {t['fg']};
        font-weight: 600; }}
    QWidget#segwrap {{ background: {t['hover']}; border-radius: 9px; }}
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {t['field']};
        border: 1px solid {t['border']}; border-radius: 7px;
        padding: 4px 8px; min-height: 20px; }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border: 1px solid {t['accent']}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{ background: {t['panel']};
        border: 1px solid {t['border']}; border-radius: 8px;
        selection-background-color: {t['accent']}; selection-color: white;
        padding: 4px; }}
    QListWidget {{ background: {t['field']}; border: 1px solid {t['border']};
        border-radius: 10px; padding: 4px; }}
    QListWidget::item {{ border-radius: 7px; padding: 6px 8px; }}
    QListWidget::item:selected {{ background: {t['accent']}; color: white; }}
    QListWidget::item:hover:!selected {{ background: {t['hover']}; }}
    QTabWidget::pane {{ border: none; top: -1px; }}
    QTabBar::tab {{ background: transparent; color: {t['muted']};
        padding: 6px 12px; margin-right: 2px; border-radius: 7px; }}
    QTabBar::tab:selected {{ background: {t['hover']}; color: {t['fg']};
        font-weight: 600; }}
    QCheckBox {{ spacing: 7px; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px;
        border: 1px solid {t['border']}; background: {t['field']}; }}
    QCheckBox::indicator:checked {{ background: {t['accent']};
        border: 1px solid {t['accent']}; }}
    QScrollArea {{ border: none; background: {t['bg']}; }}
    QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {t['border']};
        border-radius: 5px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {t['muted']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 11px;
        margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {t['border']};
        border-radius: 5px; min-width: 30px; }}
    QToolTip {{ background: {t['panel']}; color: {t['fg']};
        border: 1px solid {t['border']}; border-radius: 6px; padding: 4px 6px; }}
    QPlainTextEdit, QTextBrowser {{ background: {t['field']};
        border: 1px solid {t['border']}; border-radius: 10px; padding: 8px; }}
    """


# ==========================================================================
# Custom widgets
# ==========================================================================
class MplCanvas(FigureCanvasQTAgg):
    """Figure canvas that routes the wheel: in a scrollable compare view a
    plain wheel scrolls the stack (event passed to the QScrollArea); hold
    Ctrl, or be in single / overlay view, to zoom at the cursor."""

    def __init__(self, fig, app):
        super().__init__(fig)
        self._app = app

    def wheelEvent(self, event):
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if (self._app and self._app.mode == "compare"
                and self._app._scrollable and not ctrl):
            event.ignore()
        else:
            super().wheelEvent(event)


class PlotScrollArea(QtWidgets.QScrollArea):
    def __init__(self, app):
        super().__init__()
        self._app = app

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._app:
            self._app._fit_canvas()


# Spin boxes / combos in a scroll area normally swallow the wheel and change
# their value when you only meant to scroll the sidebar. These ignore the
# wheel unless they actually hold focus, so the event bubbles up to the
# QScrollArea and scrolls instead. Click (or tab) into one to wheel-adjust it.
class _NoScrollSpin(QtWidgets.QSpinBox):
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class _NoScrollDoubleSpin(QtWidgets.QDoubleSpinBox):
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class _NoScrollCombo(QtWidgets.QComboBox):
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class LayoutDialog(QtWidgets.QDialog):
    """Exact-value plot-layout controls (spin boxes) + 'Fit to window'."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Plot layout")
        self.setMinimumWidth(320)
        form = QtWidgets.QFormLayout(self)
        form.setSpacing(8)
        p = app.p
        self.h = QtWidgets.QSpinBox(); self.h.setRange(60, 900)
        self.h.setValue(p.graph_px); self.h.setSuffix(" px")
        self.left = QtWidgets.QDoubleSpinBox(); self.left.setRange(0.02, 0.40)
        self.left.setSingleStep(0.01); self.left.setDecimals(2)
        self.left.setValue(p.sp_left)
        self.right = QtWidgets.QDoubleSpinBox(); self.right.setRange(0.60, 0.99)
        self.right.setSingleStep(0.01); self.right.setDecimals(2)
        self.right.setValue(p.sp_right)
        self.top = QtWidgets.QSpinBox(); self.top.setRange(10, 160)
        self.top.setValue(p.sp_top_px); self.top.setSuffix(" px")
        self.bot = QtWidgets.QSpinBox(); self.bot.setRange(10, 160)
        self.bot.setValue(p.sp_bot_px); self.bot.setSuffix(" px")
        self.hsp = QtWidgets.QDoubleSpinBox(); self.hsp.setRange(0.0, 1.5)
        self.hsp.setSingleStep(0.05); self.hsp.setDecimals(2)
        self.hsp.setValue(p.sp_hspace)
        form.addRow("Graph height", self.h)
        form.addRow("Left edge", self.left)
        form.addRow("Right edge", self.right)
        form.addRow("Top gap", self.top)
        form.addRow("Bottom gap", self.bot)
        form.addRow("Gap between plots", self.hsp)
        for w in (self.h, self.top, self.bot, self.left, self.right, self.hsp):
            w.valueChanged.connect(self._apply)
        btns = QtWidgets.QHBoxLayout()
        fit = QtWidgets.QPushButton("Fit all in window"); fit.clicked.connect(self._fit)
        reset = QtWidgets.QPushButton("Reset"); reset.clicked.connect(self._reset)
        btns.addWidget(fit); btns.addWidget(reset)
        form.addRow(btns)

    def _apply(self):
        p = self.app.p
        p.graph_px = self.h.value(); p.sp_left = self.left.value()
        p.sp_right = self.right.value(); p.sp_top_px = self.top.value()
        p.sp_bot_px = self.bot.value(); p.sp_hspace = self.hsp.value()
        self.app.rerender_keep_zoom()

    def _fit(self):
        n = max(1, len(self.app._views))
        vh = max(1, self.app._view_h)
        self.h.setValue(max(80, int(vh / n)))

    def _reset(self):
        self.h.setValue(220); self.left.setValue(0.10); self.right.setValue(0.97)
        self.top.setValue(46); self.bot.setValue(48); self.hsp.setValue(0.45)


class TextDialog(QtWidgets.QDialog):
    def __init__(self, parent, title, text):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(770, 650)
        lay = QtWidgets.QVBoxLayout(self)
        view = QtWidgets.QPlainTextEdit()
        view.setReadOnly(True)
        view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        view.setPlainText(text)
        view.setFont(QtGui.QFont("Menlo", 11))
        lay.addWidget(view)


class HelpDialog(QtWidgets.QDialog):
    def __init__(self, parent, html):
        super().__init__(parent)
        self.setWindowTitle("Help \u2014 FFT Analyser")
        self.resize(680, 680)
        lay = QtWidgets.QVBoxLayout(self)
        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setHtml(html)
        lay.addWidget(view)


# ==========================================================================
# Cute kitty (the single most important feature) - YOUR ASSETS GO HERE
# ==========================================================================
# All of the following are OPTIONAL. Anything left empty falls back to a
# built-in vector kitty / synthesised meow. You can mix base64 and file
# paths freely. File size is not a concern - paste away.
#
# IMAGE shown in the window. Provide ONE of (base64 wins if both set):
#   * KITTY_IMAGE_B64  : base64 of a PNG or JPG (format auto-detected)
#   * KITTY_IMAGE_PATH : a path to a .png / .jpg on disk
KITTY_IMAGE_B64 = ""        # <-- paste base64 of your kitty photo here
KITTY_IMAGE_PATH = ""       # <-- ...or give a file path instead, e.g.
#                                   r"C:\Users\Pepijn\Pictures\kitty.png"
#
# MEOW SOUNDS - give as MANY as you like; a RANDOM one plays per click.
# Each entry may be .wav OR .mp3. Use base64, file paths, or both.
#   * MEOW_B64   : list of base64 strings (WAV or MP3; auto-detected)
#   * MEOW_PATHS : list of file paths (.wav / .mp3)
MEOW_B64 = [
    # "<paste base64 of meow1.mp3 here>",
    # "<paste base64 of meow2.mp3 here>",
    # "<paste base64 of meow3.mp3 here>",
    # "<paste base64 of meow4.mp3 here>",
]
MEOW_PATHS = [
    # r"C:\Users\Pepijn\Desktop\meows\meow1.mp3",
    # r"C:\Users\Pepijn\Desktop\meows\meow2.mp3",
    # r"C:\Users\Pepijn\Desktop\meows\meow3.mp3",
    # r"C:\Users\Pepijn\Desktop\meows\meow4.mp3",
]


def kitty_pixmap(size, t):
    """Draw a built-in cute kitty face to a QPixmap (used when no base64
    image is pasted). Pure QPainter, so it needs no external assets."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    cx = cy = size / 2.0
    s = size / 280.0                      # design grid is 280 px
    fur = QtGui.QColor("#b9b6c4" if t.get("fig", "#fff") != "#1c1c1e"
                       else "#c9c6d4")
    dark = QtGui.QColor("#3a3a3c")
    pink = QtGui.QColor("#ff9bb3")
    accent = QtGui.QColor(t.get("accent", "#0a84ff"))

    def P(x, y):
        return QtCore.QPointF(cx + (x - 140) * s, cy + (y - 140) * s)

    def poly(pts):
        pg = QtGui.QPolygonF([P(*xy) for xy in pts])
        return pg

    # soft halo
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QtGui.QColor(accent.red(), accent.green(), accent.blue(), 26))
    p.drawEllipse(P(20, 24), 120 * s, 120 * s)

    # ears (outer + inner)
    p.setBrush(fur)
    p.drawPolygon(poly([(72, 96), (60, 28), (126, 78)]))
    p.drawPolygon(poly([(208, 96), (220, 28), (154, 78)]))
    p.setBrush(pink)
    p.drawPolygon(poly([(80, 90), (72, 46), (114, 78)]))
    p.drawPolygon(poly([(200, 90), (208, 46), (166, 78)]))

    # head
    p.setBrush(fur)
    p.drawEllipse(P(140, 152), 88 * s, 80 * s)

    # cheeks blush
    p.setBrush(QtGui.QColor(pink.red(), pink.green(), pink.blue(), 150))
    p.drawEllipse(P(96, 182), 16 * s, 11 * s)
    p.drawEllipse(P(184, 182), 16 * s, 11 * s)

    # happy closed eyes (^_^) as downward arcs
    pen = QtGui.QPen(dark, 6 * s)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for ex in (108, 172):
        rect = QtCore.QRectF(P(ex - 16, 150).x(), P(ex - 16, 150).y(),
                             32 * s, 22 * s)
        p.drawArc(rect, 200 * 16, 140 * 16)

    # nose
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(pink)
    p.drawPolygon(poly([(132, 168), (148, 168), (140, 178)]))

    # mouth (two little arcs)
    p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    for mx in (-1, 1):
        rect = QtCore.QRectF(P(140 + mx * 14 - 14, 178).x(),
                             P(140 + mx * 14 - 14, 178).y(), 28 * s, 18 * s)
        p.drawArc(rect, (200 if mx > 0 else 140) * 16, 100 * 16)

    # whiskers
    wpen = QtGui.QPen(dark, 3 * s); wpen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(wpen)
    for dy in (-10, 2, 14):
        p.drawLine(P(96, 176 + dy), P(40, 168 + dy * 1.4))
        p.drawLine(P(184, 176 + dy), P(240, 168 + dy * 1.4))
    p.end()
    return pm


class KittyDialog(QtWidgets.QDialog):
    """A picture of a cute kitty and a Meow button. Essential."""

    _MEOWS = ["Meow!", "Mrrrow~", "Mew :3", "MEOWWW", "prr\u2026 meow",
              "nyaa~", "mrrp!", "mreow?"]

    def __init__(self, parent, theme):
        super().__init__(parent)
        self.t = theme
        self._n = 0
        self._fx = None              # keep refs so players aren't GC'd
        self._player = None
        self._audio = None
        self._meow_files = self._materialize_meows()
        self.setWindowTitle("\U0001F431 Cute kitty")
        self.setMinimumWidth(360)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18); lay.setSpacing(10)

        pm = QtGui.QPixmap()
        if KITTY_IMAGE_B64.strip():
            try:
                import base64
                pm.loadFromData(base64.b64decode(KITTY_IMAGE_B64))
            except Exception:
                pm = QtGui.QPixmap()
        if pm.isNull() and KITTY_IMAGE_PATH.strip():
            pm = QtGui.QPixmap(KITTY_IMAGE_PATH)
        if pm.isNull():
            pm = kitty_pixmap(280, self.t)
        elif pm.width() > 360:        # keep a big photo from blowing up the dialog
            pm = pm.scaledToWidth(
                360, Qt.TransformationMode.SmoothTransformation)

        pic = QtWidgets.QLabel(); pic.setPixmap(pm)
        pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(pic)

        caption = QtWidgets.QLabel("KITTYYYYYYYYYYYYYYYYYYYYYYYYYYY")
        caption.setObjectName("muted")
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(caption)

        meow_btn = QtWidgets.QPushButton("  Meow")
        meow_btn.setIcon(QtGui.QIcon(kitty_pixmap(20, self.t)))
        meow_btn.setIconSize(QtCore.QSize(18, 18))
        meow_btn.setObjectName("primary")
        meow_btn.clicked.connect(self._meow)
        lay.addWidget(meow_btn)

        self.meow_lbl = QtWidgets.QLabel("")
        self.meow_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = self.meow_lbl.font(); f.setPointSize(15); f.setBold(True)
        self.meow_lbl.setFont(f)
        lay.addWidget(self.meow_lbl)

    def _meow(self):
        import random
        self._n += 1
        extra = "!" * min(self._n // 3, 6)
        self.meow_lbl.setText(random.choice(self._MEOWS) + extra)
        self._play_meow()

    # ---- sound: base64 override or in-memory synthesis ----
    def _synth_meow_wav(self):
        """Synthesise a short pitch-arched 'meow' and return WAV bytes."""
        import io, wave
        fs, dur = 22050, 0.55
        t = np.linspace(0.0, dur, int(fs * dur), endpoint=False)
        f0 = 430.0 + 230.0 * np.sin(np.pi * t / dur)     # me -> ow arch
        f0 += 12.0 * np.sin(2 * np.pi * 6.0 * t)          # vibrato
        phase = 2 * np.pi * np.cumsum(f0) / fs
        y = (np.sin(phase) + 0.5 * np.sin(2 * phase)
             + 0.25 * np.sin(3 * phase))
        env = np.clip(np.minimum(t / 0.04, (dur - t) / 0.18), 0.0, 1.0)
        y *= env
        y = y / (np.max(np.abs(y)) or 1.0)
        pcm = (y * 0.9 * 32767).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs)
            w.writeframes(pcm.tobytes())
        return buf.getvalue()

    @staticmethod
    def _sniff_audio_ext(data):
        """Guess a file extension from the first bytes of an audio blob."""
        if data[:4] == b"RIFF":
            return ".wav"
        if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF
                                  and (data[1] & 0xE0) == 0xE0):
            return ".mp3"
        return ".mp3"      # default; the players sniff the content anyway

    def _materialize_meows(self):
        """Build the list of playable meow files: MEOW_B64 entries decoded to
        temp files + MEOW_PATHS used as-is. Empty -> the meow is synthesised."""
        import base64
        import tempfile
        files, tmp = [], tempfile.gettempdir()
        for i, b in enumerate(MEOW_B64):
            b = (b or "").strip()
            if not b:
                continue
            try:
                blob = base64.b64decode(b)
            except Exception:
                continue
            p = os.path.join(tmp, f"_kitty_meow_{i}{self._sniff_audio_ext(blob)}")
            try:
                with open(p, "wb") as fh:
                    fh.write(blob)
                files.append(p)
            except Exception:
                pass
        for p in MEOW_PATHS:
            p = (p or "").strip()
            if p and os.path.isfile(p):
                files.append(p)
        return files

    def _play_meow(self):
        """Play a meow. A RANDOM one of your configured files if any, else
        the synthesised fallback."""
        import random
        if self._meow_files:
            if self._play_file(random.choice(self._meow_files)):
                return
        self._play_synth()

    def _play_file(self, path):
        """Play one audio file (.wav or .mp3). Returns True if a player
        started. WAV on Windows uses winsound (bulletproof); everything else
        prefers QMediaPlayer (MP3 + WAV via OS codecs), with MCI / CLI
        fallbacks so MP3 still plays if Qt multimedia is unavailable."""
        ext = os.path.splitext(path)[1].lower()
        win = sys.platform.startswith("win")
        # Windows + WAV: winsound first - synchronous load, no codec needed.
        if win and ext == ".wav":
            try:
                import winsound
                winsound.PlaySound(
                    path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return True
            except Exception:
                pass
        # QMediaPlayer: plays MP3 and WAV through the OS codecs.
        if self._qmedia_play(path):
            return True
        # Native fallbacks.
        if win:
            try:
                if self._mci_play(path):       # winmm/MCI - plays MP3 too
                    return True
            except Exception:
                pass
        else:
            import shutil
            import subprocess
            for spec in (("afplay",),
                         ("ffplay", "-nodisp", "-autoexit",
                          "-loglevel", "quiet"),
                         ("mpg123", "-q"), ("paplay",), ("aplay",)):
                exe = shutil.which(spec[0])
                if exe:
                    try:
                        subprocess.Popen([exe, *spec[1:], path],
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                        return True
                    except Exception:
                        pass
        return False

    def _qmedia_play(self, path):
        """Play via Qt's QMediaPlayer (handles MP3 + WAV). Imported
        dynamically so neither PyQt6 nor PySide6 is a static import."""
        try:
            import importlib
            _mm = importlib.import_module(QT_BINDING + ".QtMultimedia")
            _qc = importlib.import_module(QT_BINDING + ".QtCore")
            player = _mm.QMediaPlayer(self)
            audio = _mm.QAudioOutput(self)
            player.setAudioOutput(audio)
            audio.setVolume(0.9)
            player.setSource(_qc.QUrl.fromLocalFile(path))
            player.play()
            self._player = player          # keep refs alive past this call
            self._audio = audio
            return True
        except Exception:
            return False

    def _mci_play(self, path):
        """Windows MCI (winmm) - built in, plays MP3 and WAV. No deps."""
        import ctypes
        mci = ctypes.windll.winmm.mciSendStringW
        mci("close kittymeow", None, 0, None)          # drop any previous
        if mci(f'open "{path}" alias kittymeow', None, 0, None) != 0:
            return False
        mci("play kittymeow", None, 0, None)           # async (no 'wait')
        return True

    def _play_synth(self):
        """Synthesise a meow, write a temp WAV and play it (used only when no
        meow files are configured); bell as the final fallback."""
        try:
            data = self._synth_meow_wav()
        except Exception:
            data = b""
        if data:
            import tempfile
            p = os.path.join(tempfile.gettempdir(), "_kitty_meow_synth.wav")
            try:
                with open(p, "wb") as fh:
                    fh.write(data)
                if self._play_file(p):
                    return
            except Exception:
                pass
        try:
            QtWidgets.QApplication.beep()
        except Exception:
            pass


# ==========================================================================
# Main window
# ==========================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Planetary Gearbox FFT Analyser")
        self.resize(1320, 860)

        self.p = Params()
        self.dark = CONFIG["DARK_MODE"]
        self.spectra = []
        self.current = None
        self.mode = "single"
        self.dataset = None
        self.channels = {}
        self.cur_file = None
        self.cur_path = None
        self.gen_spec = None
        self.fs_detected = None
        self._views = []
        self._axes = []
        self._share_x = self._share_y = True
        self._has_time = False
        self._color_i = 0
        self._scrollable = False
        self._view_h = self._content_h = 1
        self._df = 0.01              # min frequency resolution (log-x floor)
        self._suppress = False
        self._manual_lines = []      # [(freq_hz, label)]
        self._manual_bands = []      # [(lo_hz, hi_hz, label)]
        self._pan_start = None       # (x_px, y_px) during a drag-pan
        self._pan_ax = None
        self._cbars = []             # spectrogram colorbars (cleared per render)

        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.canvas = MplCanvas(self.fig, self)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Expanding)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("resize_event", self._on_canvas_resize)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.scroll = PlotScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.canvas)
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._build_ui()
        self.apply_theme()
        self._render()
        QtCore.QTimer.singleShot(60, self._fit_canvas)

    # ----- small UI helpers -----
    def _card(self, title=None):
        card = QtWidgets.QFrame(); card.setObjectName("card")
        v = QtWidgets.QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 12); v.setSpacing(8)
        if title:
            lab = QtWidgets.QLabel(title); lab.setObjectName("section")
            v.addWidget(lab)
        return card, v

    def _row(self, label, widget):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        lab = QtWidgets.QLabel(label); lab.setMinimumWidth(96)
        lab.setObjectName("muted")
        h.addWidget(lab); h.addWidget(widget, 1)
        return w

    def _spin(self, lo: float, hi: float, val: float, step: float = 1,
              decimals: int | None = None, suffix: str = ""):
        if decimals is None:                       # integer spin box
            s = _NoScrollSpin()
            s.setRange(int(lo), int(hi)); s.setSingleStep(int(step))
            s.setValue(int(val))
        else:                                       # floating-point spin box
            s = _NoScrollDoubleSpin()
            s.setDecimals(decimals)
            s.setRange(float(lo), float(hi)); s.setSingleStep(float(step))
            s.setValue(float(val))
        s.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if suffix:
            s.setSuffix(suffix)
        return s

    def _combo(self, items, current):
        c = _NoScrollCombo(); c.addItems(items)
        c.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if current in items:
            c.setCurrentText(current)
        return c

    # ----- top-level layout -----
    def _build_ui(self):
        root = QtWidgets.QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        outer.addWidget(self._build_toolbar())
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_sidebar())
        rightwrap = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(rightwrap)
        rv.setContentsMargins(10, 10, 10, 0); rv.setSpacing(6)
        self.nav = NavigationToolbar2QT(self.canvas, self)
        try:
            # Reroute the toolbar's "Subplots" button to our layout dialog.
            self.nav.configure_subplots = lambda *a: self.open_layout()  # type: ignore[method-assign]
        except Exception:
            pass
        rv.addWidget(self.nav)
        rv.addWidget(self.scroll, 1)
        split.addWidget(rightwrap)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1)
        split.setSizes([360, 960])
        outer.addWidget(split, 1)
        outer.addWidget(self._build_statusbar())

    def _build_toolbar(self):
        bar = QtWidgets.QFrame(); bar.setObjectName("toolbar")
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(12, 8, 12, 8); h.setSpacing(8)
        title = QtWidgets.QLabel("FFT Analyser"); title.setObjectName("h1")
        h.addWidget(title); h.addSpacing(8)
        openb = QtWidgets.QPushButton("Open / add\u2026")
        openb.clicked.connect(self.open_files); h.addWidget(openb)
        self.btn_compute = QtWidgets.QPushButton("Compute FFT")
        self.btn_compute.setObjectName("primary")
        self.btn_compute.clicked.connect(self.compute); h.addWidget(self.btn_compute)
        self.btn_spec = QtWidgets.QPushButton("Spectrogram")
        self.btn_spec.clicked.connect(self.compute_spectrogram_cmd)
        h.addWidget(self.btn_spec)
        analyze = QtWidgets.QPushButton("Analyze")
        analyze.clicked.connect(self.analyze); h.addWidget(analyze)
        export = QtWidgets.QPushButton("Export\u2026")
        export.clicked.connect(self.export_any); h.addWidget(export)
        h.addStretch(1)
        seg = QtWidgets.QWidget(); seg.setObjectName("segwrap")
        sl = QtWidgets.QHBoxLayout(seg)
        sl.setContentsMargins(3, 3, 3, 3); sl.setSpacing(0)
        self.seg_btns = {}
        grp = QtWidgets.QButtonGroup(self); grp.setExclusive(True)
        for key, lab in (("single", "Single"), ("compare", "Compare"),
                         ("overlay", "Overlay")):
            b = QtWidgets.QPushButton(lab); b.setObjectName("seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _c=False, k=key: self.set_mode(k))
            grp.addButton(b); sl.addWidget(b); self.seg_btns[key] = b
        self.seg_btns["single"].setChecked(True)
        h.addWidget(seg); h.addSpacing(8)
        self.theme_btn = QtWidgets.QPushButton("\u263d")
        self.theme_btn.setObjectName("ghost")
        self.theme_btn.setToolTip("Toggle dark / light")
        self.theme_btn.clicked.connect(self.toggle_theme); h.addWidget(self.theme_btn)
        helpb = QtWidgets.QPushButton("?"); helpb.setObjectName("ghost")
        helpb.clicked.connect(self.show_help); h.addWidget(helpb)
        kitty = QtWidgets.QPushButton(); kitty.setObjectName("ghost")
        kitty.setIcon(QtGui.QIcon(kitty_pixmap(22, {})))
        kitty.setIconSize(QtCore.QSize(20, 20))
        kitty.setToolTip("Cute kitty (essential)")
        kitty.clicked.connect(self.show_kitty); h.addWidget(kitty)
        return bar

    def _build_sidebar(self):
        side = QtWidgets.QFrame(); side.setObjectName("sidebar")
        side.setMinimumWidth(330); side.setMaximumWidth(440)
        sc = QtWidgets.QScrollArea(); sc.setWidgetResizable(True)
        sc.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget(); sc.setWidget(inner)
        col = QtWidgets.QVBoxLayout(inner)
        col.setContentsMargins(12, 12, 12, 12); col.setSpacing(12)
        col.addWidget(self._build_terminal_card())
        col.addWidget(self._build_selected_card())
        col.addWidget(self._build_tabs_card())
        col.addWidget(self._build_speed_card())
        col.addStretch(1)
        wrap = QtWidgets.QVBoxLayout(side)
        wrap.setContentsMargins(0, 0, 0, 0); wrap.addWidget(sc)
        return side

    def _build_terminal_card(self):
        card, v = self._card("Graphs")
        self.listw = QtWidgets.QListWidget()
        self.listw.setMinimumHeight(150)
        self.listw.currentRowChanged.connect(self._on_select_row)
        v.addWidget(self.listw)
        r1 = QtWidgets.QHBoxLayout()
        up = QtWidgets.QPushButton("\u25b2"); up.setMaximumWidth(42)
        up.clicked.connect(lambda: self._move(-1))
        dn = QtWidgets.QPushButton("\u25bc"); dn.setMaximumWidth(42)
        dn.clicked.connect(lambda: self._move(1))
        srt = QtWidgets.QPushButton("Sort"); srt.clicked.connect(self.sort_stack)
        for b in (up, dn, srt):
            r1.addWidget(b)
        r2 = QtWidgets.QHBoxLayout()
        rm = QtWidgets.QPushButton("Remove"); rm.clicked.connect(self.remove_selected)
        cl = QtWidgets.QPushButton("Clear"); cl.clicked.connect(self.clear_stack)
        for b in (rm, cl):
            r2.addWidget(b)
        v.addLayout(r1); v.addLayout(r2)
        return card

    def _build_selected_card(self):
        card, v = self._card("Selected graph")
        self.channel_box = QtWidgets.QComboBox()
        self.channel_box.currentTextChanged.connect(self._on_channel_change)
        v.addWidget(self._row("Channel", self.channel_box))
        axlab = QtWidgets.QLabel("Axes \u2014 select one or more "
                                 "(Ctrl/Shift-click)")
        axlab.setObjectName("muted"); axlab.setWordWrap(True)
        v.addWidget(axlab)
        self.axis_list = QtWidgets.QListWidget()
        self.axis_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.axis_list.setMaximumHeight(118)
        v.addWidget(self.axis_list)
        self.fbg_box = self._combo(["microstrain", "wavelength", "temperature"],
                                   self.p.fbg_quantity)
        self.fbg_box.currentTextChanged.connect(
            lambda s: setattr(self.p, "fbg_quantity", s))
        v.addWidget(self._row("FBG qty", self.fbg_box))
        fbg_all = QtWidgets.QPushButton("Add all FBG nodes \u2192 stack")
        fbg_all.clicked.connect(self.add_all_fbg_nodes)
        v.addWidget(fbg_all)
        r1 = QtWidgets.QHBoxLayout()
        fb = QtWidgets.QPushButton("Focus"); fb.clicked.connect(self.focus_selected)
        cb = QtWidgets.QPushButton("Colour\u2026"); cb.clicked.connect(self.pick_colour)
        r1.addWidget(fb); r1.addWidget(cb)
        sh = QtWidgets.QPushButton("Show / Hide"); sh.clicked.connect(self.toggle_visible)
        v.addLayout(r1); v.addWidget(sh)
        hint = QtWidgets.QLabel("Compute FFT makes one graph per selected axis "
                                "(several \u2192 Overlay). Analyze acts on the "
                                "highlighted graph.")
        hint.setObjectName("muted"); hint.setWordWrap(True); v.addWidget(hint)
        return card

    def _build_tabs_card(self):
        card, v = self._card(None)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._tab_fft(), "FFT")
        tabs.addTab(self._tab_spectrogram(), "Spectrogram")
        tabs.addTab(self._tab_filter(), "Filter")
        tabs.addTab(self._tab_theory(), "Theory")
        v.addWidget(tabs)
        return card

    def _tab_fft(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(2, 8, 2, 2); v.setSpacing(7)
        self.window_box = self._combo(
            ["hann", "hamming", "blackman", "flattop", "boxcar"], self.p.window)
        self.spec_box = self._combo(["amplitude", "psd", "asd"], self.p.spec_type)
        self.detrend_chk = QtWidgets.QCheckBox("Detrend (remove DC)")
        self.detrend_chk.setChecked(self.p.detrend)
        self.welch_chk = QtWidgets.QCheckBox("Welch averaging")
        self.welch_chk.setChecked(self.p.use_welch)
        self.welch_seg = self._spin(2, 256, self.p.welch_segs)
        self.fmax_spin = self._spin(0, 200000, int(self.p.f_max), 50, suffix=" Hz")
        self.logx_chk = QtWidgets.QCheckBox("Log frequency (x)")
        self.logy_chk = QtWidgets.QCheckBox("Log amplitude (y)")
        self.avg_chk = QtWidgets.QCheckBox("Average line (overlay)")
        self.avg_chk.setChecked(self.p.show_avg)
        self.avg_chk.setToolTip("In Overlay mode, draw the mean of all visible "
                                "frequency spectra as one bold line.")
        self.avg_only_chk = QtWidgets.QCheckBox("Average only (hide spectra)")
        self.avg_only_chk.setChecked(self.p.avg_only)
        self.avg_only_chk.setToolTip("In Overlay mode, hide every individual "
                                     "spectrum and show only the average line.")
        for wdg in (self._row("Window", self.window_box),
                    self._row("Spectrum", self.spec_box), self.detrend_chk,
                    self.welch_chk, self._row("Segments", self.welch_seg),
                    self._row("Plot max", self.fmax_spin),
                    self.logx_chk, self.logy_chk, self.avg_chk,
                    self.avg_only_chk):
            v.addWidget(wdg)
        self.fmax_spin.valueChanged.connect(self._on_view_change)
        self.logx_chk.toggled.connect(self._on_view_change)
        self.logy_chk.toggled.connect(self._on_view_change)
        self.avg_chk.toggled.connect(self._on_view_change)
        self.avg_only_chk.toggled.connect(self._on_view_change)
        v.addStretch(1)
        return w

    def _tab_spectrogram(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(2, 8, 2, 2); v.setSpacing(7)

        def _hdr(text):
            lab = QtWidgets.QLabel(text); lab.setObjectName("section")
            return lab

        # --- generation ---
        v.addWidget(_hdr("Generation"))
        self.sg_window_box = self._combo(
            ["hann", "hamming", "blackman", "flattop", "boxcar"],
            self.p.sg_window)
        self.sg_nperseg_spin = self._spin(64, 65536, self.p.sg_nperseg,
                                          64, suffix=" pts")
        self.sg_overlap_spin = self._spin(0, 95, int(self.p.sg_overlap_pct),
                                          5, suffix=" %")
        self.sg_detrend_chk = QtWidgets.QCheckBox("Detrend (remove DC)")
        self.sg_detrend_chk.setChecked(self.p.sg_detrend)
        self.sg_db_chk = QtWidgets.QCheckBox("dB scale (10\u00b7log\u2081\u2080)")
        self.sg_db_chk.setChecked(self.p.sg_db)
        self.sg_cmap_box = self._combo(
            ["viridis", "magma", "inferno", "plasma", "turbo", "jet"],
            self.p.sg_cmap)
        self.sg_fmax_spin = self._spin(0, 200000, int(self.p.sg_fmax),
                                       50, suffix=" Hz")
        for wdg in (self._row("Window", self.sg_window_box),
                    self._row("Seg. length", self.sg_nperseg_spin),
                    self._row("Overlap", self.sg_overlap_spin),
                    self.sg_detrend_chk, self.sg_db_chk,
                    self._row("Colormap", self.sg_cmap_box),
                    self._row("Freq max", self.sg_fmax_spin)):
            v.addWidget(wdg)
        # dB / colormap / freq-max restyle live; the rest need a regenerate
        self.sg_db_chk.toggled.connect(self._on_view_change)
        self.sg_cmap_box.currentTextChanged.connect(self._on_view_change)
        self.sg_fmax_spin.valueChanged.connect(self._on_view_change)

        # --- independent filter (does NOT touch the FFT Filter tab) ---
        v.addWidget(_hdr("Filter (spectrogram only)"))
        self.sg_filt_chk = QtWidgets.QCheckBox("IIR filter (zero-phase)")
        self.sg_filt_chk.setChecked(self.p.sg_filt_on)
        self.sg_filt_type_box = self._combo(
            ["bandpass", "lowpass", "highpass", "bandstop"], self.p.sg_filt_type)
        self.sg_filt_fam_box = self._combo(
            ["butter", "cheby1", "cheby2", "bessel", "ellip"],
            self.p.sg_filt_family)
        self.sg_flo_spin = self._spin(0.0, 100000.0, self.p.sg_f_lo, 1.0, 2, " Hz")
        self.sg_fhi_spin = self._spin(0.0, 100000.0, self.p.sg_f_hi, 1.0, 2, " Hz")
        self.sg_ford_spin = self._spin(1, 12, self.p.sg_f_order)
        self.sg_frp_spin = self._spin(0.1, 10.0, self.p.sg_f_rp, 0.1, 1, " dB")
        self.sg_frs_spin = self._spin(10.0, 120.0, self.p.sg_f_rs, 1.0, 1, " dB")
        self.sg_notch_chk = QtWidgets.QCheckBox("Notch")
        self.sg_notch_chk.setChecked(self.p.sg_notch_on)
        self.sg_notch_hz_spin = self._spin(1.0, 100000.0, self.p.sg_notch_hz,
                                           1.0, 1, " Hz")
        self.sg_notch_q_spin = self._spin(1.0, 200.0, self.p.sg_notch_q, 1.0, 1)
        self.sg_hampel_chk = QtWidgets.QCheckBox("Hampel spike removal")
        self.sg_hampel_chk.setChecked(self.p.sg_hampel_on)
        self.sg_hampel_win_spin = self._spin(1, 64, self.p.sg_hampel_win)
        self.sg_hampel_sig_spin = self._spin(1.0, 10.0, self.p.sg_hampel_sigma,
                                             0.5, 1)
        for wdg in (self.sg_filt_chk, self._row("Type", self.sg_filt_type_box),
                    self._row("Family", self.sg_filt_fam_box),
                    self._row("Low", self.sg_flo_spin),
                    self._row("High", self.sg_fhi_spin),
                    self._row("Order", self.sg_ford_spin),
                    self._row("Ripple", self.sg_frp_spin),
                    self._row("Atten.", self.sg_frs_spin),
                    self.sg_notch_chk,
                    self._row("Notch f", self.sg_notch_hz_spin),
                    self._row("Notch Q", self.sg_notch_q_spin),
                    self.sg_hampel_chk,
                    self._row("Win", self.sg_hampel_win_spin),
                    self._row("Sigma", self.sg_hampel_sig_spin)):
            v.addWidget(wdg)

        # --- build / manage ---
        v.addWidget(_hdr("Build / manage"))
        gen_btn = QtWidgets.QPushButton("Generate spectrogram(s)")
        gen_btn.setObjectName("primary")
        gen_btn.clicked.connect(self.compute_spectrogram_cmd)
        gen_btn.setToolTip("One spectrogram per selected axis, using the "
                           "settings above (filter included).")
        v.addWidget(gen_btn)
        avg_btn = QtWidgets.QPushButton("Average spectrogram \u2192 stack")
        avg_btn.clicked.connect(self.add_average_spectrogram)
        avg_btn.setToolTip("Mean of all visible spectrograms (e.g. repeat runs "
                           "of one stage) as a new panel.")
        v.addWidget(avg_btn)
        hint = QtWidgets.QLabel(
            "Spectrograms join the graph list. Switch to Compare to stack them "
            "or Single to view one; Overlay isn't available for heatmaps. "
            "Window / length / overlap / filter need a regenerate; dB / "
            "colormap / freq-max restyle live.")
        hint.setObjectName("muted"); hint.setWordWrap(True)
        v.addWidget(hint)
        v.addStretch(1)
        return w

    def _tab_filter(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(2, 8, 2, 2); v.setSpacing(7)
        self.filt_chk = QtWidgets.QCheckBox("IIR filter (zero-phase)")
        self.filt_chk.setChecked(self.p.filt_on)
        self.filt_type_box = self._combo(
            ["bandpass", "lowpass", "highpass", "bandstop"], self.p.filt_type)
        self.filt_fam_box = self._combo(
            ["butter", "cheby1", "cheby2", "bessel", "ellip"], self.p.filt_family)
        self.flo_spin = self._spin(0.0, 100000.0, self.p.f_lo, 1.0, 2, " Hz")
        self.fhi_spin = self._spin(0.0, 100000.0, self.p.f_hi, 1.0, 2, " Hz")
        self.ford_spin = self._spin(1, 12, self.p.f_order)
        self.frp_spin = self._spin(0.1, 10.0, self.p.f_rp, 0.1, 1, " dB")
        self.frs_spin = self._spin(10.0, 120.0, self.p.f_rs, 1.0, 1, " dB")
        self.notch_chk = QtWidgets.QCheckBox("Notch")
        self.notch_chk.setChecked(self.p.notch_on)
        self.notch_hz_spin = self._spin(1.0, 100000.0, self.p.notch_hz, 1.0, 1, " Hz")
        self.notch_q_spin = self._spin(1.0, 200.0, self.p.notch_q, 1.0, 1)
        self.hampel_chk = QtWidgets.QCheckBox("Hampel spike removal")
        self.hampel_chk.setChecked(self.p.hampel_on)
        self.hampel_win_spin = self._spin(1, 64, self.p.hampel_win)
        self.hampel_sig_spin = self._spin(1.0, 10.0, self.p.hampel_sigma, 0.5, 1)
        for wdg in (self.filt_chk, self._row("Type", self.filt_type_box),
                    self._row("Family", self.filt_fam_box),
                    self._row("Low", self.flo_spin), self._row("High", self.fhi_spin),
                    self._row("Order", self.ford_spin),
                    self._row("Ripple", self.frp_spin),
                    self._row("Atten.", self.frs_spin),
                    self.notch_chk, self._row("Notch f", self.notch_hz_spin),
                    self._row("Notch Q", self.notch_q_spin),
                    self.hampel_chk, self._row("Win", self.hampel_win_spin),
                    self._row("Sigma", self.hampel_sig_spin)):
            v.addWidget(wdg)
        hint = QtWidgets.QLabel("Press Compute FFT to apply filter changes.")
        hint.setObjectName("muted"); hint.setWordWrap(True)
        v.addWidget(hint); v.addStretch(1)
        return w

    def _tab_theory(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(2, 8, 2, 2); v.setSpacing(7)
        self.zs_spin = self._spin(1, 400, self.p.zs)
        self.zr_spin = self._spin(1, 400, self.p.zr)
        self.zp_spin = self._spin(1, 400, self.p.zp)
        self.npl_spin = self._spin(1, 12, self.p.np_)
        self.nharm_spin = self._spin(1, 12, self.p.n_harm)
        self.nsb_spin = self._spin(0, 12, self.p.n_sb)
        self.sbsp_box = self._combo(["f_cpl", "f_pp", "f_c", "f_s"], self.p.sb_spacing)
        self.beta_spin = self._spin(0.0, 2.0, self.p.beta, 0.01, 2, " /mm")
        self.boa_spin = self._spin(0.0, 2.0, self.p.boa, 0.05, 2)
        self.torque_spin = self._spin(0.0, 100000.0, self.p.torque, 1.0, 1, " Nm")
        self.torque_ref_spin = self._spin(0.1, 100000.0, self.p.torque_ref,
                                          1.0, 1, " Nm")
        self.gamma_spin = self._spin(0.0, 20.0, self.p.gamma, 0.1, 2)
        self.show_theory_chk = QtWidgets.QCheckBox("Show expected-signal overlay")
        self.show_theory_chk.setChecked(self.p.show_theory)
        self.alpha_spin = self._spin(0.05, 1.0, self.p.theory_alpha, 0.05, 2)
        self.stage_spin = self._spin(0, 99, self.p.theory_stage)
        addth = QtWidgets.QPushButton("Add theory graph to stack")
        addth.clicked.connect(self.add_theory_spectrum)

        # focus tools
        self.gmf_k_spin = self._spin(1, 12, self.p.gmf_focus_k)
        focus_gmf_btn = QtWidgets.QPushButton("Focus GMF \u00d7k")
        focus_gmf_btn.clicked.connect(self.focus_gmf)
        self.focus_hz_spin = self._spin(0.0, 1_000_000.0,
                                        self.p.focus_custom_hz, 1.0, 1, " Hz")
        focus_line_btn = QtWidgets.QPushButton("Focus line")
        focus_line_btn.clicked.connect(self.focus_custom)

        # manual overlays
        self.mk_freq_spin = self._spin(0.0, 1_000_000.0, self.p.mk_freq,
                                       1.0, 1, " Hz")
        self.mk_label_edit = QtWidgets.QLineEdit()
        self.mk_label_edit.setPlaceholderText("label (optional)")
        add_line_btn = QtWidgets.QPushButton("Add line")
        add_line_btn.clicked.connect(self.add_manual_line)
        self.mk_lo_spin = self._spin(0.0, 1_000_000.0, self.p.mk_lo, 1.0, 1, " Hz")
        self.mk_hi_spin = self._spin(0.0, 1_000_000.0, self.p.mk_hi, 1.0, 1, " Hz")
        add_band_btn = QtWidgets.QPushButton("Add band")
        clear_mk_btn = QtWidgets.QPushButton("Clear manual")
        add_band_btn.clicked.connect(self.add_manual_band)
        clear_mk_btn.clicked.connect(self.clear_manual)
        mk_btn_row = QtWidgets.QHBoxLayout()
        mk_btn_row.addWidget(add_band_btn); mk_btn_row.addWidget(clear_mk_btn)
        mk_btn_wrap = QtWidgets.QWidget(); mk_btn_wrap.setLayout(mk_btn_row)

        def _hdr(text):
            lab = QtWidgets.QLabel(text); lab.setObjectName("muted")
            return lab

        for wdg in (self._row("Z sun", self.zs_spin), self._row("Z ring", self.zr_spin),
                    self._row("Z planet", self.zp_spin), self._row("N planets", self.npl_spin),
                    self._row("GMF harm.", self.nharm_spin), self._row("SB pairs", self.nsb_spin),
                    self._row("SB spacing", self.sbsp_box), self._row("\u03b2", self.beta_spin),
                    self._row("B/A", self.boa_spin),
                    _hdr("Load (Bartelmus susceptibility)"),
                    self._row("Motor torque", self.torque_spin),
                    self._row("Ref. torque", self.torque_ref_spin),
                    self._row("Suscept. \u03b3", self.gamma_spin),
                    self.show_theory_chk,
                    self._row("Overlay \u03b1", self.alpha_spin),
                    self._row("Stage H", self.stage_spin), addth,
                    _hdr("Focus"),
                    self._row("GMF \u00d7", self.gmf_k_spin), focus_gmf_btn,
                    self._row("Line Hz", self.focus_hz_spin), focus_line_btn,
                    _hdr("Manual overlays"),
                    self._row("Line Hz", self.mk_freq_spin),
                    self._row("Label", self.mk_label_edit), add_line_btn,
                    self._row("Band lo", self.mk_lo_spin),
                    self._row("Band hi", self.mk_hi_spin), mk_btn_wrap):
            v.addWidget(wdg)
        for s in (self.zs_spin, self.zr_spin, self.zp_spin, self.npl_spin,
                  self.nharm_spin, self.nsb_spin):
            s.valueChanged.connect(self._on_marker_change)
        self.sbsp_box.currentTextChanged.connect(self._on_marker_change)
        self.beta_spin.valueChanged.connect(self._on_marker_change)
        self.boa_spin.valueChanged.connect(self._on_marker_change)
        self.torque_spin.valueChanged.connect(self._on_marker_change)
        self.torque_ref_spin.valueChanged.connect(self._on_marker_change)
        self.gamma_spin.valueChanged.connect(self._on_marker_change)
        self.show_theory_chk.toggled.connect(self._on_marker_change)
        self.alpha_spin.valueChanged.connect(self._on_alpha_change)
        v.addStretch(1)
        return w

    def _build_speed_card(self):
        card, v = self._card("Speed / acquisition")
        self.rpm_spin = self._spin(1.0, 100000.0, self.p.rpm, 10.0, 1, " rpm")
        self.rpm_spin.valueChanged.connect(self._on_marker_change)
        meas = QtWidgets.QPushButton("Use measured average RPM")
        meas.clicked.connect(self.use_measured_rpm)
        self.fsover_chk = QtWidgets.QCheckBox("Override sample rate")
        self.fsover_chk.setChecked(self.p.fs_override_on)
        self.fsover_spin = self._spin(1.0, 200000.0, self.p.fs_override_hz, 10.0, 1, " Hz")
        v.addWidget(self._row("Input RPM", self.rpm_spin)); v.addWidget(meas)
        v.addWidget(self.fsover_chk); v.addWidget(self._row("fs", self.fsover_spin))
        gen = QtWidgets.QLabel("Generic CSV"); gen.setObjectName("section"); v.addWidget(gen)
        self.gen_cols = QtWidgets.QLineEdit()
        self.gen_cols.setPlaceholderText("value columns, comma-separated")
        self.gen_time = QtWidgets.QLineEdit()
        self.gen_time.setPlaceholderText("time column (optional, seconds)")
        self.gen_delim = self._combo(["auto", ";", ",", "tab"], "auto")
        self.gen_fs = self._spin(1.0, 200000.0, 1000.0, 10.0, 1, " Hz")
        openg = QtWidgets.QPushButton("Open generic CSV\u2026")
        openg.clicked.connect(self.open_generic_csv)
        v.addWidget(self.gen_cols); v.addWidget(self.gen_time)
        v.addWidget(self._row("Delimiter", self.gen_delim))
        v.addWidget(self._row("fs (no time)", self.gen_fs)); v.addWidget(openg)
        return card

    def _build_statusbar(self):
        bar = QtWidgets.QFrame(); bar.setObjectName("statusbar")
        h = QtWidgets.QHBoxLayout(bar); h.setContentsMargins(14, 6, 14, 6)
        self.status = QtWidgets.QLabel("Open a recording to begin.")
        self.fs_info = QtWidgets.QLabel(""); self.fs_info.setObjectName("muted")
        h.addWidget(self.status); h.addStretch(1); h.addWidget(self.fs_info)
        return bar

    def open_layout(self):
        LayoutDialog(self).show()

    # ----- parameters -----
    def _collect_params(self):
        p = self.p
        p.window = self.window_box.currentText()
        p.spec_type = self.spec_box.currentText()
        p.detrend = self.detrend_chk.isChecked()
        p.use_welch = self.welch_chk.isChecked()
        p.welch_segs = self.welch_seg.value()
        p.f_max = float(self.fmax_spin.value())
        p.log_x = self.logx_chk.isChecked()
        p.log_y = self.logy_chk.isChecked()
        p.show_avg = self.avg_chk.isChecked()
        p.avg_only = self.avg_only_chk.isChecked()
        p.sg_window = self.sg_window_box.currentText()
        p.sg_detrend = self.sg_detrend_chk.isChecked()
        p.sg_nperseg = int(self.sg_nperseg_spin.value())
        p.sg_overlap_pct = float(self.sg_overlap_spin.value())
        p.sg_db = self.sg_db_chk.isChecked()
        p.sg_cmap = self.sg_cmap_box.currentText()
        p.sg_fmax = float(self.sg_fmax_spin.value())
        p.sg_filt_on = self.sg_filt_chk.isChecked()
        p.sg_filt_type = self.sg_filt_type_box.currentText()
        p.sg_filt_family = self.sg_filt_fam_box.currentText()
        p.sg_f_lo = self.sg_flo_spin.value(); p.sg_f_hi = self.sg_fhi_spin.value()
        p.sg_f_order = self.sg_ford_spin.value()
        p.sg_f_rp = self.sg_frp_spin.value(); p.sg_f_rs = self.sg_frs_spin.value()
        p.sg_notch_on = self.sg_notch_chk.isChecked()
        p.sg_notch_hz = self.sg_notch_hz_spin.value()
        p.sg_notch_q = self.sg_notch_q_spin.value()
        p.sg_hampel_on = self.sg_hampel_chk.isChecked()
        p.sg_hampel_win = self.sg_hampel_win_spin.value()
        p.sg_hampel_sigma = self.sg_hampel_sig_spin.value()
        p.filt_on = self.filt_chk.isChecked()
        p.filt_type = self.filt_type_box.currentText()
        p.filt_family = self.filt_fam_box.currentText()
        p.f_lo = self.flo_spin.value(); p.f_hi = self.fhi_spin.value()
        p.f_order = self.ford_spin.value()
        p.f_rp = self.frp_spin.value(); p.f_rs = self.frs_spin.value()
        p.notch_on = self.notch_chk.isChecked()
        p.notch_hz = self.notch_hz_spin.value(); p.notch_q = self.notch_q_spin.value()
        p.hampel_on = self.hampel_chk.isChecked()
        p.hampel_win = self.hampel_win_spin.value()
        p.hampel_sigma = self.hampel_sig_spin.value()
        p.fs_override_on = self.fsover_chk.isChecked()
        p.fs_override_hz = self.fsover_spin.value()
        p.fbg_quantity = self.fbg_box.currentText()
        p.rpm = self.rpm_spin.value()
        p.zs = self.zs_spin.value(); p.zr = self.zr_spin.value()
        p.zp = self.zp_spin.value(); p.np_ = self.npl_spin.value()
        p.n_harm = self.nharm_spin.value(); p.n_sb = self.nsb_spin.value()
        p.sb_spacing = self.sbsp_box.currentText()
        p.beta = self.beta_spin.value(); p.boa = self.boa_spin.value()
        p.torque = self.torque_spin.value()
        p.torque_ref = self.torque_ref_spin.value()
        p.gamma = self.gamma_spin.value()
        p.show_theory = self.show_theory_chk.isChecked()
        p.theory_alpha = self.alpha_spin.value()
        p.theory_stage = self.stage_spin.value()
        p.gmf_focus_k = self.gmf_k_spin.value()
        p.focus_custom_hz = float(self.focus_hz_spin.value())
        p.mk_freq = float(self.mk_freq_spin.value())
        p.mk_label = self.mk_label_edit.text()
        p.mk_lo = float(self.mk_lo_spin.value())
        p.mk_hi = float(self.mk_hi_spin.value())
        return p

    # ----- file / data -----
    def open_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open / add recordings", "",
            "Recordings (*.IDE *.ide *.csv *.CSV);;All files (*)")
        if not paths:
            return
        self._collect_params()
        added, errors = 0, []
        capped = False
        for k, path in enumerate(paths, 1):
            if len(self.spectra) >= MAX_STACK:
                capped = True
                break
            name = os.path.basename(path)
            self.status.setText(f"Loading {k}/{len(paths)}: {name}\u2026")
            QtWidgets.QApplication.processEvents()
            try:
                handle, chans = open_path(path)
                ch = next((l for l in chans if "40g" in l), next(iter(chans)))
                subs = chans[ch][1]
                axis = next((s for s in subs
                             if s.strip().upper().startswith("Z")), subs[0])
                x, fs_det, unit, domain = extract_named(handle, chans, ch, axis, self.p)
                close_handle(handle)
                fs = get_fs(fs_det, self.p)
                src = {"path": path, "channel": ch, "axis": axis}
                base = f"{name} \u00b7 {ch} \u00b7 {axis}"
                if domain == "time":
                    spec = make_trend_spec(x, fs, base, unit, src)
                else:
                    spec, _ = process_signal(x, fs, base, unit, self.p)
                    spec["src"] = src
                spec["fs_detected"] = fs_det
                self.spectra.append(spec); added += 1
            except Exception as e:
                errors.append(f"{name}: {e}")
        if errors:
            QtWidgets.QMessageBox.warning(self, "Some files failed", "\n".join(errors))
        if capped:
            QtWidgets.QMessageBox.information(
                self, "Stack limit reached",
                f"Stopped at {MAX_STACK} graphs (the comparison limit). "
                "Remove some, or use overlay mode to view many at once.")
        if not self.spectra:
            return
        self._refresh_list()
        if len(self.spectra) >= 2 and self.mode == "single":
            self.mode = "compare"; self._sync_seg()
        self._select_row(len(self.spectra) - 1)
        gc.collect()
        self.status.setText(f"Loaded {added} file(s). "
                            f"Terminal: {len(self.spectra)} graph(s).")

    def open_generic_csv(self):
        cols = [c.strip() for c in self.gen_cols.text().split(",") if c.strip()]
        if not cols:
            QtWidgets.QMessageBox.information(
                self, "Columns needed",
                "Enter one or more value column names (comma-separated).")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open generic CSV", "", "CSV (*.csv *.CSV);;All files (*)")
        if not path:
            return
        self._collect_params()
        spec = {"value_cols": cols, "time_col": self.gen_time.text(),
                "delim": self.gen_delim.currentText(),
                "fs": float(self.gen_fs.value())}
        try:
            handle, chans = open_generic(path, spec)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return
        close_handle(self.dataset)
        self.dataset, self.channels = handle, chans
        self.cur_file = os.path.basename(path); self.cur_path = path
        self.gen_spec = spec
        self._fill_channels(chans, default=next(iter(chans)))
        self.compute()

    def _fill_channels(self, chans, default=None, axis=None):
        self._suppress = True
        self.channel_box.clear(); self.channel_box.addItems(list(chans))
        if default and default in chans:
            self.channel_box.setCurrentText(default)
        ch = self.channel_box.currentText()
        self._suppress = False
        subs = chans[ch][1] if ch in chans else []
        self._set_axes(subs, selected=[axis] if axis else None)

    def _set_axes(self, subs, selected=None):
        """Populate the multi-select axis list. If no explicit selection is
        given, default to a Z* axis when present (matching the open default),
        else the first axis."""
        self._suppress = True
        self.axis_list.clear()
        self.axis_list.addItems(list(subs))
        sel = set(selected or [])
        if not sel and subs:
            sel = {next((s for s in subs
                         if s.strip().upper().startswith("Z")), subs[0])}
        for i in range(self.axis_list.count()):
            it = self.axis_list.item(i)
            if it.text() in sel:
                it.setSelected(True)
        self._suppress = False

    def _selected_axes(self):
        """Selected axis names, in list (display) order for deterministic
        X, Y, Z ordering."""
        return [self.axis_list.item(i).text()
                for i in range(self.axis_list.count())
                if self.axis_list.item(i).isSelected()]

    def _on_channel_change(self, *_):
        if self._suppress:
            return
        entry = self.channels.get(self.channel_box.currentText()) \
            if self.channels else None
        self._set_axes(entry[1] if entry else [])

    def _on_select_row(self, i):
        if self._suppress or i < 0 or i >= len(self.spectra):
            return
        self._select_row(i, from_user=True)

    def _select_row(self, i, from_user=False):
        if i < 0 or i >= len(self.spectra):
            return
        sp = self.spectra[i]
        self.current = sp
        if not from_user:
            self._suppress = True
            self.listw.setCurrentRow(i)
            self._suppress = False
        src = sp.get("src") or {}
        path = src.get("path")
        if path:
            try:
                if src.get("loader") == "gen":
                    handle, chans = open_generic(path, src.get("spec") or {})
                else:
                    handle, chans = open_path(path)
                close_handle(self.dataset)
                self.dataset, self.channels = handle, chans
                self.cur_file = os.path.basename(path); self.cur_path = path
                self._fill_channels(chans, default=src.get("channel"),
                                    axis=src.get("axis"))
                self.fs_detected = sp.get("fs_detected", sp.get("fs"))
            except Exception as e:
                self.status.setText(f"Could not reopen {path}: {e}")
        if self.mode == "single":
            self._render()
        self.status.setText(f"Selected: {sp['label']}")

    # ----- compute (one graph per selected axis; rest kept consistent) -----
    def compute(self):
        if self.dataset is None:
            QtWidgets.QMessageBox.information(self, "No file", "Open a file first.")
            return
        self._collect_params()
        ch = self.channel_box.currentText()
        axes = self._selected_axes()
        if not axes:
            QtWidgets.QMessageBox.information(
                self, "No axis selected",
                "Select at least one axis to plot "
                "(Ctrl/Shift-click for several).")
            return
        is_gen = isinstance(self.dataset, tuple) and self.dataset[0] == "gen"
        placed, errors, capped = [], [], False
        for axis in axes:
            try:
                x, fs_det, unit, domain = extract_named(
                    self.dataset, self.channels, ch, axis, self.p)
            except Exception as e:
                errors.append(f"{axis}: {e}"); continue
            self.fs_detected = fs_det
            fs = get_fs(fs_det, self.p)
            src = {"path": self.cur_path, "channel": ch, "axis": axis}
            if is_gen:
                src["loader"] = "gen"; src["spec"] = self.gen_spec
            base = f"{self.cur_file} \u00b7 {ch} \u00b7 {axis}"
            if domain == "time":
                spec = make_trend_spec(x, fs, base, unit, src)
            else:
                try:
                    spec, _ = process_signal(x, fs, base, unit, self.p)
                except Exception as e:
                    errors.append(f"{axis}: {e}"); continue
                spec["src"] = src
            spec["fs_detected"] = fs_det
            idx = self._place_spectrum(spec)
            if idx is None:
                capped = True; break
            placed.append(idx)
        if self.fs_detected is not None:
            fs = get_fs(self.fs_detected, self.p)
            self.fs_info.setText(
                f"fs detected {self.fs_detected:.2f} Hz \u00b7 used {fs:.2f} Hz")
        if placed:
            self.current = self.spectra[placed[-1]]
        # keep every other graph on the current settings too
        self._recompute_indices_except(set(placed))
        self._refresh_list()
        if placed:
            self._select_in_list(placed[-1])
        # several axes at once -> overlay them on one plot
        if len(placed) > 1 and self.mode == "single":
            self.mode = "overlay"; self._sync_seg()
        self._render()
        if errors:
            QtWidgets.QMessageBox.warning(self, "Some axes failed",
                                          "\n".join(errors))
        if capped:
            QtWidgets.QMessageBox.information(
                self, "Stack limit reached",
                f"Stopped at {MAX_STACK} graphs (the comparison limit).")
        self.status.setText(
            f"Computed {len(placed)} axis spectrum/spectra; "
            f"stack: {len(self.spectra)} graph(s).")

    def _place_spectrum(self, spec):
        """Insert spec into the stack, replacing an existing graph with the
        same (path, channel, axis) if present (inheriting its style), else
        appending. Returns the index, or None if the stack is full."""
        src = spec.get("src") or {}
        key = (src.get("path"), src.get("channel"), src.get("axis"))
        if key[0] is not None:
            for k, sp in enumerate(self.spectra):
                s = sp.get("src") or {}
                if (s.get("path"), s.get("channel"), s.get("axis")) == key:
                    self._style(spec, inherit=sp)
                    self.spectra[k] = spec
                    return k
        if len(self.spectra) >= MAX_STACK:
            return None
        self._style(spec)
        self.spectra.append(spec)
        return len(self.spectra) - 1

    def _recompute_indices_except(self, keep):
        """Recompute every graph whose index is not in `keep`, from its stored
        source, with the current settings."""
        errs = []
        for k in range(len(self.spectra)):
            if k in keep:
                continue
            e = self._recompute_spec_at(k)
            if e:
                errs.append(e)
        if errs:
            QtWidgets.QMessageBox.warning(self, "Some graphs failed",
                                          "\n".join(errs))

    def _store_spectrum(self, spec):
        i = self.listw.currentRow()
        new_path = (spec.get("src") or {}).get("path")
        same = (0 <= i < len(self.spectra) and new_path is not None
                and (self.spectra[i].get("src") or {}).get("path") == new_path)
        if same:
            self._style(spec, inherit=self.spectra[i]); self.spectra[i] = spec
            sel = i
        else:
            if len(self.spectra) >= MAX_STACK:
                QtWidgets.QMessageBox.information(
                    self, "Stack limit reached",
                    f"The comparison stack is full ({MAX_STACK}). Remove a "
                    "graph first, or use overlay mode.")
                self.current = spec
                self.mode = "single"; self._sync_seg(); self._render()
                return
            self._style(spec); self.spectra.append(spec)
            sel = len(self.spectra) - 1
        self.current = spec
        self._refresh_list(); self._select_in_list(sel)
        if self.mode not in ("compare", "overlay"):
            self.mode = "single"; self._sync_seg()
        self._render()

    def add_all_fbg_nodes(self):
        """Add a spectrum for every node of the open FBG file to the stack,
        using the current FBG-quantity and FFT/filter settings."""
        if (not isinstance(self.dataset, tuple)
                or self.dataset[0] != "fbg"):
            QtWidgets.QMessageBox.information(
                self, "No FBG file", "Open a PhotonFirst FBG .csv first.")
            return
        self._collect_params()
        nodes = sorted(self.dataset[1])
        added, errors = 0, []
        capped = False
        for k, node in enumerate(nodes, 1):
            if len(self.spectra) >= MAX_STACK:
                capped = True
                break
            self.status.setText(f"FBG node {k}/{len(nodes)}: {node}\u2026")
            QtWidgets.QApplication.processEvents()
            try:
                x, fs_det, unit, domain = extract_named(
                    self.dataset, self.channels, "[FBG] PhotonFirst",
                    node, self.p)
                fs = get_fs(fs_det, self.p)
                src = {"path": self.cur_path,
                       "channel": "[FBG] PhotonFirst", "axis": node}
                base = f"{self.cur_file} \u00b7 {node}"
                if domain == "time":
                    spec = make_trend_spec(x, fs, base, unit, src)
                else:
                    spec, _ = process_signal(x, fs, base, unit, self.p)
                    spec["src"] = src
                spec["fs_detected"] = fs_det
                self._style(spec)
                self.spectra.append(spec)
                added += 1
            except Exception as e:
                errors.append(f"{node}: {e}")
        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Some nodes failed", "\n".join(errors))
        if capped:
            QtWidgets.QMessageBox.information(
                self, "Stack limit reached",
                f"Stopped at {MAX_STACK} graphs (the comparison limit).")
        if self.spectra:
            self.current = self.spectra[-1]
            self.mode = "compare" if len(self.spectra) >= 2 else "single"
            self._sync_seg(); self._refresh_list()
            self._select_in_list(len(self.spectra) - 1); self._render()
        gc.collect()
        self.status.setText(
            f"Added {added} FBG node(s). Stack: {len(self.spectra)}.")

    def _recompute_spec_at(self, k):
        """Reopen graph k's source and recompute it with the current
        settings (used by Compute to keep the whole stack consistent).
        Returns None on success, or an error string."""
        sp = self.spectra[k]
        src = sp.get("src")
        if not src:
            return None
        name = os.path.basename(src["path"])
        try:
            if src.get("loader") == "gen":
                handle, chans = open_generic(src["path"], src["spec"])
            else:
                handle, chans = open_path(src["path"])
            ch = src["channel"] if src["channel"] in chans else next(iter(chans))
            subs = chans[ch][1]
            axis = src["axis"] if src["axis"] in subs else subs[0]
            x, fs_det, unit, domain = extract_named(handle, chans, ch, axis, self.p)
            close_handle(handle)
            fs = get_fs(fs_det, self.p)
            base = f"{name} \u00b7 {ch} \u00b7 {axis}"
            if sp.get("domain") == "spectrogram":
                new = self._make_spectrogram(x, fs, base, unit, src)
            elif domain == "time":
                new = make_trend_spec(x, fs, base, unit, src)
            else:
                new, _ = process_signal(x, fs, base, unit, self.p)
                new["src"] = src
            new["fs_detected"] = fs_det
            self._style(new, inherit=self.spectra[k]); self.spectra[k] = new
            return None
        except Exception as e:
            return f"{name}: {e}"

    def _recompute_others(self):
        sel = self.listw.currentRow()
        errs = []
        for k in range(len(self.spectra)):
            if k == sel:
                continue
            e = self._recompute_spec_at(k)
            if e:
                errs.append(e)
        if len(self.spectra) > 1:
            self._refresh_list(); self._select_in_list(sel); self._render()
        if errs:
            QtWidgets.QMessageBox.warning(self, "Some graphs failed", "\n".join(errs))

    # ----- list ops + per-graph style -----
    def _style(self, spec, inherit=None):
        if inherit is not None:
            spec["color"] = inherit.get("color") or self._next_color()
            spec["alpha"] = inherit.get("alpha", 1.0)
            spec["visible"] = inherit.get("visible", True)
        else:
            if not spec.get("color"):
                spec["color"] = self._next_color()
            spec.setdefault("alpha", 1.0)
            spec.setdefault("visible", True)

    def _next_color(self):
        c = GRAPH_PALETTE[self._color_i % len(GRAPH_PALETTE)]
        self._color_i += 1
        return c

    def _refresh_list(self):
        self._suppress = True
        cur = self.listw.currentRow()
        self.listw.clear()
        for i, sp in enumerate(self.spectra, 1):
            self._style(sp)
            prefix = "" if sp.get("visible", True) else "(off) "
            it = QtWidgets.QListWidgetItem(f"{i}.  {prefix}{sp['label']}")
            it.setForeground(QtGui.QColor(sp.get("color", "#888888")))
            self.listw.addItem(it)
        if 0 <= cur < self.listw.count():
            self.listw.setCurrentRow(cur)
        self._suppress = False

    def _select_in_list(self, i):
        if 0 <= i < self.listw.count():
            self._suppress = True
            self.listw.setCurrentRow(i)
            self._suppress = False

    def _move(self, delta):
        i = self.listw.currentRow(); j = i + delta
        if not (0 <= i < len(self.spectra) and 0 <= j < len(self.spectra)):
            return
        self.spectra[i], self.spectra[j] = self.spectra[j], self.spectra[i]
        self._refresh_list(); self._select_in_list(j); self.rerender_keep_zoom()

    def sort_stack(self):
        def key(sp):
            return [int(t) if t.isdigit() else t.lower()
                    for t in re.split(r"(\d+)", sp["label"])]
        self.spectra.sort(key=key)
        self._refresh_list(); self.rerender_keep_zoom()

    def remove_selected(self):
        i = self.listw.currentRow()
        if not (0 <= i < len(self.spectra)):
            return
        del self.spectra[i]
        self._refresh_list()
        if self.spectra:
            self._select_row(min(i, len(self.spectra) - 1))
        else:
            self.current = None; self._render()

    def clear_stack(self):
        self.spectra.clear(); self.current = None
        self._refresh_list(); self._render()

    def focus_selected(self):
        i = self.listw.currentRow()
        if 0 <= i < len(self.spectra):
            self.current = self.spectra[i]
            self.set_mode("single")

    def toggle_visible(self):
        i = self.listw.currentRow()
        if not (0 <= i < len(self.spectra)):
            return
        sp = self.spectra[i]
        sp["visible"] = not sp.get("visible", True)
        self._refresh_list(); self._select_in_list(i); self._render()
        self.status.setText(f"{'Shown' if sp['visible'] else 'Hidden'}: {sp['label']}")

    def pick_colour(self):
        i = self.listw.currentRow()
        if not (0 <= i < len(self.spectra)):
            return
        sp = self.spectra[i]; self._style(sp)
        init = QtGui.QColor(sp.get("color", "#0a84ff"))
        init.setAlphaF(float(sp.get("alpha", 1.0)))
        col = QtWidgets.QColorDialog.getColor(
            init, self, "Graph colour",
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if not col.isValid():
            return
        sp["color"] = col.name(); sp["alpha"] = col.alphaF()
        self._refresh_list(); self._select_in_list(i); self._render()
        self.status.setText(f"Colour {col.name()} (\u03b1 {col.alphaF():.2f}): "
                            f"{sp['label']}")

    # ----- mode / view -----
    def set_mode(self, mode):
        if mode == "compare" and len(self.spectra) < 2:
            QtWidgets.QMessageBox.information(self, "Need 2 graphs",
                                              "Add at least two graphs to compare.")
            self._sync_seg(); return
        if mode == "overlay" and not self.spectra:
            QtWidgets.QMessageBox.information(self, "Empty", "Add graphs to overlay.")
            self._sync_seg(); return
        self.mode = mode
        self._sync_seg(); self._render()

    def _sync_seg(self):
        for k, b in self.seg_btns.items():
            b.setChecked(k == self.mode)

    def _on_marker_change(self, *_):
        self._collect_params(); self.refresh_markers()

    def _on_view_change(self, *_):
        self._collect_params(); self._render()

    def _on_alpha_change(self, *_):
        self._collect_params()
        for v in self._views:
            for lc in v.get("theory") or []:
                lc.set_alpha(self.p.theory_alpha)
        self.canvas.draw_idle()

    def use_measured_rpm(self):
        self._collect_params()
        specs = [s for s in (self.spectra or
                             ([self.current] if self.current else []))
                 if s and s.get("domain") == "freq"]
        if not specs:
            QtWidgets.QMessageBox.information(self, "No spectra",
                                              "Compute a frequency spectrum first.")
            return
        gf, _ = marker_params(self.p)
        rpms = []
        for sp in specs:
            _, fm_meas, _ = refine_gmf(sp, gf)
            r, _ = infer_input_rpm(fm_meas, self.p.zs, self.p.zr)
            rpms.append(r)
        avg = float(np.mean(rpms)); manual = self.p.rpm
        err = (avg - manual) / manual * 100.0 if manual else float("nan")
        self.rpm_spin.setValue(round(avg, 1))
        self.status.setText(f"Input RPM set to measured {avg:.1f} rpm over "
                            f"{len(rpms)} graph(s); \u0394 {err:+.2f}%.")

    def add_theory_spectrum(self):
        self._collect_params()
        if len(self.spectra) >= MAX_STACK:
            QtWidgets.QMessageBox.information(
                self, "Stack limit reached",
                f"The comparison stack is full ({MAX_STACK}). Remove a graph "
                "first, or use overlay mode.")
            return
        stage = self.p.theory_stage
        mod = stage_modulation(stage, self.p.beta, self.p.boa,
                               torque=self.p.torque,
                               torque_ref=self.p.torque_ref,
                               gamma=self.p.gamma)
        if mod is None:
            QtWidgets.QMessageBox.information(
                self, "Unknown stage",
                f"Stage H{stage} not in STAGE_CUT_PCT {sorted(CONFIG['STAGE_CUT_PCT'])}.")
            return
        _, A, B = mod
        gf, _ = marker_params(self.p)
        n_harm, n_sb = max(1, self.p.n_harm), self.p.n_sb
        fmax = self.p.f_max or (n_harm + 1) * gf["f_mesh"]
        if fmax <= 0:
            fmax = (n_harm + 1) * gf["f_mesh"]
        df = max(0.1, fmax / 8000.0)
        freq = np.arange(0.0, fmax, df)
        amp = np.zeros_like(freq)
        C = theory_line_spectrum(A, B, n_sb)
        c0 = abs(C.get(0, 1.0)) or 1.0
        for k in range(1, n_harm + 1):
            fh = k * gf["f_mesh"]
            if fh > fmax:
                break
            for j in range(-n_sb, n_sb + 1):
                fpos = fh + j * gf["f_cpl"]
                if 0.0 <= fpos < fmax:
                    bi = int(round(fpos / df))
                    if 0 <= bi < len(amp):
                        amp[bi] = max(amp[bi], abs(C[j]) / c0)
        posv = amp[amp > 0]
        # tag the torque into the label so correlation_report can read it back
        tag = f" {self.p.torque:g}Nm" if (self.p.gamma and self.p.torque) else ""
        spec = {"label": f"THEORY H{stage}{tag} (model)",
                "f": _store(freq), "amp": _store(amp),
                "ytype": "amplitude", "unit": "rel", "domain": "freq",
                "df": df, "floor": float(posv.min()) if posv.size else 1e-9,
                "fs": 2.0 * fmax, "n": len(freq), "src": None}
        self._style(spec); self.spectra.append(spec)
        self._refresh_list()
        if self.mode == "single" and len(self.spectra) >= 2:
            self.mode = "compare"; self._sync_seg()
        self._select_in_list(len(self.spectra) - 1); self._render()
        self.status.setText(f"Added theoretical spectrum H{stage} "
                            f"(A={A:.3f}, B={B:.3f}).")

    # ----- analyze / export -----
    def analyze(self):
        self._collect_params()
        i = self.listw.currentRow()
        sp = self.spectra[i] if 0 <= i < len(self.spectra) else self.current
        if sp is None:
            QtWidgets.QMessageBox.information(
                self, "No data", "Open / compute a graph, then select it.")
            return
        text = acquisition_report(sp, self.spectra, self.p)
        dlg = TextDialog(self, "Analysis", text); dlg.show()
        self._dialogs = getattr(self, "_dialogs", [])
        self._dialogs.append(dlg)
        self.status.setText("Analysis report opened.")

    def export_any(self):
        if not self.spectra and self.current is None:
            QtWidgets.QMessageBox.information(self, "Nothing to export",
                                              "Open or compute a graph first.")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Export")
        box.setText("What would you like to export?")
        fig_b = box.addButton("Figure (PNG/SVG/PDF)\u2026",
                              QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        csv_b = box.addButton("Data (CSV)\u2026",
                              QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is fig_b:
            self.export_figure()
        elif clicked is csv_b:
            self.export_csv()

    def export_figure(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export figure", "spectrum.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200, facecolor=self.fig.get_facecolor())
            self.status.setText(f"Saved figure: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", str(e))

    def export_csv(self):
        specs = [s for s in (self.spectra or
                             ([self.current] if self.current else []))
                 if s and s.get("domain") != "spectrogram"]
        if not specs:
            QtWidgets.QMessageBox.information(
                self, "Nothing to export",
                "CSV export covers line spectra and trends. Spectrograms are "
                "2-D \u2014 use Export \u2192 Figure for those.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export data (CSV)", "spectra.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            amps = [np.asarray(sp["amp"]) for sp in specs]
            lens = [len(a) for a in amps]
            n = max(lens) if lens else 0
            is_time = [sp.get("domain") == "time" for sp in specs]
            dfs = [float(sp["df"]) for sp in specs]
            fss = [float(sp["fs"]) for sp in specs]
            headers = []
            for sp in specs:
                lab = sp["label"].replace(",", " ")
                xname = "time_s" if sp.get("domain") == "time" else "freq_Hz"
                headers += [f"{xname} [{lab}]", f"value [{lab}]"]
            with open(path, "w") as fh:
                fh.write(",".join(headers) + "\n")
                buf = []
                for i in range(n):
                    parts = []
                    for k in range(len(specs)):
                        if i < lens[k]:
                            x = i / fss[k] if is_time[k] else i * dfs[k]
                            parts.append(f"{x:.10g}")
                            parts.append(f"{float(amps[k][i]):.8g}")
                        else:
                            parts.append(""); parts.append("")
                    buf.append(",".join(parts))
                    if len(buf) >= 20000:
                        fh.write("\n".join(buf) + "\n"); buf.clear()
                        self.status.setText(
                            f"Exporting CSV\u2026 {i + 1:,}/{n:,} rows")
                        QtWidgets.QApplication.processEvents()
                if buf:
                    fh.write("\n".join(buf) + "\n")
            self.status.setText(f"Saved {len(specs)} spectrum/spectra "
                                f"({n:,} rows): {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", str(e))

    # ----- theming -----
    def toggle_theme(self):
        self.dark = not self.dark
        self.apply_theme(); self._render()

    def apply_theme(self):
        self.t = THEMES["dark" if self.dark else "light"]
        QtWidgets.QApplication.instance().setStyleSheet(build_qss(self.t))
        self.theme_btn.setText("\u2600" if self.dark else "\u263d")
        self._recolor_toolbar()

    def _recolor_toolbar(self):
        """Tint the matplotlib nav-toolbar pictograms (home, arrows, pan,
        zoom, save, ...) to the theme text colour. matplotlib only recolours
        them once - against the OS palette - so on a dark desktop they come
        out white and vanish on the light theme; here they follow the app
        theme on every switch. Re-tinting is safe because SourceIn keeps the
        icon's alpha shape regardless of its current colour."""
        nav = getattr(self, "nav", None)
        if nav is None:
            return
        color = QtGui.QColor(self.t["fg"])
        for act in nav.actions():
            ic = act.icon()
            if ic.isNull():
                continue
            sizes = ic.availableSizes()
            base = ic.pixmap(sizes[-1]) if sizes else ic.pixmap(24, 24)
            if base.isNull():
                continue
            tinted = QtGui.QPixmap(base.size())
            tinted.setDevicePixelRatio(base.devicePixelRatio())
            tinted.fill(Qt.GlobalColor.transparent)
            p = QtGui.QPainter(tinted)
            p.drawPixmap(0, 0, base)
            p.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_SourceIn)
            p.fillRect(tinted.rect(), color)
            p.end()
            act.setIcon(QtGui.QIcon(tinted))

    def _theme_axes(self, ax):
        t = self.t
        ax.set_facecolor(t["ax"])
        for sp in ax.spines.values():
            sp.set_color(t["border"])
        ax.tick_params(colors=t["axfg"], labelsize=8)
        ax.xaxis.label.set_color(t["axfg"]); ax.yaxis.label.set_color(t["axfg"])
        ax.grid(True, color=t["grid"], lw=0.6, alpha=0.9)

    # ----- rendering helpers -----
    def _marker_lines(self, ax, fmax, gf, sb, legend):
        for k in range(1, self.p.n_harm + 1):
            f_h = k * gf["f_mesh"]
            if f_h > fmax:
                break
            ax.axvline(f_h, color="#ff453a", ls="--", lw=1.0, alpha=0.85,
                       label=("GMF harmonics" if (legend and k == 1) else None))
            for m_ in range(1, self.p.n_sb + 1):
                for f_side in (f_h - m_ * sb, f_h + m_ * sb):
                    if 0 < f_side <= fmax:
                        ax.axvline(f_side, color="#ff9f0a", ls=":", lw=0.8,
                                   alpha=0.7,
                                   label=(f"sidebands \u00b1{self.p.sb_spacing}"
                                          if (legend and k == 1 and m_ == 1
                                              and f_side == f_h - sb) else None))

    def _marker_lines_h(self, ax, fmax, gf, sb, legend):
        """Same gear-mesh / sideband markers as _marker_lines but drawn
        HORIZONTALLY, for the spectrogram where frequency is the y-axis."""
        for k in range(1, self.p.n_harm + 1):
            f_h = k * gf["f_mesh"]
            if f_h > fmax:
                break
            ax.axhline(f_h, color="#ff453a", ls="--", lw=1.0, alpha=0.85,
                       label=("GMF harmonics" if (legend and k == 1) else None))
            for m_ in range(1, self.p.n_sb + 1):
                for f_side in (f_h - m_ * sb, f_h + m_ * sb):
                    if 0 < f_side <= fmax:
                        ax.axhline(f_side, color="#ff9f0a", ls=":", lw=0.8,
                                   alpha=0.6,
                                   label=(f"sidebands \u00b1{self.p.sb_spacing}"
                                          if (legend and k == 1 and m_ == 1
                                              and f_side == f_h - sb) else None))

    def _theory_lines(self, ax, sp, gf, fmax, first):
        if not self.p.show_theory:
            return []
        stage = stage_from_label(sp["label"])
        Tq = torque_from_label(sp["label"], self.p.torque)
        mod = (stage_modulation(stage, self.p.beta, self.p.boa,
                                torque=Tq, torque_ref=self.p.torque_ref,
                                gamma=self.p.gamma)
               if stage is not None else None)
        if mod is None:
            return []
        _, A, B = mod
        C = theory_line_spectrum(A, B, self.p.n_sb)
        ex = 2.0 if sp.get("ytype") == "psd" else 1.0
        w = CONFIG["SB_WINDOW_HZ"]
        arts = []
        for k in range(1, self.p.n_harm + 1):
            fh = k * gf["f_mesh"]
            if fh > fmax:
                break
            i0, i1 = np.searchsorted(sp["f"], [fh - w, fh + w])
            seg = sp["amp"][i0:i1]
            if not seg.size:
                continue
            peak = float(seg.max())
            js = sorted(C)
            freqs = [fh + j * gf["f_cpl"] for j in js]
            amps = [(C[j] / C[0]) ** ex * peak for j in js]
            arts.append(ax.vlines(freqs, 0, amps, colors="#30d158", lw=1.4,
                                  alpha=self.p.theory_alpha,
                                  label=("expected (AM/FM model)"
                                         if (first and k == 1) else None)))
        return arts

    def _legend(self, ax):
        h, _ = ax.get_legend_handles_labels()
        if not h:
            return
        leg = ax.legend(loc="upper right", fontsize=7)
        leg.get_frame().set_facecolor(self.t["ax"])
        leg.get_frame().set_edgecolor(self.t["border"])
        for txt in leg.get_texts():
            txt.set_color(self.t["axfg"])

    # ----- the two main renderers -----
    def _render(self):
        self._collect_params()
        self._cbars = []
        if self.mode == "overlay" and self.spectra:
            self._render_overlay(); return
        specs = ([self.current] if self.mode == "single" else list(self.spectra))
        specs = [s for s in specs if s is not None]
        self.fig.clear(); self.fig.set_facecolor(self.t["fig"])
        if not specs:
            ax = self.fig.add_subplot(111); self._theme_axes(ax)
            ax.set_title("Open a recording to begin", color=self.t["axfg"])
            self._views, self._axes = [], [ax]
            self.canvas.draw_idle(); return
        if self.mode == "compare":
            specs = [s for s in specs if s.get("visible", True)]
            if not specs:
                self.canvas.draw_idle(); self._views, self._axes = [], []
                return
        gf, sb = marker_params(self.p)
        self._df = min((float(sp.get("df") or 0.01) for sp in specs),
                       default=0.01)
        domains = {sp.get("domain", "freq") for sp in specs}
        self._has_time = "time" in domains
        self._share_y = (len({(sp.get("ytype"), sp.get("unit"))
                              for sp in specs}) == 1)
        self._share_x = (len(domains) == 1)
        axes = self.fig.subplots(len(specs), 1, sharex=self._share_x,
                                 sharey=self._share_y, squeeze=False)[:, 0]

        def xmax_of(sp):
            dom = sp.get("domain", "freq")
            if dom == "spectrogram":
                t = sp.get("t")
                return float(t[-1]) if t is not None and len(t) else 1.0
            if dom == "time":
                return float(sp["f"][-1]) if len(sp["f"]) else 1.0
            return self.p.f_max or float(sp["f"][-1])

        def xlabel_for(dom):
            return "Time [s]" if dom in ("time", "spectrogram") \
                else "Frequency [Hz]"

        self._views, self._axes = [], list(axes)
        legend_ax = None
        for ax, sp in zip(axes, specs):
            dom = sp.get("domain", "freq")
            first = legend_ax is None
            if dom == "spectrogram":
                if first:
                    legend_ax = ax
                self._draw_spectrogram(ax, sp, gf, legend=first)
                self._theme_axes(ax)
                if len(specs) > 1:
                    ax.set_title(sp["label"], fontsize=8, loc="left",
                                 color=self.t["axfg"])
                self._views.append({"ax": ax, "kind": "spec",
                                    "domain": "spectrogram",
                                    "f": sp["f"], "t": sp["t"],
                                    "xmax": xmax_of(sp)})
                continue
            fmax_i = xmax_of(sp)
            line, = ax.plot([], [], lw=0.7,
                            color=sp.get("color") or self.t["line"],
                            alpha=sp.get("alpha", 1.0))
            theory = []
            if dom == "freq":
                if first:
                    legend_ax = ax
                self._marker_lines(ax, fmax_i, gf, sb, legend=first)
                theory = self._theory_lines(ax, sp, gf, fmax_i, first=first)
                self._draw_manual(ax, fmax_i, legend=first)
            self._theme_axes(ax)
            ax.set_ylabel(spec_ylabel(sp), fontsize=8)
            if len(specs) > 1:
                ax.set_title(sp["label"], fontsize=8, loc="left",
                             color=self.t["axfg"])
            self._views.append({"ax": ax, "line": line, "f": sp["f"],
                                "amp": sp["amp"], "theory": theory,
                                "domain": dom, "xmax": fmax_i})
        if self._share_x:
            axes[-1].set_xlabel(xlabel_for(next(iter(domains))))
        else:
            for v in self._views:
                v["ax"].set_xlabel(xlabel_for(v["domain"]))
        if len(specs) == 1:
            axes[0].set_title(
                f"{specs[0]['label']}   |   GMF = {gf['f_mesh']:.2f} Hz, "
                f"{self.p.sb_spacing} = {sb:.2f} Hz", fontsize=10,
                color=self.t["axfg"])
        else:
            st = self.fig.suptitle(
                f"Comparison ({len(specs)} graphs)   |   GMF = "
                f"{gf['f_mesh']:.2f} Hz, {self.p.sb_spacing} = {sb:.2f} Hz",
                fontsize=10)
            st.set_color(self.t["axfg"])
        self._legend(legend_ax or axes[0])
        self._set_limits(); self._connect_xlim()
        self._update_lines(); self._apply_scales(); self._fit_canvas()

    def _render_overlay(self):
        self._collect_params()
        self._cbars = []
        specs = [s for s in self.spectra if s.get("visible", True)
                 and s.get("domain") != "spectrogram"]
        self.fig.clear(); self.fig.set_facecolor(self.t["fig"])
        skipped_spec = any(s.get("domain") == "spectrogram"
                           and s.get("visible", True) for s in self.spectra)
        if not specs:
            ax = self.fig.add_subplot(111); self._theme_axes(ax)
            ax.set_title("Spectrograms can't be overlaid \u2014 use Compare "
                         "or Single." if skipped_spec else "Add graphs to "
                         "overlay.", color=self.t["axfg"])
            self._views, self._axes = [], [ax]
            self.canvas.draw_idle(); return
        ax = self.fig.add_subplot(111); self._axes = [ax]
        self._share_x = self._share_y = True
        gf, sb = marker_params(self.p)
        self._df = min((float(sp.get("df") or 0.01) for sp in specs),
                       default=0.01)
        domains = {sp.get("domain", "freq") for sp in specs}
        self._has_time = "time" in domains
        if domains == {"time"}:
            fmax = max(sp["f"][-1] for sp in specs); xlabel = "Time [s]"
        else:
            fmax = self.p.f_max or max(sp["f"][-1] for sp in specs)
            xlabel = "Frequency [Hz]"
        self._theme_axes(ax)
        if not self._has_time:
            self._marker_lines(ax, fmax, gf, sb, legend=True)
            self._draw_manual(ax, fmax, legend=True)
        # average across the visible spectra (needs >=2). Enabling either
        # "Average line" or "Average only" shows it; "Average only" also hides
        # the individual spectra so only the mean curve remains.
        want_avg = (self.p.show_avg or self.p.avg_only) and not self._has_time
        avg = average_spectrum(specs) if want_avg else None
        avg_only = bool(self.p.avg_only and avg is not None)
        self._views = []; units = set()
        for i, sp in enumerate(specs):
            units.add(sp.get("unit", "?"))
            if avg_only:
                continue
            color = sp.get("color") or GRAPH_PALETTE[i % len(GRAPH_PALETTE)]
            line, = ax.plot([], [], lw=0.9, color=color,
                            alpha=sp.get("alpha", 1.0), label=sp["label"])
            theory = []
            if not self._has_time and sp.get("domain", "freq") == "freq":
                theory = self._theory_lines(ax, sp, gf, fmax, first=(i == 0))
            self._views.append({"ax": ax, "line": line, "f": sp["f"],
                                "amp": sp["amp"], "theory": theory,
                                "domain": sp.get("domain", "freq"),
                                "xmax": fmax})
        if avg is not None:
            self._avg_view_line(ax, avg, fmax)
            n_avg = avg["label"].split()[-1]
            self.status.setText(
                f"Overlay average over {n_avg} spectra shown (bold)"
                + ("; individual spectra hidden." if avg_only else "."))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(spec_ylabel(specs[0]) if len(units) == 1
                      else "Amplitude (mixed units)", fontsize=9)
        ax.set_title(f"Overlay ({len(specs)} graphs)   |   GMF = "
                     f"{gf['f_mesh']:.2f} Hz, {self.p.sb_spacing} = {sb:.2f} Hz",
                     fontsize=10, color=self.t["axfg"])
        self._legend(ax)
        self._set_limits(); self._connect_xlim()
        self._update_lines(); self._apply_scales(); self._fit_canvas()

    def _avg_view_line(self, ax, avg, fmax):
        """Add the overlay 'average of all spectra' line as a normal view so
        it decimates and autoscales like any other curve, just bolder."""
        color = "#ffffff" if self.dark else "#000000"
        line, = ax.plot([], [], lw=2.2, color=color, alpha=0.95, zorder=6,
                        label=avg["label"])
        self._views.append({"ax": ax, "line": line, "f": avg["f"],
                            "amp": avg["amp"], "theory": [], "domain": "freq",
                            "xmax": fmax, "is_avg": True})

    def add_average_spectrogram(self):
        """Append the mean of all visible spectrograms as a new panel. Averages
        in linear power (dB applied at draw time), so it is a proper mean.
        Intended for repeat runs of one condition, not across fault stages."""
        self._collect_params()
        visible = [s for s in self.spectra if s.get("visible", True)
                   and s.get("domain") == "spectrogram"]
        avg = average_spectrogram(visible)
        if avg is None:
            QtWidgets.QMessageBox.information(
                self, "Need 2 spectrograms",
                "Build at least two spectrograms first; this averages them "
                "(e.g. repeat runs of one stage) into one panel.")
            return
        if len(self.spectra) >= MAX_STACK:
            QtWidgets.QMessageBox.information(
                self, "Stack limit reached",
                f"The stack is full ({MAX_STACK}). Remove a graph first.")
            return
        self._style(avg); self.spectra.append(avg)
        self.current = avg
        self._refresh_list(); self._select_in_list(len(self.spectra) - 1)
        if self.mode == "overlay":
            self.mode = "compare" if len(self.spectra) >= 2 else "single"
            self._sync_seg()
        self._render()
        self.status.setText(
            f"Added average spectrogram of {avg['label'].split()[-1]} panels.")

    # ----- spectrogram build + draw -----
    def compute_spectrogram_cmd(self):
        """Build a spectrogram per selected axis (mirrors Compute FFT). They
        join the same graph list and render in Single / Compare (one panel
        each); Overlay skips them. dB / colormap restyle live."""
        if self.dataset is None:
            QtWidgets.QMessageBox.information(self, "No file",
                                              "Open a file first.")
            return
        self._collect_params()
        ch = self.channel_box.currentText()
        axes = self._selected_axes()
        if not axes:
            QtWidgets.QMessageBox.information(
                self, "No axis selected",
                "Select at least one axis (Ctrl/Shift-click for several).")
            return
        is_gen = isinstance(self.dataset, tuple) and self.dataset[0] == "gen"
        added, errors, capped, last = 0, [], False, None
        for axis in axes:
            if len(self.spectra) >= MAX_STACK:
                capped = True
                break
            try:
                x, fs_det, unit, domain = extract_named(
                    self.dataset, self.channels, ch, axis, self.p)
            except Exception as e:
                errors.append(f"{axis}: {e}")
                continue
            if domain == "time":
                errors.append(f"{axis}: trend / temperature channel has no "
                              "spectrogram")
                continue
            self.fs_detected = fs_det
            fs = get_fs(fs_det, self.p)
            src = {"path": self.cur_path, "channel": ch, "axis": axis}
            if is_gen:
                src["loader"] = "gen"; src["spec"] = self.gen_spec
            base = f"{self.cur_file} \u00b7 {ch} \u00b7 {axis}"
            try:
                spec = self._make_spectrogram(x, fs, base, unit, src)
            except Exception as e:
                errors.append(f"{axis}: {e}")
                continue
            spec["fs_detected"] = fs_det
            self._style(spec); self.spectra.append(spec)
            last = len(self.spectra) - 1; added += 1
        if last is not None:
            self.current = self.spectra[last]
        if self.fs_detected is not None:
            fs = get_fs(self.fs_detected, self.p)
            self.fs_info.setText(
                f"fs detected {self.fs_detected:.2f} Hz \u00b7 used {fs:.2f} Hz")
        self._refresh_list()
        if last is not None:
            self._select_in_list(last)
        if self.mode == "overlay":
            self.mode = "compare" if len(self.spectra) >= 2 else "single"
            self._sync_seg()
        elif added > 1 and self.mode == "single":
            self.mode = "compare"; self._sync_seg()
        self._render()
        if errors:
            QtWidgets.QMessageBox.warning(self, "Some axes failed",
                                          "\n".join(errors))
        if capped:
            QtWidgets.QMessageBox.information(
                self, "Stack limit reached",
                f"Stopped at {MAX_STACK} graphs (the comparison limit).")
        self.status.setText(
            f"Built {added} spectrogram(s); stack: {len(self.spectra)} graph(s).")

    def _make_spectrogram(self, x, fs, label_base, unit, src):
        """Build a spectrogram spectrum dict from a time signal, using the
        Spectrogram tab's OWN settings and filter (independent of the FFT
        Filter tab). Stores linear power Sxx so the dB/linear toggle is a pure
        restyle (no rebuild)."""
        # map the spectrogram-specific filter onto a copy of Params so the
        # shared apply_filter_chain can be reused untouched.
        Pf = replace(self.p,
                     hampel_on=self.p.sg_hampel_on,
                     hampel_win=self.p.sg_hampel_win,
                     hampel_sigma=self.p.sg_hampel_sigma,
                     filt_on=self.p.sg_filt_on, filt_type=self.p.sg_filt_type,
                     filt_family=self.p.sg_filt_family, f_lo=self.p.sg_f_lo,
                     f_hi=self.p.sg_f_hi, f_order=self.p.sg_f_order,
                     f_rp=self.p.sg_f_rp, f_rs=self.p.sg_f_rs,
                     notch_on=self.p.sg_notch_on, notch_hz=self.p.sg_notch_hz,
                     notch_q=self.p.sg_notch_q)
        x, filt_txt = apply_filter_chain(x, fs, Pf)
        f, t, Sxx = compute_spectrogram(x, fs, self.p.sg_window,
                                        self.p.sg_nperseg,
                                        self.p.sg_overlap_pct / 100.0,
                                        detrend=self.p.sg_detrend)
        df = float(f[1] - f[0]) if len(f) > 1 else 1.0
        return {"label": f"{label_base} (spectrogram, {self.p.sg_window}{filt_txt})",
                "f": _store(f), "t": _store(t), "Sxx": _store(Sxx),
                "ytype": "spectrogram", "unit": unit, "domain": "spectrogram",
                "df": df, "floor": 1e-12, "fs": fs, "n": len(x), "src": src}

    def _draw_spectrogram(self, ax, sp, gf, legend):
        f = np.asarray(sp["f"], dtype=float)
        t = np.asarray(sp["t"], dtype=float)
        Sxx = np.asarray(sp["Sxx"], dtype=float)
        if self.p.sg_db:
            Z = 10.0 * np.log10(Sxx + 1e-20)
            cbl = "PSD [dB]"
        else:
            Z = Sxx
            cbl = f"PSD [{sp.get('unit', 'g')}\u00b2/Hz]"
        fmax = self.p.sg_fmax or (float(f[-1]) if len(f) else 1.0)
        jmax = min(int(np.searchsorted(f, fmax)) + 1, len(f))
        jmax = max(jmax, 2)
        mesh = ax.pcolormesh(t, f[:jmax], Z[:jmax, :], cmap=self.p.sg_cmap,
                             shading="auto")
        try:
            cb = self.fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.046)
            cb.ax.tick_params(colors=self.t["axfg"], labelsize=7)
            cb.set_label(cbl, color=self.t["axfg"], fontsize=8)
            cb.outline.set_edgecolor(self.t["border"])
            self._cbars.append(cb)
        except Exception:
            pass
        self._marker_lines_h(ax, fmax, gf, marker_params(self.p)[1],
                             legend=legend)
        ax.set_ylim(0, fmax)
        if len(t):
            ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_ylabel("Frequency [Hz]", fontsize=8)

    # ----- limits / scales / line decimation -----
    def _set_limits(self):
        if not self._views:
            return
        if self._share_x:
            self._axes[0].set_xlim(0, max(v["xmax"] for v in self._views))
        else:
            for v in self._views:
                if v.get("kind") == "spec":
                    continue
                v["ax"].set_xlim(0, v["xmax"])
        lviews = [v for v in self._views if v.get("kind") != "spec"]
        if not lviews:
            return
        if self._share_y:
            xm = max(v["xmax"] for v in lviews)
            top, bot = 0.0, 0.0
            for v in lviews:
                seg = v["amp"][v["f"] <= xm]
                if seg.size:
                    top = max(top, float(seg.max()))
                    if self._has_time:
                        bot = min(bot, float(seg.min()))
            pad = 0.1 * ((top - bot) if top > bot else (abs(top) or 1.0))
            lviews[0]["ax"].set_ylim(bot - (pad if self._has_time else 0),
                                     top + pad)
        else:
            for v in lviews:
                seg = v["amp"][v["f"] <= v["xmax"]]
                is_t = v["domain"] == "time"
                t_i = float(seg.max()) if seg.size else 1.0
                b_i = float(seg.min()) if (seg.size and is_t) else 0.0
                p_i = 0.1 * ((t_i - b_i) if t_i > b_i else (abs(t_i) or 1.0))
                v["ax"].set_ylim(b_i - (p_i if is_t else 0), t_i + p_i)

    def _connect_xlim(self):
        if self._share_x and self._axes:
            self._axes[0].callbacks.connect(
                "xlim_changed", lambda a: self._update_lines())
        else:
            for v in self._views:
                v["ax"].callbacks.connect(
                    "xlim_changed", lambda a: self._update_lines())

    def _apply_scales(self):
        for v in self._views:
            ax = v["ax"]
            if v.get("kind") == "spec":
                continue                 # spectrogram axes stay linear
            if v["domain"] == "freq":
                ax.set_xscale("log" if self.p.log_x else "linear")
            ax.set_yscale("log" if self.p.log_y else "linear")
            if not self.p.log_y and v["domain"] != "time":
                lo, hi = ax.get_ylim()
                if lo < 0:
                    ax.set_ylim(bottom=0)
        self.canvas.draw_idle()

    def _update_lines(self, *_):
        if not self._views:
            return
        try:
            width_px = int(self._axes[0].get_window_extent().width)
        except Exception:
            width_px = 1500
        if self._share_x:
            x0, x1 = self._axes[0].get_xlim()
            lo, hi = min(x0, x1), max(x0, x1)
            for v in self._views:
                if v.get("kind") == "spec":
                    continue
                fd, ad = decimate(v["f"], v["amp"], lo, hi, width_px)
                v["line"].set_data(fd, ad)
        else:
            for v in self._views:
                if v.get("kind") == "spec":
                    continue
                x0, x1 = v["ax"].get_xlim()
                lo, hi = min(x0, x1), max(x0, x1)
                fd, ad = decimate(v["f"], v["amp"], lo, hi, width_px)
                v["line"].set_data(fd, ad)
        self.canvas.draw_idle()

    def refresh_markers(self):
        if not self._views:
            return
        self.rerender_keep_zoom()

    def rerender_keep_zoom(self):
        if not self._views or not self._axes:
            self._render(); return
        xlim = self._axes[0].get_xlim()
        ylim = self._axes[0].get_ylim()
        self._render()
        if self._axes:
            self._axes[0].set_xlim(*xlim)
            self._axes[0].set_ylim(*ylim)
            self._update_lines()
            self.canvas.draw_idle()

    # ----- canvas sizing -----
    def _fit_canvas(self):
        vp = self.scroll.viewport()
        vh = vp.height()
        if vh < 10:
            return
        n = len(self._views)
        if self.mode == "compare" and n > 1:
            per = max(80, int(self.p.graph_px))
            target = min(n * per, 60000)
            self.canvas.setMinimumHeight(target)
            self._scrollable = target > vh + 1
        else:
            self.canvas.setMinimumHeight(0)
            self._scrollable = False
        self._view_h = vh
        self._apply_fig_margins()

    def _apply_fig_margins(self):
        if not self._views:
            try:
                self.fig.tight_layout()
            except Exception:
                pass
            self.canvas.draw_idle()
            return
        n = len(self._views)
        has_spec = any(v.get("kind") == "spec" for v in self._views)
        if self.mode == "compare" and n > 1 and not has_spec:
            H = max(1, self.canvas.height())
            self.fig.subplots_adjust(
                left=self.p.sp_left, right=self.p.sp_right,
                top=1.0 - self.p.sp_top_px / H,
                bottom=self.p.sp_bot_px / H,
                hspace=self.p.sp_hspace)
        else:
            try:
                self.fig.tight_layout()
            except Exception:
                pass
        self.canvas.draw_idle()

    def _on_canvas_resize(self, _event=None):
        self._apply_fig_margins()
        self._update_lines()

    # ----- wheel zoom -----
    def _on_scroll(self, event):
        ge = getattr(event, "guiEvent", None)
        ctrl = bool(getattr(ge, "modifiers", lambda: 0)()
                    & Qt.KeyboardModifier.ControlModifier) if ge else False
        if self.mode == "compare" and self._scrollable and not ctrl:
            return
        ax = event.inaxes
        if ax not in self._axes or event.xdata is None:
            return
        base = CONFIG["SCROLL_ZOOM_BASE"]
        factor = 1.0 / base if event.button == "up" else base
        is_spec = any(v.get("kind") == "spec" and v["ax"] is ax
                      for v in self._views)

        def zoomed(lo, hi, c, log):
            if log:
                lo, c = max(lo, 1e-12), max(c, 1e-12)
                ll, hh, cc = np.log10([lo, hi, c])
                return (10 ** (cc - (cc - ll) * factor),
                        10 ** (cc + (hh - cc) * factor))
            return (c - (c - lo) * factor, c + (hi - c) * factor)

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(*zoomed(x0, x1, event.xdata, self.p.log_x and not is_spec))
        ax.set_ylim(*zoomed(y0, y1, event.ydata, self.p.log_y and not is_spec))
        if not self.p.log_y and not is_spec:
            lo, hi = ax.get_ylim()
            if lo < 0:
                ax.set_ylim(bottom=0)
        self._update_lines(); self.canvas.draw_idle()

    # ----- left-click drag panning -----
    def _on_press(self, event):
        if (event.inaxes in self._axes and event.button == 1
                and not getattr(self.nav, "mode", "")
                and not self.canvas.widgetlock.locked()):
            self._pan_start = (event.x, event.y)
            self._pan_ax = event.inaxes

    def _on_motion(self, event):
        if self._pan_start is None or event.x is None or self._pan_ax is None:
            return
        ax = self._pan_ax
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        trans, inv = ax.transData, ax.transData.inverted()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        corners = trans.transform([[xlim[0], ylim[0]], [xlim[1], ylim[1]]])
        corners = corners - [dx, dy]
        (nx0, ny0), (nx1, ny1) = inv.transform(corners)
        ax.set_xlim(nx0, nx1)
        ax.set_ylim(ny0, ny1)
        self._update_lines(); self.canvas.draw_idle()

    def _on_release(self, _event):
        self._pan_start = None
        self._pan_ax = None

    # ----- focus / zoom-to-feature -----
    def _focus_on(self, x0, x1):
        """Zoom the frequency axes to [x0, x1] and autoscale their y;
        time-domain trends and spectrograms in the same view are untouched."""
        fviews = [v for v in self._views
                  if v.get("domain", "freq") == "freq"
                  and v.get("kind") != "spec"]
        if not fviews:
            return
        xaxes = ([self._axes[0]] if self._share_x
                 else [v["ax"] for v in fviews])
        for ax in xaxes:
            ax.set_xlim(x0, x1)
        if self._share_y:
            top, floor = 0.0, np.inf
            for v in fviews:
                i0, i1 = np.searchsorted(v["f"], [x0, x1])
                seg = v["amp"][i0:i1]
                if seg.size:
                    top = max(top, float(seg.max()))
                    pos = seg[seg > 0]
                    if pos.size:
                        floor = min(floor, float(pos.min()))
            if top <= 0.0:
                top = fviews[0]["ax"].get_ylim()[1]
            if self.p.log_y:
                fviews[0]["ax"].set_ylim(max(floor, 1e-12) * 0.5, top * 2)
            else:
                fviews[0]["ax"].set_ylim(0, top * 1.1)
        else:
            for v in fviews:
                i0, i1 = np.searchsorted(v["f"], [x0, x1])
                seg = v["amp"][i0:i1]
                if seg.size:
                    top = float(seg.max())
                    if self.p.log_y:
                        pos = seg[seg > 0]
                        fl = float(pos.min()) if pos.size else 1e-9
                        v["ax"].set_ylim(fl * 0.5, top * 2)
                    else:
                        v["ax"].set_ylim(0, top * 1.1)
        self._update_lines(); self.canvas.draw_idle()

    def focus_gmf(self):
        self._collect_params()
        if not self._views:
            QtWidgets.QMessageBox.information(self, "No graph",
                                              "Compute a spectrum first.")
            return
        gf, _ = marker_params(self.p)
        k = self.p.gmf_focus_k
        center = k * gf["f_mesh"]
        span = max((self.p.n_sb + 2) * gf["f_cpl"], 0.03 * center)
        self._focus_on(max(0.0, center - span), center + span)
        self.status.setText(f"Focused GMF \u00d7{k} = {center:.1f} Hz.")

    def focus_custom(self):
        self._collect_params()
        if not self._views:
            return
        c = self.p.focus_custom_hz
        if c <= 0:
            return
        gf, _ = marker_params(self.p)
        span = max((self.p.n_sb + 2) * gf["f_cpl"], 0.03 * c)
        self._focus_on(max(0.0, c - span), c + span)
        self.status.setText(f"Focused {c:.1f} Hz.")

    # ----- manual overlays -----
    def _draw_manual(self, ax, fmax, legend):
        first_l = True
        for (freq, lab) in self._manual_lines:
            if 0 < freq <= fmax:
                ax.axvline(freq, color="#64d2ff", ls="-", lw=1.0, alpha=0.9,
                           label=("manual line"
                                  if (legend and first_l) else None))
                first_l = False
                if lab:
                    ax.text(freq, 0.98, lab, rotation=90, va="top", ha="right",
                            transform=ax.get_xaxis_transform(), fontsize=7,
                            color="#64d2ff")
        first_b = True
        for (lo, hi, lab) in self._manual_bands:
            if hi > 0 and lo < fmax:
                ax.axvspan(max(lo, 0), min(hi, fmax), color="#64d2ff",
                           alpha=0.12,
                           label=("manual band"
                                  if (legend and first_b) else None))
                first_b = False
                if lab:
                    ax.text((max(lo, 0) + min(hi, fmax)) / 2, 0.98, lab,
                            va="top", ha="center",
                            transform=ax.get_xaxis_transform(), fontsize=7,
                            color="#64d2ff")

    def add_manual_line(self):
        self._collect_params()
        f = self.p.mk_freq
        if f <= 0:
            QtWidgets.QMessageBox.information(self, "Frequency needed",
                                              "Enter a positive frequency.")
            return
        self._manual_lines.append((f, self.p.mk_label.strip()))
        self.rerender_keep_zoom()
        self.status.setText(f"Added manual line at {f:.1f} Hz.")

    def add_manual_band(self):
        self._collect_params()
        lo, hi = self.p.mk_lo, self.p.mk_hi
        if hi <= lo:
            QtWidgets.QMessageBox.information(self, "Band invalid",
                                              "High must exceed low.")
            return
        self._manual_bands.append((lo, hi, self.p.mk_label.strip()))
        self.rerender_keep_zoom()
        self.status.setText(f"Added manual band {lo:.1f}-{hi:.1f} Hz.")

    def clear_manual(self):
        self._manual_lines.clear(); self._manual_bands.clear()
        self.rerender_keep_zoom()
        self.status.setText("Cleared manual overlays.")

    # ----- help / kitty -----
    def show_help(self):
        html = """
        <h2>Planetary Gearbox FFT Analyser</h2>
        <p>Load enDAQ <b>.IDE</b>, PhotonFirst FBG <b>.csv</b> or a generic
        CSV, then <b>Compute FFT</b> (one graph per selected axis). Use
        <b>Single</b>, <b>Compare</b> (stacked, scrollable) and
        <b>Overlay</b> (one axes) to view graphs.</p>

        <h3>Spectrogram</h3>
        <p>The <b>Spectrogram</b> tab is a self-contained workspace: set the
        window, segment length, overlap, colormap, frequency range and its own
        filter, then press <b>Generate spectrogram(s)</b> to build one per
        selected axis. They join the same graph list and work in <b>Single</b>
        and <b>Compare</b> (each its own panel) - ideal for comparing H0-H5 -
        but are <i>skipped in Overlay</i> (heatmaps can't share one axes). The
        spectrogram filter is independent of the FFT <b>Filter</b> tab. dB,
        colormap and frequency range restyle live; window / length / overlap /
        filter need a regenerate. <b>Average spectrogram</b> adds the mean of
        the visible spectrograms (in linear power) as a new panel.</p>

        <h3>Overlay average line</h3>
        <p>Tick <b>Average line (overlay)</b> on the FFT tab. In Overlay mode
        it draws the mean of every visible frequency spectrum as one bold
        line on a common frequency grid (needs &ge;2 spectra). It's a live
        view - toggle it on/off freely.</p>

        <h3>Markers &amp; the AM/FM model</h3>
        <p>Red dashed = gear-mesh harmonics; orange dotted = sidebands at the
        chosen spacing (default planet-fault <i>f_cpl</i>). The Theory tab's
        <b>expected-signal overlay</b> draws the Miao AM/FM line spectrum for
        a chosen fault stage.</p>

        <h3>Load (Bartelmus susceptibility)</h3>
        <p>Optional load model on the Theory tab. The carrier amplitude scales
        as A = A&#8320;&middot;(1 + &gamma;&middot;T/T_ref). With <b>&gamma; =
        0</b> (default) load is off and the sideband/carrier <i>ratio</i> is
        load-independent - the linear-model null. Set a motor torque T, the
        reference (max-load) torque T_ref and a susceptibility &gamma; to model
        the faulty-gearbox load sensitivity. Tag a graph's torque in its label
        as e.g. <i>75Nm</i> so the correlation report reads it per graph.</p>

        <h3>Focus, manual overlays, export</h3>
        <p><b>Focus GMF &times;k</b> / <b>Focus line</b> zoom to a feature.
        Add manual lines / bands on the Theory tab. <b>Export</b> saves the
        figure (PNG/SVG/PDF) or the line data (CSV); export the figure for
        spectrograms.</p>

        <h3>Tips</h3>
        <ul>
        <li>Plain wheel scrolls a tall Compare stack; Ctrl+wheel (or
        Single/Overlay) zooms at the cursor. Left-drag pans.</li>
        <li>Label graphs <i>H0..H5</i> so the correlation report can pair
        them with cut depth.</li>
        <li>Derive the real sample rate from timestamps - check it under
        <b>Analyze</b> (there's a ~0.8% offset from nominal).</li>
        </ul>
        """
        dlg = HelpDialog(self, html); dlg.show()
        self._help = getattr(self, "_help", [])
        self._help.append(dlg)

    def show_kitty(self):
        dlg = KittyDialog(self, self.t)
        dlg.show()
        self._kitty = getattr(self, "_kitty", [])
        self._kitty.append(dlg)


# ==========================================================================
# Module-level helpers for the two new features (pure; no Qt dependency)
# ==========================================================================
def average_spectrum(specs):
    """Mean of the visible frequency-domain line spectra on a common frequency
    grid (finest df, up to the shortest Nyquist). Returns a spectrum dict, or
    None if fewer than two usable spectra are present."""
    fr = [sp for sp in specs if sp.get("domain") == "freq" and len(sp["f"])]
    if len(fr) < 2:
        return None

    def _df_of(sp):
        d = float(sp.get("df") or 0.0)
        if d > 0:
            return d
        f = np.asarray(sp["f"], dtype=float)
        return float(f[1] - f[0]) if len(f) > 1 else 1.0

    df = max(min(_df_of(sp) for sp in fr), 1e-9)
    fmax = min(float(sp["f"][-1]) for sp in fr)
    if fmax <= 0:
        return None
    grid = np.arange(0.0, fmax + 0.5 * df, df)
    acc = np.zeros_like(grid)
    for sp in fr:
        acc += np.interp(grid, np.asarray(sp["f"], dtype=float),
                         np.asarray(sp["amp"], dtype=float))
    acc /= len(fr)
    units = {sp.get("unit", "?") for sp in fr}
    ytypes = {sp.get("ytype", "amplitude") for sp in fr}
    pos = acc[acc > 0]
    return {"label": f"AVERAGE of {len(fr)}", "f": _store(grid),
            "amp": _store(acc),
            "ytype": next(iter(ytypes)) if len(ytypes) == 1 else "amplitude",
            "unit": next(iter(units)) if len(units) == 1 else "mixed",
            "domain": "freq", "df": float(df),
            "floor": float(pos.min()) if pos.size else 1e-9,
            "fs": 2.0 * fmax, "n": len(grid)}


def average_spectrogram(specs):
    """Mean of the visible spectrograms, averaged in LINEAR power (dB is
    applied at draw time, so this is a true mean). Grids are aligned to the
    finest common (frequency, time) range and resampled if they differ.
    Returns a spectrogram dict, or None if fewer than two are present.
    Intended for repeat runs of one condition, not across fault stages."""
    sg = [sp for sp in specs if sp.get("domain") == "spectrogram"
          and sp.get("Sxx") is not None]
    if len(sg) < 2:
        return None
    f_lo = max(float(sp["f"][0]) for sp in sg)
    f_hi = min(float(sp["f"][-1]) for sp in sg)
    t_lo = max(float(sp["t"][0]) for sp in sg)
    t_hi = min(float(sp["t"][-1]) for sp in sg)
    if f_hi <= f_lo or t_hi <= t_lo:
        return None
    nf = min(len(sp["f"]) for sp in sg)
    nt = min(len(sp["t"]) for sp in sg)
    fg = np.linspace(f_lo, f_hi, nf)
    tg = np.linspace(t_lo, t_hi, nt)
    acc = np.zeros((nf, nt), dtype=float)
    for sp in sg:
        f = np.asarray(sp["f"], dtype=float)
        t = np.asarray(sp["t"], dtype=float)
        S = np.asarray(sp["Sxx"], dtype=float)
        if (S.shape == (nf, nt) and np.allclose(f, fg)
                and np.allclose(t, tg)):
            acc += S
            continue
        # resample: interpolate along frequency, then along time
        Sf = np.empty((nf, len(t)))
        for j in range(len(t)):
            Sf[:, j] = np.interp(fg, f, S[:, j])
        St = np.empty((nf, nt))
        for i in range(nf):
            St[i, :] = np.interp(tg, t, Sf[i, :])
        acc += St
    acc /= len(sg)
    df = float(fg[1] - fg[0]) if nf > 1 else 1.0
    units = {sp.get("unit", "g") for sp in sg}
    return {"label": f"AVERAGE spectrogram of {len(sg)}",
            "f": _store(fg), "t": _store(tg), "Sxx": _store(acc),
            "ytype": "spectrogram",
            "unit": next(iter(units)) if len(units) == 1 else "mixed",
            "domain": "spectrogram", "df": df, "floor": 1e-12,
            "fs": 2.0 * f_hi, "n": int(nt), "src": None}


def compute_spectrogram(x, fs, window, nperseg, overlap_frac, detrend=True):
    """Welch-style STFT power spectrogram via scipy.signal.spectrogram.
    Returns (f, t, Sxx) with Sxx in linear PSD units (the dB conversion is
    applied at draw time so the UI toggle needs no rebuild)."""
    x = np.asarray(x, dtype=float)
    if detrend:
        x = x - np.mean(x)
    nperseg = int(min(max(64, int(nperseg)), len(x))) if len(x) else int(nperseg)
    noverlap = int(np.clip(float(overlap_frac), 0.0, 0.95) * nperseg)
    f, t, Sxx = sps.spectrogram(x, fs=fs, window=window, nperseg=nperseg,
                                noverlap=noverlap, scaling="density",
                                mode="psd",
                                detrend=("constant" if detrend else False))
    return f, t, Sxx


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()