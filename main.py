#!/usr/bin/env python3
"""Forecast GTA VI launch-day Twitch viewers, hour by hour (hours 2..24).

Method (comparables-weighted log-ratio ensemble, two anchors + dip correction):
  * Every comparable launch contributes its release-day curve as log-ratios
    to its own hour-1 print.  Only clean, console-only, rolling-midnight
    launches are used: PC-simultaneous launches unlock on one global clock
    and their UTC shape is different, and early-access-contaminated launches
    peaked days before release day.
  * Two ridge fits per hour predict the target's log-ratio:
      (a) anchored on the target's hour-1 print:  r_h ~ 1 + ln(h1)
      (b) anchored on the target's pre-midnight level: log(v_h) ~ 1 + ln(pre12),
          where pre12 is the mean viewers in the 12 hours before the UTC day
          (New Zealand / Australia / Asia already unlocked then, so pre12 is a
          far less noisy scale anchor than the dead-of-night hour 1).
    The two are blended with a weight that grows with the hour (early hours
    stay close to hour 1, later hours lean on the pre-midnight anchor).
  * DIP CORRECTION (round 3): the blend's mean residual is predicted by the
    pre-midnight DIP ratio -- mean viewers in the last 3 hours before UTC
    midnight divided by the mean in hours -12..-4.  A launch whose audience
    is already surging into midnight grows more during release day than its
    hour-1 print predicts; a launch whose pre-midnight audience collapses
    grows less.  The correction is a linear fit of the pool's own
    leave-one-out residuals on (dip - pool mean dip), slope ~1.0, fitted at
    run time from the case's own comparables -- never from the practice
    targets.  It is the one feature that predicts residuals with the same
    sign in the pool LOO (+0.79) and on practice (+0.93); pre12/h1 and
    ln(h1) both flip sign between the two and were rejected as traps.
  * Quantile spread: the mean of (a) the leave-one-out residual std of the
    two anchor fits, combined in the same blend, and (b) the kernel-weighted
    dispersion of the comparables' log-ratios around the blend median,
    shrunk by 0.65.  Validated on the four practice launches (official 2x
    pinball): 0.1277 mean (p00 0.146, p01 0.116, p02 0.098, p03 0.151) vs
    0.1588 for round 2; pool leave-one-out 0.1706 vs 0.2735.  Both metrics
    improve, so the correction is real, not fitted to practice.
  * Everything is deterministic, standard library only, no network.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

HOURS = list(range(2, 25))
SPREAD_MULT = 0.65


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def tier_of(comp):
    """'contaminated' / 'weak' / 'comparable' from the unlock note."""
    text = (comp.get("unlock") or "").lower()
    if "contaminated" in text:
        return "contaminated"
    if "tier: weak" in text:
        return "weak"
    return "comparable"


def release_curve(comp):
    """Hour -> viewers for the comparable's UTC release day (hours 1..24)."""
    day0 = parse_ts(comp["day_start"])
    out = {}
    for pt in comp.get("hourly") or []:
        v = pt.get("viewers")
        if not v:
            continue
        try:
            t = parse_ts(pt["from"])
        except Exception:
            continue
        h = int((t - day0).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            out[h] = float(v)
    return out


def log_ratios(curve):
    if 1 not in curve or curve[1] <= 0:
        return None
    return {h: math.log(v / curve[1]) for h, v in curve.items() if h >= 2}


def pre_mean(hourly, day_start, hours_back):
    """Mean viewers in the window (day_start - hours_back, day_start]."""
    day0 = parse_ts(day_start)
    vals = []
    for pt in hourly or []:
        v = pt.get("viewers")
        if not v:
            continue
        try:
            t = parse_ts(pt["from"])
        except Exception:
            continue
        dt = (day0 - t).total_seconds() / 3600.0
        if 0 < dt <= hours_back:
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def dip_ratio(hourly, day_start):
    """Pre-midnight dip ratio: mean viewers in the last 3 hours before the
    UTC day divided by the mean in hours -12..-4.  >1 means the audience is
    already surging into midnight; <1 means it collapsed before midnight.
    Returns None when the pre-midnight window has no usable data."""
    day0 = parse_ts(day_start)
    late = []
    early = []
    for pt in hourly or []:
        v = pt.get("viewers")
        if not v:
            continue
        try:
            t = parse_ts(pt["from"])
        except Exception:
            continue
        dt = (day0 - t).total_seconds() / 3600.0
        if 0 < dt <= 3:
            late.append(float(v))
        elif 4 < dt <= 12:
            early.append(float(v))
    if not late or not early:
        return None
    return (sum(late) / len(late)) / (sum(early) / len(early))


def ridge_fit(X, y, w, lam):
    """Weighted ridge regression; returns coefficients. No penalty on intercept."""
    n = len(X[0])
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for xi, yi, wi in zip(X, y, w):
        for i in range(n):
            b[i] += wi * xi[i] * yi
            for j in range(n):
                A[i][j] += wi * xi[i] * xi[j]
    for i in range(1, n):
        A[i][i] += lam
    # Gaussian elimination with partial pivoting.
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(A[r][i]))
        if abs(A[piv][i]) < 1e-12:
            continue
        A[i], A[piv] = A[piv], A[i]
        b[i], b[piv] = b[piv], b[i]
        for r in range(i + 1, n):
            f = A[r][i] / A[i][i]
            for cc in range(i, n):
                A[r][cc] -= f * A[i][cc]
            b[r] -= f * b[i]
    beta = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < 1e-12:
            beta[i] = 0.0
            continue
        beta[i] = (b[i] - sum(A[i][j] * beta[j] for j in range(i + 1, n))) / A[i][i]
    return beta


def loo_std(X, y, lam):
    """Leave-one-out residual std for a ridge fit (honest out-of-sample
    spread; in-sample residuals understate it on a 7-comparable pool)."""
    n = len(y)
    if n < 4:
        return 0.5
    s = 0.0
    for i in range(n):
        Xi = [x for j, x in enumerate(X) if j != i]
        yi = [v for j, v in enumerate(y) if j != i]
        beta = ridge_fit(Xi, yi, [1.0] * len(yi), lam)
        pred = sum(b * x for b, x in zip(beta, X[i]))
        s += (y[i] - pred) ** 2
    return math.sqrt(s / n)


def build_comparables(case):
    """Usable comparables: clean tier, console-only, >=18 scored hours."""
    comps = []
    for comp in case.get("comparables") or []:
        if tier_of(comp) == "contaminated":
            continue
        if tier_of(comp) == "weak":
            continue
        platforms = comp.get("platforms") or ""
        if "PC" in platforms:
            continue
        curve = release_curve(comp)
        lr = log_ratios(curve)
        if not lr or len(lr) < 18:
            continue
        pre12 = pre_mean(comp.get("hourly") or [], comp["day_start"], 12)
        dip = dip_ratio(comp.get("hourly") or [], comp["day_start"])
        comps.append({"lr": lr, "h1": curve[1], "pre12": pre12, "dip": dip})
    return comps


def predict_mu_sigma(comps, p1, p12):
    """Round-2 two-anchor blend. Returns {hour: (mu, sig)} with mu the
    blend log-ratio median and sig the shrunk spread."""
    lh1 = math.log(max(p1, 1.0))
    out = {}
    for h in HOURS:
        w_pre = 0.5 + 0.5 * (h - 2) / 22.0

        # --- anchor (a): hour-1 print -------------------------------------
        rows = [cp for cp in comps if h in cp["lr"]]
        if rows:
            X = [[1.0, math.log(cp["h1"])] for cp in rows]
            y = [cp["lr"][h] for cp in rows]
            beta = ridge_fit(X, y, [1.0] * len(rows), 1.0)
            mu_h1 = beta[0] + beta[1] * lh1
            sig_h1 = loo_std(X, y, 1.0)
        else:
            mu_h1, sig_h1 = 0.0, 0.5

        # --- anchor (b): pre-midnight level -------------------------------
        mu_pre, sig_pre = mu_h1, sig_h1
        if p12:
            rows2 = [cp for cp in rows if cp.get("pre12")]
            if len(rows2) >= 5:
                X2 = [[1.0, math.log(cp["pre12"])] for cp in rows2]
                y2 = [math.log(cp["h1"]) + cp["lr"][h] for cp in rows2]
                beta2 = ridge_fit(X2, y2, [1.0] * len(rows2), 1.0)
                mu_pre = (beta2[0] + beta2[1] * math.log(p12)) - lh1
                sig_pre = loo_std(X2, y2, 1.0)

        mu = (1.0 - w_pre) * mu_h1 + w_pre * mu_pre

        # Spread: mean of the LOO residual std (blended) and the
        # kernel-weighted dispersion of the comparables around the blend
        # median, shrunk by SPREAD_MULT; floor 0.10.
        sig_loo = math.sqrt(((1.0 - w_pre) * sig_h1) ** 2 + (w_pre * sig_pre) ** 2)
        if rows:
            num = 0.0
            den = 0.0
            for cp in rows:
                wgt = math.exp(-((math.log(cp["h1"]) - lh1) ** 2) / 2.0)
                num += wgt * (cp["lr"][h] - mu) ** 2
                den += wgt
            disp = math.sqrt(num / den) if den > 0 else 0.5
        else:
            disp = 0.5
        sig = max(((sig_loo + disp) / 2.0) * SPREAD_MULT, 0.10)
        out[h] = (mu, sig)
    return out


def dip_correction(comps):
    """Fit the dip correction on the pool's own leave-one-out residuals:
    for each comparable, predict it with the rest of the pool and record
    its mean residual; regress those residuals on (dip - pool mean dip).
    Returns (slope, dip_mean); slope 0 when the pool is too small to fit.
    The correction is fitted only on the pool -- never on any target."""
    pts = []
    for i, cp in enumerate(comps):
        if cp.get("dip") is None:
            continue
        rest = [c for j, c in enumerate(comps) if j != i]
        if len(rest) < 4:
            continue
        mu_sig = predict_mu_sigma(rest, cp["h1"], cp.get("pre12"))
        res = []
        for h in HOURS:
            if h in cp["lr"]:
                res.append(cp["lr"][h] - mu_sig[h][0])
        if res:
            pts.append((cp["dip"], sum(res) / len(res)))
    if len(pts) < 4:
        return 0.0, 1.0
    md = sum(p[0] for p in pts) / len(pts)
    mr = sum(p[1] for p in pts) / len(pts)
    num = sum((d - md) * (r - mr) for d, r in pts)
    den = sum((d - md) ** 2 for d, r in pts)
    slope = num / den if den > 1e-9 else 0.0
    return slope, md


def predict_case(case):
    known = (case.get("known") or [{}])[0]
    p1 = float(known.get("viewers") or 0) or 1.0
    if p1 < 1.0:
        p1 = 1.0
    pre12 = pre_mean(case.get("history_hourly") or [], case["day_start"], 12)
    dip = dip_ratio(case.get("history_hourly") or [], case["day_start"])

    comps = build_comparables(case)
    # Fallback pool: if the console-only pool is too small, use every clean launch.
    pool_all = []
    if len(comps) < 5:
        for comp in case.get("comparables") or []:
            if tier_of(comp) != "comparable":
                continue
            curve = release_curve(comp)
            lr = log_ratios(curve)
            if not lr or len(lr) < 18:
                continue
            pre12c = pre_mean(comp.get("hourly") or [], comp["day_start"], 12)
            dipc = dip_ratio(comp.get("hourly") or [], comp["day_start"])
            pool_all.append({"lr": lr, "h1": curve[1], "pre12": pre12c, "dip": dipc})
        if len(pool_all) > len(comps):
            comps = pool_all

    mu_sig = predict_mu_sigma(comps, p1, pre12)
    slope, dip_mean = dip_correction(comps)
    corr = slope * (dip - dip_mean) if dip is not None else 0.0

    out_hours = []
    for h in HOURS:
        mu, sig = mu_sig[h]
        qs = []
        prev = 0.0
        for z in Z:
            q = p1 * math.exp(mu + corr + z * sig)
            if q <= 0.0 or not math.isfinite(q):
                q = 1.0
            if q > 1e9:
                q = 1e9
            if q < prev:
                q = prev
            qs.append(round(q, 2))
            prev = q
        out_hours.append({"hour": h, "q": qs})
    return {"hours": out_hours}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        case_id = fn[:-3]
        with open(os.path.join(in_dir, fn), "r", encoding="utf-8") as f:
            case = json.load(f)
        forecast = predict_case(case)
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(forecast, f, separators=(",", ":"))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
