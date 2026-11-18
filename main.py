#!/usr/bin/env python3
"""GTA6 Twitch CCU forecast — round 5.

Model (all computed at runtime from the case file, stdlib only):
  For each hour h = 2..24, work in log-ratio space r = log(viewers_h / viewers_1).
  - OLS: per-hour OLS of each comparable's log-ratio on its log hour-1
    viewers, slope shrunk 10%, evaluated at the target's log hour-1.
  - CONSOLE: per-hour median log-ratio over console-only comparables
    (platforms string without "PC"), level-adjusted by an OLS slope fitted
    on that subset: each comparable's ratio is shifted to the target's
    level before taking the median.
  - Blend: 0.25 * OLS + 0.75 * CONSOLE.
  - DIP: pre-midnight dip ratio of the target (mean viewers in the 3 hours
    before UTC midnight over mean viewers in hours -12..-4), centred on the
    comparables' mean log DIP. Coefficient 1.375 plus a per-hour tilt of
    +0.25 * (h - 13) / 11 (stronger late, weaker early).
  - PROFILE: a smooth per-hour correction (piecewise-linear through knots),
    fitted on the practice cases and their leave-one-out variants. It
    corrects a systematic bias of the blend: the blend under-predicts the
    US-East unlock wave (hours 5-8) and over-predicts the midday trough
    (hours 11-14) and the first hour after midnight (hour 2).
  - The median is clamped to the comparables' per-hour log-ratio envelope.
  - Extreme hour-1 safeguard: the OLS evaluation point and the console
    level adjustment are clamped to the comparables' observed log hour-1
    range plus a 0.5 margin, so an hour-1 beyond every comparable (the
    real GTA VI launch could exceed 1,000,000 viewers) cannot blow up
    the forecast. The final quantiles still scale with the true hour-1.
  - Quantiles: hour-1 * exp(median + z * 0.19), z the standard-normal
    quantiles at 5/25/50/75/95%.

Everything is deterministic. No network, no files besides the case input.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

W_OLS = 0.25
SHRINK = 0.9
DIP_K = 1.375
DIP_SLOPE = 0.25
SPREAD = 0.19
# Profile knots: (hour, correction) — piecewise linear between knots.
PROFILE_KNOTS = (
    (2, -0.10),
    (5, 0.16),
    (8, 0.16),
    (11, -0.10),
    (14, -0.16),
    (17, 0.00),
    (20, 0.04),
    (24, 0.02),
)


def _parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _ols(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def _hour_viewers(hourly, day_start):
    """Map release-day hour (1..24) -> viewers for one hourly series."""
    out = {}
    for row in hourly or []:
        v = row.get("viewers")
        if not v:
            continue
        try:
            t = _parse_ts(row["from"])
        except Exception:
            continue
        dh = (t - day_start).total_seconds() / 3600.0
        h = int(round(dh)) + 1
        if 1 <= h <= 24:
            out[h] = float(v)
    return out


def _dip(hourly, day_start):
    """Mean viewers in [D-3h, D) over mean viewers in [D-12h, D-4h)."""
    def mean_in(a, b):
        vals = []
        for row in hourly or []:
            v = row.get("viewers")
            if not v:
                continue
            try:
                t = _parse_ts(row["from"])
            except Exception:
                continue
            if a <= t < b:
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else None

    num = mean_in(day_start - timedelta(hours=3), day_start)
    den = mean_in(day_start - timedelta(hours=12),
                  day_start - timedelta(hours=4))
    if num is None or den is None or den <= 0:
        return None
    return num / den


def _profile(hour):
    knots = PROFILE_KNOTS
    if hour <= knots[0][0]:
        return knots[0][1]
    for (h0, v0), (h1, v1) in zip(knots, knots[1:]):
        if h0 <= hour <= h1:
            t = (hour - h0) / (h1 - h0)
            return v0 * (1 - t) + v1 * t
    return knots[-1][1]


def _comparable_features(comp):
    try:
        ds = _parse_ts(comp["day_start"])
    except Exception:
        return None
    hv = _hour_viewers(comp.get("hourly"), ds)
    if 1 not in hv or hv[1] <= 0:
        return None
    h1 = hv[1]
    lr = {}
    for h in range(2, 25):
        if h in hv and hv[h] > 0:
            lr[h] = math.log(hv[h] / h1)
    if len(lr) < 20:
        return None
    platforms = str(comp.get("platforms") or "")
    return {
        "logh1": math.log(h1),
        "lr": lr,
        "dip": _dip(comp.get("hourly"), ds),
        "console_only": "PC" not in platforms.upper(),
    }


def predict(case):
    known = case.get("known") or [{}]
    h1 = float(known[0].get("viewers") or 1.0)
    if h1 < 1.0:
        h1 = 1.0
    logh1 = math.log(h1)

    comps = []
    for c in case.get("comparables") or []:
        f = _comparable_features(c)
        if f is not None:
            comps.append(f)

    # Extreme hour-1 safeguard: the OLS fit and the console level
    # adjustment both extrapolate in log hour-1. Clamp the point at
    # which they are evaluated to the comparables' observed log hour-1
    # range plus a 0.5 margin, so an out-of-range hour-1 cannot blow
    # up the forecast. The final quantiles still scale with the true
    # hour-1; only the ratio prediction is guarded.
    if comps:
        _h1_lo = min(c["logh1"] for c in comps)
        _h1_hi = max(c["logh1"] for c in comps)
        logh1_eval = max(_h1_lo - 0.5, min(_h1_hi + 0.5, logh1))
    else:
        logh1_eval = logh1

    # Target DIP from its own pre-checkpoint hourly history.
    try:
        day_start = _parse_ts(case["day_start"])
    except Exception:
        day_start = None
    tdip = None
    if day_start is not None:
        tdip = _dip(case.get("history_hourly"), day_start)

    dips = [c["dip"] for c in comps if c.get("dip")]
    mean_logdip = (sum(math.log(d) for d in dips) / len(dips)) if dips else 0.0
    dip_centred = (math.log(tdip) - mean_logdip) if tdip else 0.0

    rows = []
    for h in range(2, 25):
        xs, ys = [], []
        cxs, cys = [], []
        all_lr = []
        for c in comps:
            if h in c["lr"]:
                xs.append(c["logh1"])
                ys.append(c["lr"][h])
                all_lr.append(c["lr"][h])
                if c["console_only"]:
                    cxs.append(c["logh1"])
                    cys.append(c["lr"][h])
        if not xs:
            med = 0.0
            lo = hi = 0.0
        else:
            a, b = _ols(xs, ys)
            ols_med = a + (b * SHRINK) * logh1_eval
            if len(cxs) >= 3:
                ca, cb = _ols(cxs, cys)
                cons_med = _median(
                    [y - cb * (x - logh1_eval) for x, y in zip(cxs, cys)])
            elif cys:
                cons_med = _median(cys)
            else:
                cons_med = ols_med
            dip_k = DIP_K + DIP_SLOPE * (h - 13) / 11.0
            med = (W_OLS * ols_med + (1.0 - W_OLS) * cons_med
                   + dip_k * dip_centred + _profile(h))
            lo, hi = min(all_lr), max(all_lr)
            med = max(lo, min(hi, med))
        qs = [h1 * math.exp(med + z * SPREAD) for z in Z]
        qs = [max(q, 1.0) for q in qs]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = predict(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
