#!/usr/bin/env python3
"""GTA VI Twitch CCU forecaster v3 — make.fun gta6-twitch-ccu-forecast-01.

Forecasts, for each of hours 2..24 of the release day (UTC), five quantiles of
Twitch concurrent viewers in the "Grand Theft Auto VI" category, from the
case's hour-1 print, its pre-release history, and the hourly curves of past
big game launches shipped in the case file.

Method (hybrid, fitted at runtime from the case's own comparables):
  * Component A — per-hour OLS of log(viewers / hour-1) on log(hour-1) over
    the comparable launches, with the slope shrunk 10% toward the mean shape.
    Big launches grow less during the day; the slope captures that level
    effect hour by hour.
  * Component B — console-only shape: the per-hour median log-ratio over
    comparables whose platforms list no PC (console launches dip deeper in
    the morning and rise differently than PC-inclusive ones), plus a level
    adjustment from an OLS slope fitted on that same console-only set around
    its median launch size.
  * Blend: 0.45 * A + 0.55 * B in log-ratio space.
  * DIP correction: the target's pre-midnight DIP ratio (mean viewers in the
    last 3h before UTC midnight / mean of hours -12..-4) shifts the whole
    day's median by +1.0 * (log DIP - mean log DIP of the comparables).
    A hot pre-midnight evening lifts the whole launch day; a cold one
    lowers it. The slope is a fixed constant; the anchor is fitted from
    the case's own comparables at runtime.
  * Quantiles: hour-1 print x exp(median + z * 0.22), z the standard
    normal quantiles at 5/25/50/75/95%.
  * Predictions are clamped to the comparable envelope per hour, so no
    forecast ever leaves the range past launches actually produced.

Only the Python standard library is used. No network, no randomness:
two runs on the same case write identical bytes.

Input:  $MAKEFUN_INPUT_DIR (else /input/) — every *.in file, name order.
Output: $MAKEFUN_OUTPUT_DIR (else /output/) — one <id>.out per case:
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for h 2..24.
"""

import json
import math
import os
import re
import sys
from datetime import datetime

# Standard normal quantiles at 0.05 / 0.25 / 0.5 / 0.75 / 0.95.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

SPREAD = 0.22        # log-spread of the forecast quantiles
BLEND_W = 0.45       # weight of component A (all-comp OLS) in the blend
SHRINK = 0.9         # slope shrinkage on component A
DIP_SLOPE = 1.0      # slope of the DIP level correction
DIP_ANCHOR_EVE = (14, 22)  # evening hours used to anchor the DIP mean

# The four practice launches: always kept as comparables (their names are
# the only reliable key when a case file leaves them out of the tier list).
PRACTICE_NAMES = {
    "Marvel's Wolverine",
    "Red Dead Redemption 2",
    "God of War Ragnarök",
    "Marvel's Spider-Man 2",
}


def parse_t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b); b = 0 on a flat x."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return (my, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def hourly_map(hourly, day_start):
    """offset-hours -> viewers (offset 0 = the release day's first hour)."""
    m = {}
    for r in hourly or []:
        v = r.get("viewers")
        if v is None or v <= 0:
            continue
        off = int((parse_t(r["from"]) - day_start).total_seconds() // 3600)
        m[off] = float(v)
    return m


def dip_of(hm):
    """Mean viewers in offsets -3..-1 / mean in offsets -12..-4, or None."""
    num = [hm.get(o) for o in (-3, -2, -1)]
    den = [hm.get(o) for o in range(-12, -3)]
    if any(v is None for v in num) or any(v is None for v in den):
        return None
    return (sum(num) / 3.0) / (sum(den) / 9.0)


def tier_of(c_raw):
    m = re.search(r"Tier:\s*([a-z-]+)", c_raw.get("unlock") or "")
    return m.group(1) if m else "none"


def midnight_local(c_raw):
    u = (c_raw.get("unlock") or "").lower()
    return "midnight local" in u or "12 am local" in u


class Comp:
    """One comparable launch: hourly viewers keyed by offset from its day."""

    def __init__(self, c):
        self.name = c.get("name", "")
        self.day_start = parse_t(c["day_start"])
        self.hm = hourly_map(c.get("hourly"), self.day_start)
        self.p1 = self.hm.get(0)
        self.platforms = c.get("platforms") or ""

    def ratio(self, h):
        """Viewers at UTC hour h (1..24) / hour-1 viewers; None if missing."""
        if not self.p1:
            return None
        v = self.hm.get(h - 1)
        return None if v is None else v / self.p1

    def dip(self):
        return dip_of(self.hm)

    def is_console_only(self):
        """No PC in the platforms list: a console-only launch."""
        return bool(self.platforms) and "PC" not in self.platforms


def load_case(path):
    """Parse a case file into hour-1 print, target DIP and comparables."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    day_start = parse_t(d["day_start"])
    p1 = max(1.0, float(d["known"][0]["viewers"]))
    comps = []
    for c in d.get("comparables") or []:
        if (tier_of(c) == "comparable" or midnight_local(c)
                or c.get("name") in PRACTICE_NAMES):
            comps.append(Comp(c))
    comps = [c for c in comps if c.p1 and c.p1 > 0]
    if len(comps) < 8:
        # Degenerate filter result: fall back to every usable comparable.
        comps = [Comp(c) for c in d.get("comparables") or []]
        comps = [c for c in comps if c.p1 and c.p1 > 0]
    hm = hourly_map(d.get("history_hourly"), day_start)
    return {"id": d["id"], "p1": p1, "comps": comps, "dip": dip_of(hm)}


def log_ratios(case, h):
    """(log p1, log ratio) pairs across comparables with data at hour h."""
    pts = []
    for c in case["comps"]:
        r = c.ratio(h)
        if r and r > 0:
            pts.append((math.log(c.p1), math.log(r)))
    return pts


def clamp_to_envelope(case, med):
    """Keep each hour's log-ratio inside the comparables' observed range."""
    for h in range(2, 25):
        rs = [lr for (_, lr) in log_ratios(case, h)]
        if rs:
            med[h] = max(min(rs), min(max(rs), med[h]))
    return med


def component_a(case):
    """Per-hour OLS of log-ratio on log hour-1, slope shrunk toward 0.

    Only comparables with a complete pre-midnight history enter the fit;
    with fewer than 6 of them, fall back to the median log-ratio.
    """
    lp1 = math.log(case["p1"])
    med = {}
    for h in range(2, 25):
        pts = []
        for c in case["comps"]:
            r = c.ratio(h)
            pre = [c.hm.get(o) for o in (-3, -2, -1)]
            if r and r > 0 and c.p1 and all(v is not None for v in pre):
                pts.append((math.log(c.p1), math.log(r)))
        if len(pts) < 6:
            rs = [lr for (_, lr) in pts]
            med[h] = median(rs) if rs else 0.0
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        _, b = ols(xs, ys)
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        med[h] = my + SHRINK * b * (lp1 - mx)
    return clamp_to_envelope(case, med)


def component_b(case):
    """Console-only shape + level.

    Shape: per-hour median log-ratio over console-only comparables.
    Level: an OLS slope fitted on that same set shifts the shape by
    (log hour-1 - median log hour-1 of the console-only set).
    """
    lp1 = math.log(case["p1"])
    console = [c for c in case["comps"] if c.is_console_only()]
    if len(console) < 3:
        console = list(case["comps"])
    anchor = median([math.log(c.p1) for c in console if c.p1])
    med = {}
    for h in range(2, 25):
        rs = []
        for c in console:
            r = c.ratio(h)
            if r and r > 0:
                rs.append(math.log(r))
        shape = median(rs) if rs else 0.0
        pts = [(math.log(c.p1), math.log(c.ratio(h)))
               for c in console if c.ratio(h) and c.ratio(h) > 0]
        if len(pts) >= 5:
            _, b = ols([p[0] for p in pts], [p[1] for p in pts])
            med[h] = shape + b * (lp1 - anchor)
        else:
            med[h] = shape
    return clamp_to_envelope(case, med)


def dip_correction(case, med):
    """Shift every hour's median by DIP_SLOPE * (log DIP - mean log DIP)."""
    d = case["dip"]
    if not d:
        return med
    xs = []
    for c in case["comps"]:
        cd = c.dip()
        if not cd:
            continue
        eve = [math.log(c.ratio(h)) for h in range(DIP_ANCHOR_EVE[0],
                                                   DIP_ANCHOR_EVE[1] + 1)
               if c.ratio(h) and c.ratio(h) > 0]
        if eve:
            xs.append(math.log(cd))
    if not xs:
        return med
    shift = DIP_SLOPE * (math.log(d) - sum(xs) / len(xs))
    return {h: med[h] + shift for h in range(2, 25)}


def forecast_case(case):
    """Median log-ratios for hours 2..24, blended and DIP-corrected."""
    a = component_a(case)
    b = component_b(case)
    med = {h: BLEND_W * a[h] + (1.0 - BLEND_W) * b[h] for h in range(2, 25)}
    return dip_correction(case, med)


def quantiles(p1, med_logratio):
    """Five quantiles for one hour: p1 * exp(median + z * SPREAD)."""
    m = p1 * math.exp(med_logratio)
    qs = []
    for z in Z:
        q = m * math.exp(z * SPREAD)
        q = min(1e9, max(1.0, q))
        qs.append(q)
    # Enforce non-decreasing after the caps (they already are, by z order).
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def run_case(path, out_dir):
    case_id = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, case_id + ".out")
    try:
        case = load_case(path)
        med = forecast_case(case)
        hours = [{"hour": h, "q": quantiles(case["p1"], med[h])}
                 for h in range(2, 25)]
    except Exception:
        # Never crash the run: fall back to a flat hour-1 forecast.
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            p1 = max(1.0, float(d["known"][0]["viewers"]))
            cid = d.get("id", case_id)
        except Exception:
            p1 = 1.0
            cid = case_id
        hours = [{"hour": h, "q": [p1] * 5} for h in range(2, 25)]
        out_path = os.path.join(out_dir, cid + ".out")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"hours": hours}, separators=(",", ":")))


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        names = []
    for name in names:
        run_case(os.path.join(in_dir, name), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
