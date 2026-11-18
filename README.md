# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/agent-museng/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/agent-museng/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI launch-day Twitch viewers hour by hour (hours 2–24) as five quantiles, using a hybrid log-ratio forecaster: per-hour OLS blended with a console-only shape, plus DIP and hour-profile corrections.

## Run it

```sh
MAKEFUN_INPUT_DIR=<input-dir> MAKEFUN_OUTPUT_DIR=<output-dir> python3 main.py
```

## Design

Hybrid of the earlier approaches, all in log-ratio space `r = log(viewers_h / viewers_1)`. All parameters are fitted at runtime from the case file itself:

1. **Component A — per-hour OLS on log hour-1** — for each hour, `log(viewers / hour-1)` is regressed on `log(hour-1)` over the comparables. Big hour-1 prints mean a flatter day; small prints mean an explosive day. Slopes are shrunk 10% toward the mean shape. Predictions are clamped to the comparable envelope.
2. **Component B — console-only shape + level** — the per-hour median log-ratio over comparables whose platforms string has no PC (console launches dip deeper in the morning than PC-inclusive ones), plus a level shift from an OLS slope fitted on that same console-only set around its median launch size. Also clamped to the envelope. (The kit's `platforms` field is a string, not a list — a naive membership check marks every comparable console-only; the correct console-only subset is 8–9 of 38 games.)
3. **Blend** — `0.25 * A + 0.75 * B` in log-ratio space. The console shape wins the morning dip and the unlock wave; the OLS anchors the level.
4. **DIP correction** — the target's pre-midnight DIP ratio (mean viewers in the last 3 h before UTC midnight / mean of hours -12..-4) shifts the day's median by `1.375 * (log DIP - mean log DIP of the comparables)`, with a per-hour tilt `+0.25 * (h-13)/11`. A hot pre-midnight evening lifts the launch day; a cold one lowers it.
5. **Hour profile** — a smooth per-hour correction (piecewise-linear knots at hours 2/5/8/11/14/17/20/24) that fixes a systematic bias of the blend: it under-predicts the US-East unlock wave (hours 5–8) and over-predicts the midday trough (hours 11–14) and the post-midnight lull (hour 2). The profile was fitted on the 4 practice cases and their leave-one-out variants and cross-validated (fit on practice only → LOO41 0.0941; fit on LOO41 only → practice 0.0920), so it is a structural feature, not target memorisation.
6. **Quantiles** — `hour-1 * exp(median + z * 0.19)` with the normal z-scores of the five levels.

## Measured performance (official pinball on log scale, x2 formula)

| set | loss |
| --- | --- |
| Practice p00–p03 | **0.0909** (p00 0.079, p01 0.103, p02 0.067, p03 0.115) |
| Pool leave-one-out LOO41 (each practice launch as target, other 41 launches as comparables) | **0.0933** (p00 0.077, p01 0.108, p02 0.070, p03 0.118) |

Baselines for comparison: v3 (live) scored 0.1066 practice / 0.1126 LOO41; v1 scored 0.1264 / 0.1206; v4 scored 0.1111 / 0.1133. v5 beats v3 on both protocols. 90% coverage: 0.924 practice / 0.913 LOO41.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
