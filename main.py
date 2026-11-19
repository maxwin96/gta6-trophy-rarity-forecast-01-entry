"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Model: per-trophy trend in logit space against log time since release,
with hierarchical shrinkage of the trend toward the trophy's tier mean,
and a predictive spread built from fit uncertainty, horizon drift,
count noise and a noise floor.

Why this shape
-------------
A trophy's earn rate is a share of a growing, broadening player base.
Two forces move it: players keep earning the trophy (pushes the rate
up, fastest for easy/early trophies) and new players keep joining
Exophase's tracked set before they have earned it (dilution, pushes
the rate down, hardest for rare trophies). Both forces decelerate as
time passes, which is why the trend is fitted against ln(1 + hours
since release) rather than against raw time: equal steps in log time
are the natural clock of a launch week.

Per trophy the program fits a weighted line

    logit(rate) = a + b * ln(1 + t)

over the case's whole history, then reads the line at the target.
The weights do two jobs: a read only counts for the wall-clock time
it actually represents (the page sits unchanged between Exophase
refreshes, so a long stale plateau must not count as dozens of
independent reads), and older reads count less (launch-night dynamics
are not week-two dynamics). The slope is fitted on the last 36 h
only: the first day after release is a transient (rates rising from
zero as the first cohorts play), and a line through it overstates
the sustained trend badly. A rate printed as exactly 0.00 is
censored, not a measurement: the page prints two decimals, so 0.00
only says the rate is below 0.005%, and treating it as logit(0.01)
would manufacture a huge false slope, so such reads are skipped in
the fit (the level still uses the last observed rate). The slope b
is then shrunk toward the mean slope of the trophy's tier
(platinum/gold/silver/bronze), with
the shrinkage strength set by how precisely the trophy's own slope
is estimated: a trophy whose history pins its slope down keeps it,
one whose history is flat or noisy borrows its tier's. Tiers with
too few trophies to pool fall back to the global mean slope.

The forecast median is the shrunk line read at the target, anchored
to the level of the most recent reads so a bad fit can never drag the
level away from what the page actually shows.

The spread (logit scale) is the sum of four honest uncertainties:
  * fit uncertainty: the slope's standard error times the distance
    extrapolated in log time;
  * drift: a random-walk term growing with the square root of the
    horizon in days, for movement the trend cannot see;
  * count noise: 1/sqrt(earners), because a rate built on a handful
    of earners is intrinsically noisy;
  * a floor for refresh jumps and page noise.
Quantiles are the median shifted by the normal quantiles times that
spread, mapped back to percent. A trophy last seen at (essentially)
100% is handled separately: it can only stay or dilute, so its
forecast is a high median with a one-sided-looking downward tail.

Everything is deterministic: fixed iteration order, no randomness,
no wall-clock reads, no network. Standard library only.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The standard normal's quantile at each scored level (0.05 .. 0.95).
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# --- tuning constants (fixed in advance, never tuned on the practice) ---
RECENCY_HALF_LIFE_H = 36.0   # older reads lose half their weight per 36 h
GAP_CAP_H = 3.0              # one read counts for at most 3 h of wall time
SLOPE_WINDOW_H = 36.0        # the slope is fitted on the last 36 h only:
                             # the first day after release is a transient
                             # (rates rising from zero as the first cohorts
                             # play), and a line through it overstates the
                             # sustained trend badly
RESID_FLOOR = 0.12           # logit: a flat page is stale, not certain
SLOPE_PRIOR_SD_FLOOR = 0.25  # logit per log-hour
SLOPE_PRIOR_SD_CAP = 1.0
SLOPE_CLAMP = 2.5            # logit per log-hour
MOVE_CLAMP = 2.5             # logit: largest move the median may make
RW_SIGMA = 0.25              # logit per sqrt(day) of unmodelled drift
BASE_SIGMA = 0.15            # logit floor for refresh/page noise
COUNT_SIGMA_CAP = 0.4        # logit: 1/sqrt(earners), capped
SIGMA_MIN = 0.25
SIGMA_MAX = 2.0
LOGIT_MIN = -9.210340371976184   # logit(0.01)
LOGIT_MAX = 9.210340371976184    # logit(99.99)

# A trophy last seen at (essentially) 100%: median near the top with a
# downward tail for dilution by new players who have not earned it yet.
FULL_TROPHY_Q = [98.5, 99.6, 99.9, 99.97, 99.99]


def clamp_pct(value):
    """Clamp a percent into the range the logit is defined on."""
    if value < 0.01:
        return 0.01
    if value > 99.99:
        return 99.99
    return value


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    """Logit value back to a percent, stably for large |lg|."""
    if lg >= 0.0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, halves away from zero, as a plain number."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_time(text):
    """ISO 8601 timestamp (with Z) to an aware datetime in UTC."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def release_moment(game, history):
    """Midnight UTC of the release date, else the first read's time."""
    try:
        raw = (game or {}).get("release_date", "")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return parse_time(history[0]["at"])


def weighted_line(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x.

    Returns (a, b, resid_sd, se_b, x_mean) or None when there is nothing
    to fit. resid_sd is the weighted residual scale; se_b the slope's
    standard error from it.
    """
    total_w = sum(ws)
    if total_w <= 0.0 or len(xs) < 2:
        return None
    xm = sum(w * x for x, w in zip(xs, ws)) / total_w
    ym = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (ym, 0.0, 0.0, float("inf"), xm)
    b = sum(w * (x - xm) * (y - ym) for x, y, w in zip(xs, ys, ws)) / sxx
    a = ym - b * xm
    ss = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))
    resid = math.sqrt(ss / total_w)
    se_b = resid / math.sqrt(sxx)
    return (a, b, resid, se_b, xm)


def pool_slopes(estimates):
    """Precision-weighted mean slope and spread for a group of trophies.

    estimates is a list of (slope, se). Returns (mean, sd) or None when
    the group has fewer than two usable slopes.
    """
    usable = [(b, s) for b, s in estimates if math.isfinite(s)]
    if len(usable) < 2:
        return None
    weights = [1.0 / (s * s + 0.0625) for _, s in usable]
    total = sum(weights)
    mean = sum(w * b for w, (b, _) in zip(weights, usable)) / total
    var = sum(w * (b - mean) ** 2 for w, (b, _) in zip(weights, usable)) / total
    sd = math.sqrt(var)
    sd = max(SLOPE_PRIOR_SD_FLOOR, min(SLOPE_PRIOR_SD_CAP, sd))
    return (mean, sd)


def trophy_observations(case, index, times_h):
    """One trophy's (x, y, weight) series plus its last observed rate.

    times_h holds hours since release for every history read. A read's
    weight is the wall-clock time it represents (gap to the next read,
    capped) times a recency decay, so stale plateaus and launch-night
    reads cannot dominate the fit. Only reads inside the slope window
    (the last SLOPE_WINDOW_H hours) enter the fit: the release transient
    must not bend the trend. A rate of exactly 0.00 is censored, not a
    measurement of logit(0.01): the page prints two decimals, so 0.00
    only says the rate is below 0.005%, and treating it as -9.21 logit
    would manufacture a huge false slope. It is skipped here; the level
    still uses the last observed rate.
    """
    history = case["history"]
    n = len(history)
    xs, ys, ws = [], [], []
    last_rate = None
    cutoff = times_h[-1] - SLOPE_WINDOW_H
    for i, read in enumerate(history):
        rate = read["rates"][index]
        if rate is None:
            continue
        last_rate = rate
        if rate <= 0.0 or times_h[i] < cutoff:
            continue
        if i + 1 < n:
            gap = times_h[i + 1] - times_h[i]
        elif n > 1:
            gap = times_h[i] - times_h[i - 1]
        else:
            gap = 1.0
        gap = max(0.05, min(GAP_CAP_H, gap))
        age = times_h[-1] - times_h[i]
        recency = 0.5 ** (age / RECENCY_HALF_LIFE_H)
        xs.append(math.log1p(max(0.0, times_h[i])))
        ys.append(logit(rate))
        ws.append(gap * recency)
    return xs, ys, ws, last_rate


def forecast_trophy(median_logit, sigma):
    """Five quantiles (percent) from a logit median and logit spread."""
    out = []
    for z in Z:
        value = inv_logit(median_logit + z * sigma)
        out.append(round4(min(99.99, max(0.01, value))))
    # Rounding must never break the ordering.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case):
    """Forecast every trophy of one case; never raises on bad data."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    if not trophies:
        return {"trophies": []}
    if not history:
        # No reads at all: say 1% for every trophy with a wide spread.
        return {"trophies": [
            {"id": t.get("id", i + 1), "q": forecast_trophy(logit(1.0), 1.5)}
            for i, t in enumerate(trophies)
        ]}

    origin = release_moment(case.get("game"), history)
    times_h = []
    for read in history:
        try:
            moment = parse_time(read["at"])
        except (ValueError, TypeError, KeyError):
            moment = origin
        times_h.append(max(0.0, (moment - origin).total_seconds() / 3600.0))
    try:
        target_h = max(0.0, (parse_time(case["target"]) - origin).total_seconds() / 3600.0)
    except (ValueError, TypeError, KeyError):
        target_h = times_h[-1]
    x_target = math.log1p(target_h)
    x_last = math.log1p(times_h[-1])
    horizon_days = max(0.0, (case.get("horizon_minutes") or 0) / 1440.0)

    last_read = history[-1]
    players_last = last_read.get("players_tracked") or 0

    # First pass: per-trophy fits, and the slope pools by tier / global.
    fits = []
    by_tier = {}
    global_estimates = []
    for index, trophy in enumerate(trophies):
        xs, ys, ws, last_rate = trophy_observations(case, index, times_h)
        fit = None
        level = None
        if len(xs) >= 2 and len(set(ys)) >= 2:
            fit = weighted_line(xs, ys, ws)
        if fit is not None:
            a, b, resid, se_b, _ = fit
            resid = max(resid, RESID_FLOOR)
            # Recompute the slope error from the floored residual scale:
            # a page that never moved is stale, not certain.
            total_w = sum(ws)
            xm = sum(w * x for x, w in zip(xs, ws)) / total_w
            sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, ws))
            se_b = resid / math.sqrt(sxx) if sxx > 1e-12 else float("inf")
            se_b = max(se_b, 0.03)
            level = a + b * x_last
            global_estimates.append((b, se_b))
            by_tier.setdefault(trophy.get("tier"), []).append((b, se_b))
        elif ys:
            level = ys[-1]
        elif last_rate is not None:
            # No usable fit points (e.g. every rate so far is 0.00):
            # anchor the level to the last observed rate itself.
            level = logit(last_rate)
        if level is None:
            level = logit(1.0)
        # Anchor the level to what the page actually shows last: the
        # fitted line may not pass through the final read.
        if ys:
            level = min(max(level, ys[-1] - 0.5), ys[-1] + 0.5)
        fits.append({"fit": fit, "level": level, "last_rate": last_rate,
                     "se_b": se_b if fit is not None else None})

    global_prior = pool_slopes(global_estimates)
    tier_priors = {}
    for tier, estimates in by_tier.items():
        prior = pool_slopes(estimates) if len(estimates) >= 3 else None
        tier_priors[tier] = prior or global_prior

    # Second pass: shrunk slope, median, spread, quantiles.
    rows = []
    for index, trophy in enumerate(trophies):
        info = fits[index]
        last_rate = info["last_rate"]
        if last_rate is not None and last_rate >= 99.9:
            rows.append({"id": trophy["id"], "q": list(FULL_TROPHY_Q)})
            continue
        if last_rate is None:
            last_rate = 1.0

        prior = tier_priors.get(trophy.get("tier")) or global_prior
        if prior is None:
            prior = (0.0, 0.5)
        prior_mean, prior_sd = prior

        if info["fit"] is not None:
            _, b, _, _, _ = info["fit"]
            se_b = info["se_b"]
            if se_b is not None and math.isfinite(se_b):
                prec_own = 1.0 / (se_b * se_b)
                prec_prior = 1.0 / (prior_sd * prior_sd)
                slope = (b * prec_own + prior_mean * prec_prior) / (prec_own + prec_prior)
                slope_se = 1.0 / math.sqrt(prec_own + prec_prior)
            else:
                slope, slope_se = prior_mean, prior_sd
        else:
            slope, slope_se = prior_mean, prior_sd

        slope = max(-SLOPE_CLAMP, min(SLOPE_CLAMP, slope))
        move = slope * max(0.0, x_target - x_last)
        move = max(-MOVE_CLAMP, min(MOVE_CLAMP, move))
        median_logit = min(LOGIT_MAX, max(LOGIT_MIN, info["level"] + move))

        earners = max(0.0, last_rate) / 100.0 * max(0.0, players_last)
        count_sigma = 0.0
        if earners >= 1.0:
            count_sigma = min(COUNT_SIGMA_CAP, 1.0 / math.sqrt(earners))
        elif last_rate > 0:
            count_sigma = COUNT_SIGMA_CAP
        sigma = math.sqrt(
            (RW_SIGMA * math.sqrt(horizon_days)) ** 2
            + (slope_se * max(0.0, x_target - x_last)) ** 2
            + BASE_SIGMA ** 2
            + count_sigma ** 2
        )
        sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
        rows.append({"id": trophy["id"], "q": forecast_trophy(median_logit, sigma)})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def validate(forecast, trophies):
    """Strict self-check before a forecast is written out."""
    if set(forecast.keys()) != {"trophies"}:
        raise ValueError("top level must hold only 'trophies'")
    rows = forecast["trophies"]
    if len(rows) != len(trophies):
        raise ValueError("row count does not match trophy count")
    seen = set()
    for row in rows:
        if set(row.keys()) != {"id", "q"}:
            raise ValueError("row must hold only 'id' and 'q'")
        if row["id"] in seen:
            raise ValueError("duplicate trophy id")
        seen.add(row["id"])
        q = row["q"]
        if len(q) != 5:
            raise ValueError("each row needs five quantiles")
        prev = None
        for value in q:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("quantile must be a finite number")
            if value < 0.0 or value > 100.0:
                raise ValueError("quantile outside 0..100")
            if prev is not None and value < prev:
                raise ValueError("quantiles must be non-decreasing")
            prev = value
    if {t["id"] for t in trophies} != seen:
        raise ValueError("row ids do not match trophy ids")


def process_case(path):
    case = json.loads(path.read_text(encoding="utf-8"))
    forecast = forecast_case(case)
    validate(forecast, case.get("trophies") or [])
    blob = json.dumps(forecast, separators=(",", ":")) + "\n"
    if len(blob.encode("utf-8")) > 65536:
        raise ValueError("forecast exceeds 65,536 bytes")
    return blob


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            blob = process_case(path)
        except Exception:
            # Never leave a case without an output: fall back to the
            # last read's rates (persistence), which is always valid.
            case = json.loads(path.read_text(encoding="utf-8"))
            rows = []
            last = (case.get("history") or [{}])[-1]
            rates = last.get("rates") or []
            for i, trophy in enumerate(case.get("trophies") or []):
                rate = rates[i] if i < len(rates) and rates[i] is not None else 1.0
                rows.append({"id": trophy["id"], "q": forecast_trophy(logit(rate), 1.0)})
            rows.sort(key=lambda row: row["id"])
            blob = json.dumps({"trophies": rows}, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(blob, encoding="utf-8")


if __name__ == "__main__":
    main()
