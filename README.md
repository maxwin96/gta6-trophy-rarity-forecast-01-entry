# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/maxwin96/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/maxwin96/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [gta6-trophy-rarity-forecast-01 bounty on make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, its earn
rate on Exophase at the case's target, as five quantiles per trophy
(5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## The model

A trophy's earn rate is a share of a growing, broadening player base.
Two forces move it: players keep earning the trophy (rate up, fastest
for easy/early trophies) and new players keep joining Exophase's
tracked set before they have earned it (dilution, rate down, hardest
for rare trophies). Both forces decelerate over a launch week, so the
trend is fitted against **ln(1 + hours since release)**, not raw time.

Per trophy the program fits a weighted line in logit space,

    logit(rate) = a + b * ln(1 + t),

over the case's whole history, and reads the line at the target.

**Weights.** A read counts only for the wall-clock time it represents
(gap to the next read, capped at 3 h), because the page sits unchanged
between Exophase refreshes and a long stale plateau is not dozens of
independent reads. Older reads also decay with a 36 h half-life. The
slope is fitted on the last 36 h only: the first day after release is
a transient (rates rising from zero as the first cohorts play), and a
line through it overstates the sustained trend badly. A rate printed
as exactly 0.00 is censored, not a measurement — the page prints two
decimals, so 0.00 only says the rate is below 0.005% — and treating
it as logit(0.01) would manufacture a huge false slope, so such reads
are skipped in the fit (the level still uses the last observed rate).

**Shrinkage.** The slope `b` is shrunk toward the mean slope of the
trophy's tier (platinum / gold / silver / bronze), with the strength
set by how precisely the trophy's own slope is estimated. A trophy
whose history pins its slope down keeps it; one whose history is flat
or noisy borrows its tier's. Tiers too small to pool (platinum, gold)
fall back to the global mean slope. The median is anchored to the
level of the most recent reads, so a bad fit can never drag the level
away from what the page actually shows.

**Spread (logit scale).** Four honest uncertainties, added in
quadrature:

- fit uncertainty: the slope's standard error times the log-time
  distance extrapolated;
- drift: a random-walk term, 0.25 logit per sqrt(day) of horizon,
  for movement the trend cannot see;
- count noise: 1/sqrt(earners), capped at 0.4 — a rate built on a
  handful of earners is intrinsically noisy;
- a 0.15 floor for refresh jumps and page noise.

Total spread is clamped to [0.25, 2.0] logit. Quantiles are the
median shifted by the standard normal quantiles times the spread,
mapped back to percent and rounded to 4 decimals.

**Edge cases.** A trophy last seen at (essentially) 100% can only
stay or dilute, so it gets a fixed high forecast (median 99.9%,
5% quantile 98.5%). Rates of 0% are fitted at the 0.01% clamp and
allowed to rise with their tier's slope. Null rates in history are
skipped for that trophy. If a case has no usable history at all, the
forecast falls back to a wide spread around 1%.

## Validity

The program self-checks every forecast before writing it (exactly
one row per trophy, five finite quantiles in 0..100, non-decreasing,
≤ 65,536 bytes) and, if anything unexpected happens, falls back to a
persistence forecast so no case is ever left without a valid output.

Deterministic: fixed iteration order, no randomness, no network,
Python 3.13 standard library only. Two runs on the same case write
byte-identical output.

## Running

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Reads every `<id>.in` in the input directory and writes `<id>.out`
to the output directory.
