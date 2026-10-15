# NFL Prediction Lab: Model Development Roadmap

## Objective

Build a robust NFL regular-season prediction model using historical game results and schedules only. Optimise for a balanced combination of probability quality, winner accuracy, stability, and generalisation to future seasons.

The model must forecast each game using only information available before that game's kickoff. Added complexity is accepted only when it improves chronological out-of-sample results against established baselines.

## Current system

The current browser-based lab (historical-tester.js) downloads the public nflverse games CSV and evaluates these candidates:

- Glicko-2
- Glicko-2 + recent form
- Elo
- Poisson scores
- Dynamic offence/defence
- NFL-aware margin distribution
- Points differential
- Home-rate baseline
- Neural network

Each base model also has a temperature-calibrated test variant, with the temperature fitted on validation predictions only. Predictions are grouped by season/week and results are applied after the full week's forecasts have been generated. Complete seasons are split chronologically into 60% warm-up, 20% validation, and 20% final test.

The dynamic offence/defence candidate is an initial, deliberately simple Phase 2 prototype. It maintains bounded log-scale team offence and defence states, updates them from capped pregame score residuals after each week, and regresses states 30% toward league average between seasons. It uses the independent-Poisson scoreline engine, so it tests whether evolving team states add value beyond the static historical score-average model. It does not yet model state uncertainty or use a fully estimated probabilistic score likelihood.

The NFL-aware margin-distribution candidate reweights the independent-Poisson score grid using the discrepancy between warm-up games' observed scoring margins and the margin probabilities those games' pregame Poisson forecasts assigned. The adjustment uses a 30-game prior, bounded multipliers, and normalization. Its global margin weights are fitted on warm-up seasons only and remain frozen throughout validation and final test. It is intended to learn systematic NFL margin-shape deviations (including possible concentration around key margins) without hard-coding those effects. It remains a candidate until validation and held-out testing show a repeatable improvement.

Keep these baselines and their definitions available throughout development. Do not silently change an existing baseline when adding a new model.

## Non-negotiable evaluation rules

1. **No future information:** every feature and rating for a forecast must be computed from data available before kickoff.
2. **Week-level snapshots:** predict all games in a week before applying any result from that week. If game dates/times are trustworthy and same-week results were available before a later kickoff, that can be modelled in a separate, explicit evaluation mode; never do so accidentally.
3. **Chronological splits:** reserve the earliest seasons for warm-up, the next block for validation/model selection, and the latest block as the untouched final test. Record the exact season boundaries in every run.
4. **No test-driven selection:** never choose the “best” model, calibrator, feature, threshold, or hyperparameter using final-test performance. Use validation (preferably rolling-origin validation within it) for decisions. Run the final test only after a candidate is frozen.
5. **Same games:** compare models on the same eligible games. Report sample counts and missing-data exclusions.
6. **Reproducibility:** log model version, settings, data-source URL, run date, split boundaries, and metrics.
7. **Honest uncertainty:** report uncertainty for metric differences, ideally with season-block bootstrap intervals. Do not declare a win based on a tiny metric difference without uncertainty analysis.
8. **No synthetic replacement data:** fail visibly if source data cannot be loaded or required fields are absent.

## Metrics

Primary metrics:
- Multiclass Brier score for home win / tie / away win.
- Multiclass log loss.

Secondary metrics:
- Outcome accuracy.
- Home-win probability calibration and three-way reliability.
- Performance by season and probability/confidence band.
- Ranking quality of team-strength estimates.
- Stability under alternative season boundaries and leave-one-season-out checks.

Lower Brier score and log loss are better. Accuracy alone cannot establish probability quality. Calibration bins must show sample counts and should not be overinterpreted when sparse.

## Implementation phases

### Phase 0 — Establish a trustworthy benchmark

Tasks:
- Add a run manifest with the data range, exact season split, game counts, and model settings.
- Add automated checks for duplicate games, missing scores, invalid teams, invalid probabilities, and non-finite values.
- Verify the week-level forecast-before-update invariant with a small fixture where changing a game's result cannot change another prediction made in the same week.
- Ensure all compared models use the same test games.
- Remove final-test log loss from model selection and “best model” messaging. The default selection/highlight must be based on validation performance, with final-test results presented as confirmation only.
- Keep final-test results visible but label them clearly as held out.

Acceptance criteria:
- A clean run reproduces the existing six baseline models.
- Every probability vector is finite, in [0, 1], and sums to 1 within numerical tolerance.
- The app never declares the final-test winner to be the chosen model.
- Split boundaries and game counts are displayed and stored with the run.

### Phase 1 — Fix venue and home-advantage semantics

Tasks:
- Inspect the actual nflverse CSV schema for neutral-site or venue fields before using them.
- Separate designated home/away labels from actual home-field advantage.
- Give neutral-site games zero ordinary home-field bonus, subject to data verification.
- Treat shared-stadium games carefully; do not infer a full ordinary home advantage solely from the designated home label.
- Estimate home-field advantage on training/validation data only, with shrinkage and a sensible bound. Compare one league-wide effect with a small number of justified venue categories only if the data supports them.
- Do not infer travel distance or weather without external data; the chosen data scope is historical results and schedules only.

Acceptance criteria:
- Venue handling is documented and tested.
- The model does not apply an ordinary home advantage to known neutral-site games.
- Home advantage is selected without test-set optimisation.

### Phase 2 — Strong dynamic team-strength model

Build a new candidate alongside, not in place of, existing Glicko-2 and Elo baselines.

Recommended starting point:
- Latent offensive strength and defensive strength per team, rather than one rating alone.
- A time-evolving state with gradual drift between games/seasons.
- Partial pooling/shrinkage toward league average for teams with little data.
- Explicit offseason regression toward league average, tuned on validation.
- Separate home-field effect.
- A robust scoring-margin likelihood or outcome likelihood that limits the influence of extreme games.
- Uncertainty that grows during inactivity and contracts as evidence accumulates.

Begin with a tractable dynamic linear/Gaussian or regularised state-space formulation. Use a more elaborate Bayesian implementation only if it can be tested, explained, and maintained in the current browser app.

Acceptance criteria:
- The new candidate beats or meaningfully complements the best baseline on validation Brier/log loss.
- Ratings adapt after team performance changes without overreacting to one blowout.
- Uncertainty and offseason behaviour have unit tests.
- Every parameter is fixed before final-test evaluation.

### Phase 3 — Model the score distribution

Add a score-based candidate that predicts both teams' points using only prior game results and schedule context.

Candidate formulation:
- Team offence and opponent defence effects.
- League scoring environment and home effect.
- Partial pooling for sparse team histories.
- A score likelihood that respects the discrete, non-negative nature of points and is checked against actual NFL score distributions.
- Derive home-win, tie, and away-win probabilities from the joint score distribution.
- Enumerate a bounded grid of plausible score pairs (for example, each team's scores from 0 to a validated upper bound), then sum the joint probabilities into win/tie/loss classes. This is computationally cheap and gives a transparent route from score predictions to outcome probabilities.
- Treat NFL scoring as discrete and structured. Test whether a model reproduces the observed distribution of scoring totals and margins, especially mass around margins of 3 and 7. These key numbers matter directly to margin predictions; they must be learned from the historical sample rather than hard-coded as universal percentages.
- Compare an empirical or smoothed NFL-aware score distribution with independent Poisson and overdispersed alternatives. Check tail behaviour and ties explicitly.

The lab's chosen data scope excludes sportsbook odds, so implied-probability conversion, vig removal, spread prices, and market line movement are contextual knowledge only, not model inputs or benchmark targets. A future separate market-comparison experiment would need its own clearly labelled data source and leakage policy.

Acceptance criteria:
- Predicted score distributions are numerically stable.
- Outcome probabilities are obtained consistently from score outcomes, including ties.
- Calibration and proper scoring rules improve or the candidate remains an experimental option.

### Phase 4 — Combine models only if useful

Test a small regularised ensemble of the strongest dynamic-rating and score-distribution candidates.

- Learn weights on validation data only.
- Compare the ensemble with each component on identical games.
- Run ablations: remove one component at a time and measure the change.
- Avoid stacking many weak or redundant models.

Acceptance criteria:
- The ensemble improves validation metrics consistently across multiple seasons.
- Any apparent improvement is larger than plausible sampling noise.
- The final-test set remains untouched until the ensemble is frozen.

### Phase 5 — Probability calibration

- Fit calibration parameters only on validation predictions.
- Compare uncalibrated probabilities, temperature scaling, and a regularised logistic/multiclass calibrator.
- Use rolling-origin validation or cross-fitting inside the validation period where practical, to reduce overfitting the calibrator.
- Apply the selected calibrator unchanged to the final test.
- Report calibration with counts, uncertainty, Brier score, and log loss.
- Preserve ranking comparisons: temperature scaling preserves class ranking in many cases, but can still change the most likely class if probabilities are transformed in a multiclass setting.

Acceptance criteria:
- Calibration is selected without test feedback.
- It improves validation probability metrics and does not create pathological extremes.
- Final-test results are reported once for the frozen candidate.

### Phase 6 — Robustness and release

- Evaluate rolling-origin splits and season-by-season performance.
- Use season-block bootstrap intervals for comparisons.
- Test sensitivity to starting ratings, offseason regression, home advantage, and unusual-score handling.
- Document known limitations: no player availability, injuries, weather, betting odds, or play-by-play data.
- Promote a model only after it passes the same benchmark suite as every other candidate.

## Concrete build specification

### First: make the benchmark trustworthy
- Fix and test the prediction-record path before interpreting any leaderboard. The neural-network candidate must contribute exactly one prediction per eligible validation/test game; no undefined placeholder rows or duplicate predictions.
- Add a deterministic Node fixture covering duplicate games, invalid probability vectors, same-week updates, and complete model prediction coverage; run it in GitHub Actions.
- Run the real nflverse historical backtest in CI and print data range, split boundaries, per-model validation/test metrics, calibration temperatures, and per-model home-win calibration bins.
- Inspect the real backtest report before judging model quality; the synthetic fixture is only an integrity test.
- Keep the recent calibration table as a diagnostic, not a verdict. Show the model name and sample count; include uncertainty intervals or clearly flag sparse bins. The 0–20% bin with 39 games should trigger investigation, not a confident conclusion.

### Mathematical starting point for the score model

Use a log-link so expected points stay positive. Let `o[i,t]` be team i's latent offensive strength, `d[i,t]` its defensive strength (positive means it suppresses the opponent), `mu_H` and `mu_A` the league home/away scoring baselines, and `h` the home effect:

```text
log(lambda_home) = mu_H + h + offense[home] - defense[away]
log(lambda_away) = mu_A     + offense[away] - defense[home]
```

These are expected-score predictors, not yet the final outcome probabilities. Centre team effects around zero and shrink them toward zero when evidence is sparse; otherwise the model can mistake two early blowouts for a team's permanent identity.

Start with a deliberately simple state update. After a game, compute each side's scoring residual against its pregame expectation. Update offensive strength in proportion to its own residual and defensive strength in proportion to the opponent's residual, with small learning rates and bounded changes. Apply the updates only after all games in that week have been forecast. During the offseason, regress states toward league average and increase uncertainty. Select learning rates, shrinkage, and offseason regression on validation only.

This online update is an engineering baseline, not a claim that one gradient step is the statistically optimal state estimator. If it improves validation reliably, compare it with a regularised state-space/Kalman-style update. Do not start with a high-dimensional Bayesian model before the simpler state update is measured.

### Candidate score distributions and training protocol

1. **Reference:** independent Poisson, where `P(H=h, A=a) = Pois(h; lambda_home) * Pois(a; lambda_away)`.
2. **Overdispersion candidate:** use a negative-binomial or another non-negative count distribution with a dispersion parameter estimated from training scores. Compare score log likelihood and outcome metrics, not just visual fit.
3. **NFL-aware candidate:** estimate a smoothed discrete residual/score distribution from warm-up history, conditional on predicted scoring level and matchup strength. Preserve empirical mass around plausible scoring totals and key margins. Use shrinkage so rare score pairs do not receive unstable probabilities.
4. **Joint dependence candidate:** only if residual diagnostics and validation support it, add a shared latent pace/game-state term or a smoothed margin adjustment. Any adjustment must preserve non-negative probabilities and normalise the full score grid.

For a grid where both scores range from 0 to K, calculate each joint cell, include or bound the tail beyond K, and normalise only when the omitted tail is demonstrably negligible or explicitly accounted for. Then aggregate cells into home win, tie, and away win. Check both score distributions and the margin distribution; matching outcome accuracy while generating implausible scores is not sufficient.

**Training boundary:** fit global parameters and distributional smoothing using warm-up seasons. During validation and test, keep those global parameters frozen, but update team states only with completed earlier games. Select hyperparameters and any calibrator on validation. Final-test data is for one frozen evaluation only.

### Second: build team strength from scores
Maintain the existing models unchanged as benchmarks. Add a separate candidate with distinct latent offence and defence strengths:
- Expected home points depend on home offence, away defence, league scoring level, and an estimated home effect.
- Expected away points depend on away offence, home defence, and league scoring level.
- Use shrinkage toward league average for teams with limited history and gradual offseason regression rather than resetting teams or treating last season as permanent truth.
- Update the team states only after every forecast in a week has been generated. Freeze global model parameters before validation/test; team states may update from completed prior games as a real forecast system would.
- Estimate neutral-site/home-effect handling only from verified fields and past data. If the source cannot reliably identify neutral venues, document that limitation rather than inventing a venue feature.

### Third: turn expected points into scoreline probabilities
Test candidates in increasing complexity:
1. Independent Poisson scores as a transparent reference.
2. An overdispersed discrete score model to handle variance beyond Poisson.
3. A joint, NFL-aware discrete scoreline model that can preserve observed score and margin patterns, including mass near margins of 3 and 7.

For the joint candidate, generate a finite score grid with a validated tail limit. For every score pair (h,a), calculate P(H=h,A=a); then:
- P(home win) = sum of P(H=h,A=a) where h>a.
- P(tie) = sum where h=a.
- P(away win) = sum where h<a.

The scoreline model must remain a proper probability distribution: every cell is non-negative, and the full grid plus any explicitly modelled tail mass sums to 1. Do not hard-code claims such as “15% of games finish three points apart”; estimate key-number effects from the training history and check their stability across seasons. Preserve joint dependence if evidence supports it; do not assume the two teams' scores are independent by convenience alone.

### Fourth: calibrate and combine only after the base models work
- Compare uncalibrated and calibrated three-class probabilities using validation-only fitting.
- Investigate the low-probability home-win calibration gap by model, season, sample size, and probability range. Check for a feature/implementation bug before attempting to calibrate it away.
- Test a small ensemble of the strongest dynamic-strength and scoreline candidates, with weights selected on validation only.
- Use season-block resampling or rolling-origin validation to assess whether apparent gains persist across eras. Keep the final test untouched until the candidate and all settings are frozen.

### Promotion gate
A candidate is promoted only if it improves validation Brier score and/or log loss consistently, does not materially damage the other primary metric, remains competitive across multiple seasons, and passes the integrity tests. Report accuracy, calibration, score-distribution diagnostics, and uncertainty alongside the headline metrics. If the gain is indistinguishable from noise, retain the simpler model.

## Proposed order of work

1. Fix evaluation/reporting leakage and add invariants.
2. Audit venue fields and home advantage.
3. Implement a dynamic offence/defence strength candidate.
4. Validate a score-distribution model.
5. Test a small ensemble.
6. Revisit calibration after selecting the underlying model.
7. Freeze and evaluate once on the final test set.

## Decision policy

A candidate advances only if it improves validation Brier score or log loss in a repeatable way, does not materially degrade the other primary metric, and remains stable across seasons. If the evidence is inconclusive, keep the simpler model.

The target is not maximum complexity. It is the most reliable forecast the available data can support.
