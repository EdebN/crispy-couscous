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
- Points differential
- Home-rate baseline
- Neural network

Each base model also has a temperature-calibrated test variant, with the temperature fitted on validation predictions only. Predictions are grouped by season/week and results are applied after the full week's forecasts have been generated. Complete seasons are split chronologically into 60% warm-up, 20% validation, and 20% final test.

The dynamic offence/defence candidate is an initial, deliberately simple Phase 2 prototype. It maintains bounded log-scale team offence and defence states, updates them from capped pregame score residuals after each week, and regresses states 30% toward league average between seasons. It uses the independent-Poisson scoreline engine, so it tests whether evolving team states add value beyond the static historical score-average model. It does not yet model state uncertainty or use a fully estimated probabilistic score likelihood.

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

Compare independent Poisson as the simple reference against an overdispersed or NFL-score-aware alternative. Do not assume Poisson is correct simply because it is convenient. Avoid treating scores as independent if validation evidence shows meaningful residual dependence.

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
