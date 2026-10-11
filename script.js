
"use strict";

/*
 * RATING LAB
 * Glicko-2 parameter experiment
 *
 * Baseline:
 *   Initial RD = 150
 *   Volatility = 0.06
 *   Tau = 0.5
 *
 * Important:
 * - All candidates receive the same simulated matches for each seed.
 * - Draws are included consistently in prediction and rating updates.
 * - A candidate cannot be recommended if predictive accuracy degrades
 *   beyond the configured guardrails.
 * - This is an experiment, not an automatic change to your live ladder.
 */

const CONFIG = {
  ratingCenter: 1000,
  scale: 173.7178,
  initialRD: 150,
  initialVolatility: 0.06,
  initialTau: 0.5,
  maxRD: 350,
  drawProbability: 0.08,
  epsilon: 0.000001,

  // Accuracy guardrails against the baseline.
  maxBrierDegradation: 0.002,
  maxLogLossDegradation: 0.002,

  weights: {
    brier: 0.50,
    logLoss: 0.30,
    rankError: 0.15,
    ratingSwing: 0.05
  },

  // Broad sweep: 11 initial-RD values × 8 volatility values × 12 tau values
  // = 1,056 combinations, including the unchanged baseline.
  rdValues: [50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350],
  volatilityValues: [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12],
  tauValues: [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.50]
};

const BASELINE = {
  rd: 150,
  volatility: 0.06,
  tau: 0.5
};

const $ = id => document.getElementById(id);

let latestResults = [];
let isRunning = false;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function createRandom(seed) {
  // Deterministic Mulberry32 PRNG.
  return function random() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(array, random) {
  const result = array.slice();

  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }

  return result;
}

function toMu(rating) {
  return (rating - CONFIG.ratingCenter) / CONFIG.scale;
}

function fromMu(mu) {
  return CONFIG.ratingCenter + CONFIG.scale * mu;
}

function toPhi(rd) {
  return rd / CONFIG.scale;
}

function fromPhi(phi) {
  return CONFIG.scale * phi;
}

function g(phi) {
  return 1 / Math.sqrt(1 + (3 * phi * phi) / (Math.PI * Math.PI));
}

function logistic(x) {
  // Numerically stable logistic function.
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }

  const ex = Math.exp(x);
  return ex / (1 + ex);
}

/*
 * Conditional win probability, excluding draws.
 * This is the Glicko expected score against the opponent,
 * adjusted for the opponent's rating uncertainty.
 */
function winProbability(a, b) {
  const muA = toMu(a.rating);
  const muB = toMu(b.rating);
  const phiB = toPhi(b.rd);

  return logistic(g(phiB) * (muA - muB));
}

/*
 * Three-outcome model:
 * P(draw) = d
 * P(win)  = (1-d) * p
 * P(loss) = (1-d) * (1-p)
 *
 * Its expected score is P(win) + 0.5 * P(draw).
 * The rating update below uses this same expected score.
 */
function predictProbabilities(a, b) {
  const p = winProbability(a, b);
  const d = CONFIG.drawProbability;

  return {
    win: (1 - d) * p,
    draw: d,
    loss: (1 - d) * (1 - p),
    expectedScore: (1 - d) * p + 0.5 * d
  };
}

function simulateOutcome(skillA, skillB, random) {
  const drawChance = CONFIG.drawProbability;

  if (random() < drawChance) {
    return 0.5;
  }

  // Conditional win probability; no-draw matches are binary.
  const p = logistic((skillA - skillB) / 173.7178);
  return random() < p ? 1 : 0;
}

/*
 * Generate one fixed synthetic competition per seed.
 * Every candidate is evaluated on exactly the same matches.
 */
function generateCompetition(seed, settings) {
  const random = createRandom(seed);
  const teamCount = settings.teams % 2 === 0
    ? settings.teams
    : settings.teams + 1;

  const teams = Array.from({ length: teamCount }, (_, i) => ({
    id: i,
    trueSkill: (random() - 0.5) * 500
  }));

  const seasons = [];

  for (let season = 0; season < settings.seasons; season++) {
    // Skill changes are real, but bounded and reproducible.
    for (const team of teams) {
      team.trueSkill += (random() - 0.5) * 55;
      team.trueSkill = clamp(team.trueSkill, -400, 400);
    }

    const matches = [];
    const pairCounts = new Map();

    for (let round = 0; round < settings.matches; round++) {
      const order = shuffle(
        teams.map(team => team.id),
        random
      );

      for (let i = 0; i < order.length; i += 2) {
        const a = teams[order[i]];
        const b = teams[order[i + 1]];

        const pairKey = [a.id, b.id].sort((x, y) => x - y).join(":");
        const count = pairCounts.get(pairKey) || 0;
        pairCounts.set(pairKey, count + 1);

        matches.push({
          a: a.id,
          b: b.id,
          outcome: simulateOutcome(a.trueSkill, b.trueSkill, random)
        });
      }
    }

    seasons.push({
      matches,
      trueSkills: teams.map(team => team.trueSkill)
    });
  }

  return {
    teamCount,
    seasons
  };
}

/*
 * Correct Glicko-2 volatility root finder.
 */
function updateVolatility(phi, sigma, delta, v, tau) {
  const a = Math.log(sigma * sigma);

  function f(x) {
    const ex = Math.exp(x);
    const numerator =
      ex * (delta * delta - phi * phi - v - ex);

    const denominator =
      2 * Math.pow(phi * phi + v + ex, 2);

    return numerator / denominator -
      (x - a) / (tau * tau);
  }

  let A = a;
  let B;

  if (delta * delta > phi * phi + v) {
    B = Math.log(delta * delta - phi * phi - v);
  } else {
    let k = 1;

    while (f(a - k * tau) < 0 && k < 100) {
      k++;
    }

    B = a - k * tau;
  }

  let fA = f(A);
  let fB = f(B);

  for (let iteration = 0;
       Math.abs(B - A) > CONFIG.epsilon && iteration < 100;
       iteration++) {

    const denominator = fB - fA;

    if (Math.abs(denominator) < 1e-15) {
      break;
    }

    const C = A + ((A - B) * fA) / denominator;
    const fC = f(C);

    if (!Number.isFinite(fC)) {
      break;
    }

    if (fC * fB < 0) {
      A = B;
      fA = fB;
    } else {
      fA /= 2;
    }

    B = C;
    fB = fC;
  }

  return Math.exp(A / 2);
}

/*
 * Apply one rating-period update to one player.
 * Results are [opponent, observed score] pairs.
 */
function updatePlayer(player, results, params) {
  const mu = toMu(player.rating);
  const phi = toPhi(player.rd);
  const sigma = player.volatility;

  if (results.length === 0) {
    const newPhi = Math.sqrt(phi * phi + sigma * sigma);

    return {
      ...player,
      rd: Math.min(CONFIG.maxRD, fromPhi(newPhi))
    };
  }

  const phiStar = Math.sqrt(phi * phi + sigma * sigma);
  let varianceSum = 0;
  let residualSum = 0;

  for (const result of results) {
    const opponent = result.opponent;
    const score = result.score;
    const opponentMu = toMu(opponent.rating);
    const opponentPhi = toPhi(opponent.rd);

    const gj = g(opponentPhi);
    const conditionalWin = logistic(gj * (mu - opponentMu));

    // Match the three-outcome draw model used by predictions.
    const expected =
      (1 - CONFIG.drawProbability) * conditionalWin +
      0.5 * CONFIG.drawProbability;

    varianceSum += gj * gj * expected * (1 - expected);
    residualSum += gj * (score - expected);
  }

  if (varianceSum <= 0 || !Number.isFinite(varianceSum)) {
    return { ...player };
  }

  const v = 1 / varianceSum;
  const delta = v * residualSum;

  const newSigma = updateVolatility(
    phi,
    sigma,
    delta,
    v,
    params.tau
  );

  const newPhiStar = Math.sqrt(phi * phi + newSigma * newSigma);
  const newPhi = 1 / Math.sqrt(
    1 / (newPhiStar * newPhiStar) + 1 / v
  );

  const newMu = mu + newPhi * newPhi * residualSum;

  return {
    rating: fromMu(newMu),
    rd: Math.min(CONFIG.maxRD, fromPhi(newPhi)),
    volatility: newSigma
  };
}

function calculateSpearman(valuesA, valuesB) {
  function ranks(values) {
    const sorted = values
      .map((value, index) => ({ value, index }))
      .sort((a, b) => a.value - b.value);

    const result = new Array(values.length);
    let start = 0;

    while (start < sorted.length) {
      let end = start;

      while (
        end + 1 < sorted.length &&
        sorted[end + 1].value === sorted[start].value
      ) {
        end++;
      }

      const averageRank = (start + end) / 2 + 1;

      for (let i = start; i <= end; i++) {
        result[sorted[i].index] = averageRank;
      }

      start = end + 1;
    }

    return result;
  }

  const a = ranks(valuesA);
  const b = ranks(valuesB);

  const meanA = a.reduce((sum, x) => sum + x, 0) / a.length;
  const meanB = b.reduce((sum, x) => sum + x, 0) / b.length;

  let numerator = 0;
  let squareA = 0;
  let squareB = 0;

  for (let i = 0; i < a.length; i++) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;

    numerator += da * db;
    squareA += da * da;
    squareB += db * db;
  }

  const denominator = Math.sqrt(squareA * squareB);
  return denominator === 0 ? 0 : numerator / denominator;
}

function evaluateCandidate(params, competitions) {
  const totals = {
    brier: 0,
    logLoss: 0,
    rankError: 0,
    ratingSwing: 0,
    sigmaChange: 0,
    finalVolatility: 0,
    matchCount: 0,
    playerSeasonCount: 0
  };

  for (const competition of competitions) {
    const players = Array.from(
      { length: competition.teamCount },
      () => ({
        rating: CONFIG.ratingCenter,
        rd: params.rd,
        volatility: params.volatility
      })
    );

    let totalAbsoluteSwing = 0;

    for (const season of competition.seasons) {
      // Capture predictions using ratings before this period's updates.
      const periodResults = Array.from(
        { length: competition.teamCount },
        () => []
      );

      for (const match of season.matches) {
        const a = players[match.a];
        const b = players[match.b];
        const prediction = predictProbabilities(a, b);

        const actual = match.outcome === 1
          ? [1, 0, 0]
          : match.outcome === 0.5
            ? [0, 1, 0]
            : [0, 0, 1];

        const predicted = [
          prediction.win,
          prediction.draw,
          prediction.loss
        ];

        for (let i = 0; i < 3; i++) {
          totals.brier += Math.pow(predicted[i] - actual[i], 2);
        }

        const actualIndex = match.outcome === 1
          ? 0
          : match.outcome === 0.5
            ? 1
            : 2;

        totals.logLoss -= Math.log(
          Math.max(1e-15, predicted[actualIndex])
        );

        totals.matchCount++;

        periodResults[match.a].push({
          opponent: { ...b },
          score: match.outcome
        });

        periodResults[match.b].push({
          opponent: { ...a },
          score: 1 - match.outcome
        });
      }

      // All players update from the same frozen period snapshot.
      const previous = players.map(player => ({ ...player }));

      for (let i = 0; i < players.length; i++) {
        const updated = updatePlayer(
          previous[i],
          periodResults[i],
          params
        );

        totalAbsoluteSwing += Math.abs(
          updated.rating - previous[i].rating
        );

        totals.sigmaChange += Math.abs(
          updated.volatility - previous[i].volatility
        );

        players[i] = updated;
        totals.playerSeasonCount++;
      }
    }

    const trueSkills = competition.seasons[
      competition.seasons.length - 1
    ].trueSkills;

    const finalRatings = players.map(player => player.rating);
    totals.rankError += 1 - calculateSpearman(finalRatings, trueSkills);

    for (const player of players) {
      totals.finalVolatility += player.volatility;
    }

    totals.ratingSwing += totalAbsoluteSwing;
  }

  const matches = Math.max(1, totals.matchCount);
  const playerSeasons = Math.max(1, totals.playerSeasonCount);

  return {
    ...params,
    brier: totals.brier / matches,
    logLoss: totals.logLoss / matches,
    rankError: totals.rankError / competitions.length,
    rankCorrelation: 1 -
      totals.rankError / competitions.length,
    ratingSwing: totals.ratingSwing / playerSeasons,
    meanVolatilityChange: totals.sigmaChange / playerSeasons,
    finalVolatility: totals.finalVolatility / playerSeasons
  };
}

function isBaseline(result) {
  return result.rd === BASELINE.rd &&
    result.volatility === BASELINE.volatility &&
    result.tau === BASELINE.tau;
}

function calculateObjective(result, baseline) {
  const ratio = (value, reference) =>
    value / Math.max(reference, 1e-12);

  return (
    CONFIG.weights.brier *
      ratio(result.brier, baseline.brier) +
    CONFIG.weights.logLoss *
      ratio(result.logLoss, baseline.logLoss) +
    CONFIG.weights.rankError *
      ratio(result.rankError, Math.max(baseline.rankError, 0.02)) +
    CONFIG.weights.ratingSwing *
      ratio(result.ratingSwing, Math.max(baseline.ratingSwing, 1))
  );
}

function passesGuardrails(result, baseline) {
  return (
    result.brier <= baseline.brier *
      (1 + CONFIG.maxBrierDegradation) &&
    result.logLoss <= baseline.logLoss *
      (1 + CONFIG.maxLogLossDegradation)
  );
}

function getSettings() {
  const read = (id, fallback, min, max) => {
    const value = Number($(id)?.value);
    return Number.isFinite(value)
      ? clamp(Math.round(value), min, max)
      : fallback;
  };

  return {
    seeds: read("seeds", 10, 1, 500),
    seasons: read("seasons", 8, 1, 100),
    teams: Math.max(4, read("teams", 16, 4, 100)),
    matches: read("matches", 8, 1, 100)
  };
}

function buildCandidates() {
  const candidates = [];

  for (const rd of CONFIG.rdValues) {
    for (const volatility of CONFIG.volatilityValues) {
      for (const tau of CONFIG.tauValues) {
        candidates.push({ rd, volatility, tau });
      }
    }
  }

  return candidates;
}

function setProgress(done, total) {
  const percent = total === 0 ? 0 : Math.round(done / total * 100);

  if ($("progressText")) {
    $("progressText").textContent = `${done} / ${total} candidate tests`;
  }

  if ($("progressPercent")) {
    $("progressPercent").textContent = `${percent}%`;
  }

  if ($("progressBar")) {
    $("progressBar").style.width = `${percent}%`;
  }
}

function fmt(value, digits = 4) {
  return Number.isFinite(value) ? value.toFixed(digits) : "N/A";
}

function renderResults(results, baseline) {
  const body = $("resultsBody");
  if (!body) return;

  const ordered = results.slice().sort((a, b) => {
    const aEligible = passesGuardrails(a, baseline);
    const bEligible = passesGuardrails(b, baseline);

    if (aEligible !== bEligible) return aEligible ? -1 : 1;
    return a.objective - b.objective;
  });

  const limit = clamp(Number($("rowLimit")?.value) || 25, 1, 2000);

  body.innerHTML = "";

  for (const result of ordered.slice(0, limit)) {
    const tr = document.createElement("tr");
    const eligible = passesGuardrails(result, baseline);
    const baselineRow = isBaseline(result);

    if (baselineRow) tr.classList.add("baseline-row");

    const cells = [
      baselineRow ? "BASELINE" : (eligible ? "Eligible" : "Guardrail failed"),
      result.rd,
      result.volatility.toFixed(3),
      result.tau.toFixed(2),
      fmt(result.brier, 6),
      fmt(result.logLoss, 6),
      fmt(result.rankCorrelation, 4),
      fmt(result.ratingSwing, 3),
      fmt(result.meanVolatilityChange, 7),
      fmt(result.objective, 5)
    ];

    for (const value of cells) {
      const td = document.createElement("td");
      td.textContent = String(value);
      tr.appendChild(td);
    }

    body.appendChild(tr);
  }
}

function renderSummary(results, baseline) {
  const eligible = results
    .filter(result => passesGuardrails(result, baseline))
    .sort((a, b) => a.objective - b.objective);

  // Only recommend a non-baseline candidate when it actually improves
  // the composite objective. Passing accuracy guardrails alone is not enough.
  const baselineResult = results.find(isBaseline) || baseline;
  const bestAlternative = eligible
    .filter(result => !isBaseline(result))
    .sort((a, b) => a.objective - b.objective)[0];
  const best = bestAlternative &&
    bestAlternative.objective < baselineResult.objective - 1e-12
      ? bestAlternative
      : baselineResult;

  // These cards report their named metric, not the composite-objective winner.
  const bestBrierResult = eligible
    .slice()
    .sort((a, b) => a.brier - b.brier)[0] || baselineResult;
  const bestRankResult = eligible
    .slice()
    .sort((a, b) => b.rankCorrelation - a.rankCorrelation)[0] || baselineResult;

  if ($("candidateCount")) {
    $("candidateCount").textContent = String(results.length);
  }

  if ($("bestBrier")) {
    $("bestBrier").textContent = fmt(bestBrierResult.brier, 6);
  }

  if ($("bestRank")) {
    $("bestRank").textContent = fmt(bestRankResult.rankCorrelation, 4);
  }

  if ($("baselineBrier")) {
    $("baselineBrier").textContent = fmt(baseline.brier, 6);
  }

  if ($("bestRD")) $("bestRD").textContent = String(best.rd);
  if ($("bestVolatility")) {
    $("bestVolatility").textContent = best.volatility.toFixed(3);
  }
  if ($("bestTau")) $("bestTau").textContent = best.tau.toFixed(2);
  if ($("bestObjective")) {
    $("bestObjective").textContent = fmt(best.objective, 5);
  }

  const status = $("verdict");

  if (status) {
    if (isBaseline(best)) {
      status.textContent =
        "Keep the baseline. No non-baseline candidate passed the accuracy guardrails and improved the objective.";
    } else {
      status.textContent =
        `Candidate ${best.rd} / ${best.volatility.toFixed(3)} / ${best.tau.toFixed(2)} passed the accuracy guardrails and scored best among eligible candidates. Validate on real historical matches before adopting it.`;
    }
  }

  const tauGroups = new Map();

  for (const result of results) {
    const key = `${result.rd}|${result.volatility}`;
    if (!tauGroups.has(key)) tauGroups.set(key, []);
    tauGroups.get(key).push(result);
  }

  let identicalTauGroups = 0;

  for (const group of tauGroups.values()) {
    if (group.length < 2) continue;

    const first = group[0];

    const identical = group.every(item =>
      Math.abs(item.brier - first.brier) < 1e-12 &&
      Math.abs(item.logLoss - first.logLoss) < 1e-12 &&
      Math.abs(item.ratingSwing - first.ratingSwing) < 1e-12 &&
      Math.abs(item.meanVolatilityChange - first.meanVolatilityChange) < 1e-12
    );

    if (identical) identicalTauGroups++;
  }

  if ($("status")) {
    $("status").textContent =
      `Complete. ${eligible.length} of ${results.length} candidates passed accuracy guardrails. ` +
      `${identicalTauGroups} RD/volatility groups showed no measurable tau sensitivity.`;
  }

  renderResults(results, baseline);
}

async function runExperiment() {
  if (isRunning) return;

  isRunning = true;

  const runButton = $("runBtn");
  if (runButton) runButton.disabled = true;

  if ($("status")) $("status").textContent = "Generating common test competitions…";

  try {
    const settings = getSettings();
    const candidates = buildCandidates();

    // Generate the test data once. Every candidate sees the same data.
    const competitions = [];

    for (let i = 0; i < settings.seeds; i++) {
      competitions.push(
        generateCompetition(12345 + i * 7919, settings)
      );
    }

    const baselineRaw = evaluateCandidate(BASELINE, competitions);
    baselineRaw.objective = 1;

    const results = [];
    const total = candidates.length;

    for (let i = 0; i < candidates.length; i++) {
      const candidate = candidates[i];
      let result;

      if (isBaseline(candidate)) {
        result = { ...baselineRaw };
      } else {
        result = evaluateCandidate(candidate, competitions);
      }

      result.objective = calculateObjective(result, baselineRaw);
      result.eligible = passesGuardrails(result, baselineRaw);
      results.push(result);

      setProgress(i + 1, total);

      // Yield to the browser so the progress UI can repaint.
      if (i % 2 === 1) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    latestResults = results;

    renderSummary(results, baselineRaw);

    if ($("status")) {
      $("status").textContent +=
        ` Baseline: Brier ${fmt(baselineRaw.brier, 6)}, ` +
        `log loss ${fmt(baselineRaw.logLoss, 6)}, ` +
        `mean volatility change ${fmt(baselineRaw.meanVolatilityChange, 8)}.`;
    }
  } catch (error) {
    console.error(error);

    if ($("status")) {
      $("status").textContent = `Experiment failed: ${error.message}`;
    }
  } finally {
    isRunning = false;
    if (runButton) runButton.disabled = false;
  }
}

function resetDashboard() {
  latestResults = [];
  setProgress(0, 0);

  if ($("resultsBody")) {
    $("resultsBody").innerHTML =
      '<tr><td colspan="10" class="empty">No results yet. Run the parameter sweep.</td></tr>';
  }
  if ($("candidateCount")) $("candidateCount").textContent = "0";
  if ($("bestBrier")) $("bestBrier").textContent = "—";
  if ($("bestRank")) $("bestRank").textContent = "—";
  if ($("baselineBrier")) $("baselineBrier").textContent = "—";
  if ($("bestRD")) $("bestRD").textContent = "—";
  if ($("bestVolatility")) $("bestVolatility").textContent = "—";
  if ($("bestTau")) $("bestTau").textContent = "—";
  if ($("bestObjective")) $("bestObjective").textContent = "—";
  if ($("verdict")) $("verdict").textContent = "Run the experiment to evaluate candidates.";
  if ($("status")) $("status").textContent = "Ready.";
}

if ($("runBtn")) {
  $("runBtn").addEventListener("click", runExperiment);
}

if ($("resetBtn")) {
  $("resetBtn").addEventListener("click", resetDashboard);
}

if ($("rowLimit")) {
  $("rowLimit").addEventListener("change", () => {
    if (latestResults.length === 0) return;

    // Re-render the existing results only. Re-generating all simulations
    // here would waste time and could make the display feel unresponsive.
    const baseline = latestResults.find(isBaseline);
    if (baseline) renderResults(latestResults, baseline);
  });
}

if ($("status")) {
  $("status").textContent = "Ready. Run the broad sweep to compare 1,056 parameter combinations.";
}
