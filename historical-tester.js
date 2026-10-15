"use strict";

const DATA_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv";
const CURRENT_YEAR = new Date().getFullYear();
const LAST_COMPLETE_SEASON = CURRENT_YEAR - 1;
const CENTER = 1500;
const SCALE = 173.7178;
const TAU = 0.5;
const EPS = 1e-6;
const MODEL_NAMES = ["Glicko-2", "Glicko-2 + recent form", "Elo", "Poisson scores", "Points differential", "Home-rate baseline"];
const $ = id => document.getElementById(id);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const logistic = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
let games = [];
let lastResultRows = [];
let busy = false;

function parseCSV(text) {
  const rows = [];
  let row = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"' && quoted && text[i + 1] === '"') { cell += '"'; i++; }
    else if (ch === '"') quoted = !quoted;
    else if (ch === "," && !quoted) { row.push(cell); cell = ""; }
    else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); cell = "";
      if (row.some(v => v !== "")) rows.push(row);
      row = [];
    } else cell += ch;
  }
  if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
  if (!rows.length) return [];
  const headers = rows.shift().map(v => v.trim());
  return rows.map(values => Object.fromEntries(headers.map((h, i) => [h, (values[i] ?? "").trim()])));
}

function parseGames(csv) {
  return parseCSV(csv).map(r => {
    const season = Number(r.season), week = Number(r.week);
    const hs = Number(r.home_score), as = Number(r.away_score);
    return {
      season, week, home: r.home_team, away: r.away_team, hs, as,
      date: r.gameday || r.game_date || "",
      gameType: r.game_type || "",
      scoresPresent: r.home_score !== "" && r.away_score !== ""
    };
  }).filter(g =>
    g.scoresPresent && Number.isFinite(g.season) && g.season >= 1999 && g.season <= LAST_COMPLETE_SEASON &&
    g.gameType === "REG" && Number.isFinite(g.week) && g.home && g.away &&
    Number.isFinite(g.hs) && Number.isFinite(g.as)
  ).sort((a, b) => a.season - b.season || a.week - b.week || a.date.localeCompare(b.date) || a.home.localeCompare(b.home));
}

function splitSeasons(data) {
  const seasons = [...new Set(data.map(g => g.season))].sort((a, b) => a - b);
  if (seasons.length < 8) throw new Error("The NFL dataset contains too few complete seasons for a chronological comparison.");
  const validationStart = Math.floor(seasons.length * 0.60);
  const testStart = Math.floor(seasons.length * 0.80);
  return {
    seasons,
    warmup: new Set(seasons.slice(0, validationStart)),
    validation: new Set(seasons.slice(validationStart, testStart)),
    test: new Set(seasons.slice(testStart))
  };
}

function makeTeams(data) {
  return [...new Set(data.flatMap(g => [g.home, g.away]))].sort();
}
function freshGlicko() { return { rating: CENTER, rd: 350, sigma: 0.06 }; }
function freshStats() {
  return { homePF: 0, homePA: 0, homeN: 0, awayPF: 0, awayPA: 0, awayN: 0, neutralMarginSum: 0, marginN: 0 };
}
function recentForm(history) {
  let weighted = 0, weights = 0;
  for (let i = history.length - 1, age = 0; i >= 0; i--, age++) {
    const w = Math.pow(0.78, age);
    weighted += history[i] * w;
    weights += w;
  }
  const mean = weights ? weighted / weights : 0;
  return mean * history.length / (history.length + 4);
}
function mu(ratingValue) { return (ratingValue - CENTER) / SCALE; }
function phi(rd) { return rd / SCALE; }
function rating(m) { return CENTER + SCALE * m; }
function g(phiValue) { return 1 / Math.sqrt(1 + 3 * phiValue * phiValue / (Math.PI * Math.PI)); }

function drawRate(history) {
  return clamp((history.draws + 0.5) / (history.games + 50), 0.002, 0.08);
}
function threeWayFromHomeWin(pHomeGivenNoDraw, pDraw) {
  const p = clamp(pHomeGivenNoDraw, 0.0001, 0.9999), d = clamp(pDraw, 0.002, 0.08);
  return { home: (1 - d) * p, draw: d, away: (1 - d) * (1 - p) };
}
function resultClass(game) { return game.hs > game.as ? "home" : game.hs < game.as ? "away" : "draw"; }

function poissonDistribution(lambda, maxScore) {
  const probabilities = [Math.exp(-lambda)];
  for (let k = 1; k <= maxScore; k++) probabilities[k] = probabilities[k - 1] * lambda / k;
  return probabilities;
}
function poissonOutcome(lambdaHome, lambdaAway) {
  const maxScore = 60;
  const hp = poissonDistribution(lambdaHome, maxScore);
  const ap = poissonDistribution(lambdaAway, maxScore);
  let home = 0, draw = 0, away = 0;
  for (let h = 0; h <= maxScore; h++) {
    for (let a = 0; a <= maxScore; a++) {
      const p = hp[h] * ap[a];
      if (h > a) home += p;
      else if (h === a) draw += p;
      else away += p;
    }
  }
  const total = home + draw + away;
  return { home: home / total, draw: draw / total, away: away / total };
}

function glickoForecast(home, away, pDraw) {
  const p = logistic(g(phi(away.rd)) * (mu(home.rating + 35) - mu(away.rating)));
  return threeWayFromHomeWin(p, pDraw);
}
function recentFormGlickoForecast(home, away, pDraw, homeForm, awayForm) {
  const formAdjustment = clamp((homeForm - awayForm) * 12, -75, 75);
  const p = logistic(g(phi(away.rd)) * (mu(home.rating + 35 + formAdjustment) - mu(away.rating)));
  return threeWayFromHomeWin(p, pDraw);
}
function eloForecast(homeRating, awayRating, pDraw) {
  return threeWayFromHomeWin(1 / (1 + Math.pow(10, (awayRating - (homeRating + 55)) / 400)), pDraw);
}

function updateGlickoPeriod(states, periodGames) {
  const frozen = new Map([...states].map(([team, s]) => [team, { ...s }]));
  const byTeam = new Map([...states.keys()].map(team => [team, []]));
  for (const game of periodGames) {
    byTeam.get(game.home).push({ opponent: game.away, score: game.hs > game.as ? 1 : game.hs < game.as ? 0 : 0.5 });
    byTeam.get(game.away).push({ opponent: game.home, score: game.as > game.hs ? 1 : game.as < game.hs ? 0 : 0.5 });
  }
  for (const [team, old] of frozen) {
    const results = byTeam.get(team);
    const oldMu = mu(old.rating), oldPhi = phi(old.rd);
    if (!results.length) {
      const agedPhi = Math.sqrt(oldPhi * oldPhi + old.sigma * old.sigma);
      states.set(team, { ...old, rd: Math.min(350, SCALE * agedPhi) });
      continue;
    }
    let varianceInv = 0, scoreSum = 0;
    for (const result of results) {
      const opp = frozen.get(result.opponent);
      const oppMu = mu(opp.rating), oppPhi = phi(opp.rd);
      const gj = g(oppPhi), expected = logistic(gj * (oldMu - oppMu));
      varianceInv += gj * gj * expected * (1 - expected);
      scoreSum += gj * (result.score - expected);
    }
    if (!(varianceInv > 0)) continue;
    const v = 1 / varianceInv, delta = v * scoreSum, a = Math.log(old.sigma * old.sigma);
    const f = x => {
      const ex = Math.exp(x), denom = oldPhi * oldPhi + v + ex;
      return ex * (delta * delta - oldPhi * oldPhi - v - ex) / (2 * denom * denom) - (x - a) / (TAU * TAU);
    };
    let A = a, B;
    if (delta * delta > oldPhi * oldPhi + v) B = Math.log(delta * delta - oldPhi * oldPhi - v);
    else {
      let k = 1;
      while (f(a - k * TAU) < 0 && k < 100) k++;
      B = a - k * TAU;
    }
    let fA = f(A), fB = f(B);
    for (let i = 0; i < 100 && Math.abs(B - A) > EPS; i++) {
      const denominator = fB - fA;
      if (Math.abs(denominator) < 1e-15) break;
      const C = A + (A - B) * fA / denominator, fC = f(C);
      if (!Number.isFinite(fC)) break;
      if (fC * fB < 0) { A = B; fA = fB; }
      else fA /= 2;
      B = C; fB = fC;
    }
    const newSigma = Math.exp(A / 2);
    const phiStar = Math.sqrt(oldPhi * oldPhi + newSigma * newSigma);
    const newPhi = 1 / Math.sqrt(1 / (phiStar * phiStar) + 1 / v);
    const newMu = oldMu + newPhi * newPhi * scoreSum;
    states.set(team, { rating: rating(newMu), rd: Math.min(350, SCALE * newPhi), sigma: newSigma });
  }
}

function evaluate(records, splitName) {
  const subset = records.filter(r => r.split === splitName);
  if (!subset.length) return { games: 0, brier: NaN, logLoss: NaN, accuracy: NaN };
  let brier = 0, logLoss = 0, correct = 0;
  for (const r of subset) {
    const actual = r.actual;
    const target = { home: actual === "home" ? 1 : 0, draw: actual === "draw" ? 1 : 0, away: actual === "away" ? 1 : 0 };
    brier += (r.p.home - target.home) ** 2 + (r.p.draw - target.draw) ** 2 + (r.p.away - target.away) ** 2;
    logLoss -= Math.log(Math.max(1e-12, r.p[actual]));
    const predicted = Object.entries(r.p).sort((a, b) => b[1] - a[1])[0][0];
    if (predicted === actual) correct++;
  }
  return { games: subset.length, brier: brier / subset.length, logLoss: logLoss / subset.length, accuracy: correct / subset.length };
}
function fmt(value, digits = 4) { return Number.isFinite(value) ? value.toFixed(digits) : "—"; }

function runExperiment(data) {
  const split = splitSeasons(data), teams = makeTeams(data);
  const glicko = new Map(teams.map(t => [t, freshGlicko()]));
  const elo = new Map(teams.map(t => [t, CENTER]));
  const stats = new Map(teams.map(t => [t, freshStats()]));
  const recentMargins = new Map(teams.map(t => [t, []]));
  const history = { games: 0, draws: 0, homeWins: 0, awayWins: 0, homeScoreSum: 0, awayScoreSum: 0, homeMarginSum: 0 };
  const records = [];
  let lastSeason = null;
  for (let i = 0; i < data.length;) {
    const first = data[i], period = [];
    while (i < data.length && data[i].season === first.season && data[i].week === first.week) period.push(data[i++]);
    if (lastSeason !== null && first.season !== lastSeason) {
      for (const [team, r] of glicko) glicko.set(team, { ...r, rd: Math.min(350, Math.sqrt(r.rd * r.rd + 50 * 50)) });
      for (const [team, r] of elo) elo.set(team, CENTER + 0.95 * (r - CENTER));
      // Recent form is season-specific; don't carry last year's short-term momentum into a new season.
      for (const team of teams) recentMargins.set(team, []);
    }
    lastSeason = first.season;
    const splitName = split.validation.has(first.season) ? "validation" : split.test.has(first.season) ? "test" : "warmup";
    const dRate = drawRate(history);
    const leagueHomeMean = (history.homeScoreSum + 8 * 22.5) / (history.games + 8);
    const leagueAwayMean = (history.awayScoreSum + 8 * 21.5) / (history.games + 8);
    const homeAdv = history.games ? history.homeMarginSum / history.games : 2.5;
    for (const game of period) {
      const gh = glicko.get(game.home), ga = glicko.get(game.away);
      const eh = elo.get(game.home), ea = elo.get(game.away);
      const sh = stats.get(game.home), sa = stats.get(game.away);
      const shrink = 8;
      const homeOff = (sh.homePF + shrink * leagueHomeMean) / (sh.homeN + shrink);
      const awayDef = (sa.awayPA + shrink * leagueHomeMean) / (sa.awayN + shrink);
      const awayOff = (sa.awayPF + shrink * leagueAwayMean) / (sa.awayN + shrink);
      const homeDef = (sh.homePA + shrink * leagueAwayMean) / (sh.homeN + shrink);
      const lambdaHome = clamp((homeOff + awayDef) / 2, 3, 50);
      const lambdaAway = clamp((awayOff + homeDef) / 2, 3, 50);
      const homeStrength = sh.marginN ? sh.neutralMarginSum / (sh.marginN + 5) : 0;
      const awayStrength = sa.marginN ? sa.neutralMarginSum / (sa.marginN + 5) : 0;
      const expectedMargin = homeStrength - awayStrength + (Number.isFinite(homeAdv) ? homeAdv : 2.5);
      const pMargin = logistic(expectedMargin / 6.5);
      const baselineHome = (history.homeWins + 28.5) / (history.homeWins + history.awayWins + 50);
      const p = {
        "Glicko-2": glickoForecast(gh, ga, dRate),
        "Glicko-2 + recent form": recentFormGlickoForecast(gh, ga, dRate, recentForm(recentMargins.get(game.home)), recentForm(recentMargins.get(game.away))),
        "Elo": eloForecast(eh, ea, dRate),
        "Poisson scores": poissonOutcome(lambdaHome, lambdaAway),
        "Points differential": threeWayFromHomeWin(pMargin, dRate),
        "Home-rate baseline": threeWayFromHomeWin(baselineHome, dRate)
      };
      if (splitName !== "warmup") {
        for (const name of MODEL_NAMES) records.push({ model: name, split: splitName, p: p[name], actual: resultClass(game), season: game.season, week: game.week });
      }
    }
    // Update only after every game in this week has been forecast.
    updateGlickoPeriod(glicko, period);
    const frozenElo = new Map(elo);
    const eloDeltas = new Map(teams.map(t => [t, 0]));
    for (const game of period) {
      const expected = 1 / (1 + Math.pow(10, ((frozenElo.get(game.away) - frozenElo.get(game.home) - 55) / 400)));
      const actual = game.hs > game.as ? 1 : game.hs < game.as ? 0 : 0.5;
      const change = 20 * (actual - expected);
      eloDeltas.set(game.home, eloDeltas.get(game.home) + change);
      eloDeltas.set(game.away, eloDeltas.get(game.away) - change);
    }
    for (const team of teams) elo.set(team, frozenElo.get(team) + eloDeltas.get(team));
    for (const game of period) {
      const sh = stats.get(game.home), sa = stats.get(game.away);
      sh.homePF += game.hs; sh.homePA += game.as; sh.homeN++;
      sa.awayPF += game.as; sa.awayPA += game.hs; sa.awayN++;
      sh.neutralMarginSum += game.hs - game.as - 2.5; sh.marginN++;
      sa.neutralMarginSum += game.as - game.hs + 2.5; sa.marginN++;
      const homeRecent = recentMargins.get(game.home), awayRecent = recentMargins.get(game.away);
      homeRecent.push(game.hs - game.as - 2.5);
      awayRecent.push(game.as - game.hs + 2.5);
      if (homeRecent.length > 12) homeRecent.shift();
      if (awayRecent.length > 12) awayRecent.shift();
      history.games++;
      history.homeScoreSum += game.hs; history.awayScoreSum += game.as;
      history.homeMarginSum += game.hs - game.as;
      if (game.hs > game.as) history.homeWins++;
      else if (game.hs < game.as) history.awayWins++;
      else history.draws++;
    }
  }
  const result = {};
  for (const name of MODEL_NAMES) {
    result[name] = {
      validation: evaluate(records.filter(r => r.model === name), "validation"),
      test: evaluate(records.filter(r => r.model === name), "test")
    };
  }
  return { result, records, split, data };
}

function renderResults(output) {
  const tbody = $("resultsBody");
  const rows = MODEL_NAMES.map(name => ({ name, ...output.result[name] }));
  rows.sort((a, b) => a.test.logLoss - b.test.logLoss);
  const best = rows[0]?.name;
  tbody.innerHTML = rows.map(r => `<tr class="${r.name === best ? "best" : ""}">
    <td>${r.name}</td>
    <td>${fmt(r.validation.brier)}</td><td>${fmt(r.validation.logLoss)}</td><td>${fmt(r.validation.accuracy * 100, 1)}%</td>
    <td>${fmt(r.test.brier)}</td><td>${fmt(r.test.logLoss)}</td><td>${fmt(r.test.accuracy * 100, 1)}%</td><td>${r.test.games.toLocaleString()}</td>
  </tr>`).join("");
  const v = output.result[best];
  $("verdict").textContent = `Lowest final-test log loss in this run: ${best} (log loss ${fmt(v.test.logLoss)}, Brier ${fmt(v.test.brier)}, accuracy ${fmt(v.test.accuracy * 100, 1)}%). This is a result for the held-out seasons, not proof that the method is universally best.`;
  const select = $("calibrationModel");
  select.innerHTML = MODEL_NAMES.map(name => `<option value="${name}">${name}</option>`).join("");
  select.disabled = false;
  select.value = best;
  lastResultRows = output.records;
  renderCalibration();
}

function renderCalibration() {
  const name = $("calibrationModel").value;
  const rows = lastResultRows.filter(r => r.model === name && r.split === "test");
  if (!rows.length) {
    $("calibrationBody").innerHTML = '<tr><td colspan="4" class="empty">No final-test predictions available.</td></tr>';
    return;
  }
  const bins = Array.from({ length: 5 }, (_, i) => ({ lo: i * 0.2, hi: (i + 1) * 0.2, sum: 0, wins: 0, n: 0 }));
  for (const r of rows) {
    const bin = bins[Math.min(4, Math.floor(r.p.home / 0.2))];
    bin.sum += r.p.home; bin.wins += r.actual === "home" ? 1 : 0; bin.n++;
  }
  $("calibrationBody").innerHTML = bins.filter(b => b.n).map(b => `<tr><td>${b.lo.toFixed(1)}–${b.hi.toFixed(1)}</td><td>${(b.sum / b.n).toFixed(3)}</td><td>${(b.wins / b.n).toFixed(3)}</td><td>${b.n}</td></tr>`).join("");
}

async function loadData() {
  $("status").textContent = "LOADING DATA";
  $("progressText").textContent = "Fetching the public nflverse game-results CSV…";
  try {
    const response = await fetch(DATA_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`Data request failed (HTTP ${response.status}).`);
    games = parseGames(await response.text());
    if (!games.length) throw new Error("The data loaded, but no completed regular-season games were recognised.");
    const split = splitSeasons(games);
    $("gamesMetric").textContent = games.length.toLocaleString();
    $("seasonsMetric").textContent = `${split.seasons[0]}–${split.seasons.at(-1)} · ${split.seasons.length} seasons`;
    $("validationMetric").textContent = `${Math.min(...split.validation)}–${Math.max(...split.validation)}`;
    $("testMetric").textContent = `${Math.min(...split.test)}–${Math.max(...split.test)}`;
    $("status").textContent = "DATA READY";
    $("runBtn").disabled = false;
    $("progressText").textContent = "Historical scores loaded. Ready to run the comparison.";
    $("resultsBody").innerHTML = '<tr><td colspan="8" class="empty">Data loaded. Run the backtest to calculate predictions.</td></tr>';
  } catch (error) {
    $("status").textContent = "DATA ERROR";
    $("progressText").textContent = error.message;
    $("resultsBody").innerHTML = `<tr><td colspan="8" class="empty">${error.message} Check the data source link below and reload.</td></tr>`;
    $("dataWarning").textContent = "The live dataset could not be fetched by this browser session. No synthetic NFL results are substituted.";
  }
}

async function runBacktest() {
  if (busy || !games.length) return;
  busy = true;
  $("runBtn").disabled = true;
  $("runStatus").textContent = "RUNNING";
  $("progressText").textContent = "Replaying seasons chronologically. Every week is forecast before its results are applied.";
  try {
    await new Promise(resolve => requestAnimationFrame(resolve));
    const output = runExperiment(games);
    renderResults(output);
    $("runStatus").textContent = "COMPLETE";
    $("progressText").textContent = `Compared ${MODEL_NAMES.length} methods across ${output.result[MODEL_NAMES[0]].test.games.toLocaleString()} final-test games.`;
  } catch (error) {
    $("runStatus").textContent = "ERROR";
    $("progressText").textContent = error.message;
    $("verdict").textContent = "The backtest stopped before producing a valid comparison. See the status message and reload after correcting the issue.";
    console.error(error);
  } finally {
    busy = false;
    $("runBtn").disabled = false;
  }
}

$("runBtn").addEventListener("click", runBacktest);
$("calibrationModel").addEventListener("change", renderCalibration);
loadData();
