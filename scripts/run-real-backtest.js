#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const DATA_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv";
const source = fs.readFileSync(path.join(__dirname, "..", "historical-tester.js"), "utf8");
const browserStart = source.indexOf('$("runBtn").addEventListener("click", runBacktest);');
if (browserStart < 0) throw new Error("Could not locate the browser startup boundary in historical-tester.js.");

const core = new Function(source.slice(0, browserStart) + `
return { parseGames, runExperiment, BASE_MODEL_NAMES, CALIBRATED_NAMES, MODEL_NAMES };
`)();
const fmt = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : "—";

async function main() {
  console.log("Fetching historical games from nflverse…");
  const response = await fetch(DATA_URL, { headers: { "user-agent": "NFL-Prediction-Lab-CI" } });
  if (!response.ok) throw new Error(`nflverse CSV request failed: HTTP ${response.status}`);
  const csv = await response.text();
  const games = core.parseGames(csv);
  if (games.length < 1000) throw new Error(`Expected at least 1,000 completed regular-season games; received ${games.length}.`);

  console.log(`Loaded ${games.length.toLocaleString()} completed regular-season games (${games[0].season}–${games[games.length - 1].season}).`);
  const start = Date.now();
  const output = core.runExperiment(games);
  const seconds = ((Date.now() - start) / 1000).toFixed(1);
  const split = output.split;
  const selected = core.MODEL_NAMES
    .map(name => ({ name, ...output.result[name] }))
    .sort((a, b) => {
      if (Math.abs(a.validation.logLoss - b.validation.logLoss) < 1e-12) {
        const ac = a.name.endsWith(" · temperature-calibrated");
        const bc = b.name.endsWith(" · temperature-calibrated");
        if (ac !== bc) return ac ? 1 : -1;
      }
      return a.validation.logLoss - b.validation.logLoss;
    })[0];

  console.log(`Seasons: ${split.seasons[0]}–${split.seasons.at(-1)} (${split.seasons.length} complete seasons)`);
  console.log(`Model training: ${Math.min(...split.modelTraining)}–${Math.max(...split.modelTraining)} | Calibration holdout: ${Math.min(...split.calibration)}–${Math.max(...split.calibration)} | Validation: ${Math.min(...split.validation)}–${Math.max(...split.validation)} | Final test: ${Math.min(...split.test)}–${Math.max(...split.test)}`);
  console.log(`Backtest completed in ${seconds}s. Model selection uses validation log loss only.`);
  console.log("");
  console.log(["Model", "Val Brier", "Val LogLoss", "Val Acc", "Test Brier", "Test LogLoss", "Test Acc", "Temperature"].join("\t"));
  for (const name of core.MODEL_NAMES) {
    const r = output.result[name];
    const baseName = name.replace(" · temperature-calibrated", "");
    console.log([
      name, fmt(r.validation.brier), fmt(r.validation.logLoss), fmt(r.validation.accuracy * 100, 1) + "%",
      fmt(r.test.brier), fmt(r.test.logLoss), fmt(r.test.accuracy * 100, 1) + "%",
      fmt(output.temperatures[baseName], 3)
    ].join("\t"));
  }

  console.log("");
  console.log(`Validation leader: ${selected.name} (validation log loss ${fmt(selected.validation.logLoss)}).`);
  console.log(`Frozen held-out test: log loss ${fmt(selected.test.logLoss)}, Brier ${fmt(selected.test.brier)}, accuracy ${fmt(selected.test.accuracy * 100, 1)}%.`);
  console.log("Do not choose a model or settings from the final-test results.");
  console.log("");
  console.log("HOME-WIN CALIBRATION BY MODEL (held-out test)");
  console.log(["Model", "Predicted bin", "Mean predicted", "Observed home wins", "Games"].join("\t"));
  for (const name of core.MODEL_NAMES) {
    const rows = output.records.filter(row => row.model === name && row.split === "test");
    const bins = Array.from({ length: 5 }, (_, i) => ({ lo: i * 0.2, hi: (i + 1) * 0.2, sum: 0, wins: 0, n: 0 }));
    for (const row of rows) {
      const bin = bins[Math.min(4, Math.floor(row.p.home / 0.2))];
      bin.sum += row.p.home;
      bin.wins += row.actual === "home" ? 1 : 0;
      bin.n++;
    }
    for (const bin of bins) {
      console.log([
        name, `${bin.lo.toFixed(1)}–${bin.hi.toFixed(1)}`,
        bin.n ? fmt(bin.sum / bin.n, 3) : "—",
        bin.n ? fmt(bin.wins / bin.n, 3) : "—",
        bin.n
      ].join("\t"));
    }
  }
}

main().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
