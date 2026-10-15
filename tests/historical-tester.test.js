const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const source = fs.readFileSync(path.join(__dirname, "..", "historical-tester.js"), "utf8");
const browserStart = source.indexOf('$("runBtn").addEventListener("click", runBacktest);');
assert.notEqual(browserStart, -1, "browser startup marker must exist");
const core = new Function(source.slice(0, browserStart) + `
return { runExperiment, validateGames, assertProbabilityVector, BASE_MODEL_NAMES, CALIBRATED_NAMES };
`)();
const { runExperiment, validateGames, assertProbabilityVector, BASE_MODEL_NAMES, CALIBRATED_NAMES, ENSEMBLE_NAMES } = core;

function makeFixture() {
  const teams = Array.from({ length: 16 }, (_, i) => "T" + String(i + 1).padStart(2, "0"));
  const games = [];
  for (let season = 2010; season < 2020; season++) {
    for (let week = 1; week <= 18; week++) {
      for (let pair = 0; pair < 8; pair++) {
        const hi = (pair * 2 + week - 1) % 16;
        const ai = (hi + 1 + (Math.floor((week - 1) / 2) * 2) % 15) % 16;
        const home = teams[hi], away = teams[ai];
        const hs = Math.max(0, 22 + 2 * ((hi % 5) - 2) - ((ai % 5) - 2) + (week % 4));
        const as = Math.max(0, 20 + 2 * ((ai % 5) - 2) - ((hi % 5) - 2) + ((week + pair) % 3));
        games.push({
          season, week, home, away, hs, as,
          date: `${season}-09-${String(Math.min(28, week + 1)).padStart(2, "0")}`,
          gameType: "REG", scoresPresent: true
        });
      }
    }
  }
  return games;
}

test("dataset integrity rejects duplicate games", () => {
  const games = makeFixture();
  assert.throws(() => validateGames([...games, { ...games[0] }]), /Duplicate game detected/);
});

test("probability integrity rejects invalid vectors", () => {
  assert.throws(
    () => assertProbabilityVector({ home: 0.8, draw: 0.3, away: 0.1 }, "fixture", { season: 2025, week: 1, home: "A", away: "B" }),
    /Invalid probability vector/
  );
});

test("all models complete chronological validation and test with valid probabilities", { timeout: 30000 }, () => {
  const games = makeFixture();
  const output = runExperiment(games);
  assert.equal(BASE_MODEL_NAMES.length, 9);
  assert.equal(CALIBRATED_NAMES.length, BASE_MODEL_NAMES.length);

  for (const name of BASE_MODEL_NAMES) {
    const validation = output.records.filter(row => row.model === name && row.split === "validation");
    const heldOut = output.records.filter(row => row.model === name && row.split === "test");
    assert.equal(validation.length, output.result[name].validation.games, name + " validation coverage");
    assert.equal(heldOut.length, output.result[name].test.games, name + " test coverage");
    assert.ok(validation.length > 0, name + " validation predictions exist");
    assert.ok(heldOut.length > 0, name + " test predictions exist");
    for (const row of [...validation, ...heldOut]) {
      assert.ok(Math.abs(row.p.home + row.p.draw + row.p.away - 1) < 1e-8, name + " probabilities sum to one");
      assert.ok([row.p.home, row.p.draw, row.p.away].every(p => Number.isFinite(p) && p >= 0 && p <= 1), name + " probabilities are valid");
    }
    assert.ok(Number.isFinite(output.result[name].validation.logLoss), name + " validation log loss");
    assert.ok(Number.isFinite(output.result[name].test.logLoss), name + " test log loss");
  }

  for (const name of CALIBRATED_NAMES) {
    assert.ok(output.result[name].validation.games > 0, name + " has independent validation predictions");
    assert.ok(Number.isFinite(output.result[name].validation.logLoss), name + " validation log loss");
    assert.ok(Number.isFinite(output.result[name].test.logLoss), name + " held-out test log loss");
  }
  for (const name of BASE_MODEL_NAMES) {
    assert.ok(output.records.some(row => row.model === name && row.split === "calibration"), name + " has a separate warm-up calibration holdout");
  }

  assert.equal(ENSEMBLE_NAMES.length, 1);
  const ensembleName = ENSEMBLE_NAMES[0];
  assert.ok(output.ensembleWeights[ensembleName] >= 0 && output.ensembleWeights[ensembleName] <= 1, "ensemble weight is bounded");
  for (const splitName of ["validation", "test"]) {
    const ensembleRows = output.records.filter(row => row.model === ensembleName && row.split === splitName);
    assert.equal(ensembleRows.length, output.result[ensembleName][splitName].games, "ensemble " + splitName + " coverage");
    assert.ok(Number.isFinite(output.result[ensembleName][splitName].logLoss), "ensemble " + splitName + " log loss");
    for (const row of ensembleRows) {
      assert.ok(Math.abs(row.p.home + row.p.draw + row.p.away - 1) < 1e-8, "ensemble probabilities sum to one");
    }
  }
});
