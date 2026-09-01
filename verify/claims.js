// Check the README claims that are written as words rather than as digits.
//
// scripts/check_numbers.py says of itself that it "checks quoted figures
// against results/, not claims written in words". So a sentence could name the
// right numbers and still say something false about them: that one method is
// below another, that a series rises, that a value is the largest here. One of
// those sentences was in fact wrong, which is why this file exists.
//
// Every comparison below is recomputed from results/methods.csv, and every
// sentence it belongs to has to still be in README.md. Deleting the sentence
// fails the check rather than passing it by default.

const fs = require("fs");
const path = require("path");

const root = process.argv[2] || ".";
const rows = parseCSV(path.join(root, "results", "methods.csv"));
// the README writes negatives with a unicode minus; normalise before matching
const readme = fs
  .readFileSync(path.join(root, "README.md"), "utf8")
  .replace(/−/g, "-")
  .replace(/\*/g, "")
  .replace(/\s+/g, " ");

function parseCSV(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split("\n");
  const header = lines[0].split(",").map((s) => s.trim());
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    if (cells.length !== header.length) {
      throw new Error(`${file}: a row has ${cells.length} cells, not ${header.length}`);
    }
    return Object.fromEntries(header.map((h, i) => [h, cells[i].trim()]));
  });
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) throw new Error("median of nothing");
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

function med(method, col) {
  const vals = rows.filter((r) => r.method === method).map((r) => Number(r[col]));
  if (vals.length === 0) throw new Error(`no rows for ${method}`);
  if (vals.some(Number.isNaN)) throw new Error(`${method} has a non-numeric ${col}`);
  return median(vals);
}

let failures = 0;
function check(label, ok, detail) {
  console.log(`  ${ok ? "ok  " : "FAIL"} ${label}${detail ? "   " + detail : ""}`);
  if (!ok) failures++;
}

// A claim is only checked if the sentence making it is still there. Otherwise a
// deleted sentence would look like a passing check.
function says(fragment) {
  const ok = readme.includes(fragment.replace(/\s+/g, " "));
  if (!ok) {
    console.log(`  FAIL README no longer contains: ${fragment}`);
    failures++;
  }
  return ok;
}

const PPO_WITH_PENALTY = ["PPO (beta=0.2)", "PPO (beta=0.05)", "PPO (beta=0.01)"];

console.log("Best-of-N, closed form KL");
// KL(best-of-N || reference) = log N - (N-1)/N, which is why Best-of-N cannot
// travel: the number in results/ should be the formula, not a measurement.
for (const n of [4, 16, 64]) {
  const want = Math.log(n) - (n - 1) / n;
  const got = med(`Best-of-${n}`, "kl");
  check(`N=${n} KL is log N - (N-1)/N`, Math.abs(got - want) < 1e-12,
        `results ${got.toFixed(12)}  formula ${want.toFixed(12)}`);
}

console.log("\nthe Best-of-N sentence");
if (says("at N=64 the proxy reads +9.507 while the gold is still negative at -0.054, " +
         "better than the reference it started from but short of every PPO run with " +
         "the penalty on, because the closed form pins its KL at 3.17")) {
  const bon = med("Best-of-64", "gold");
  const ref = med("SFT (reference)", "gold");
  const worstPPO = Math.min(...PPO_WITH_PENALTY.map((m) => med(m, "gold")));
  check("Best-of-64 gold is negative", bon < 0, `gold ${bon.toFixed(3)}`);
  check("Best-of-64 gold beats the reference", bon > ref,
        `${bon.toFixed(3)} against ${ref.toFixed(3)}`);
  check("Best-of-64 gold is short of every PPO run with a penalty", bon < worstPPO,
        `${bon.toFixed(3)} against ${worstPPO.toFixed(3)}`);
  check("Best-of-64 proxy rounds to 9.507", med("Best-of-64", "proxy").toFixed(3) === "9.507",
        `proxy ${med("Best-of-64", "proxy").toFixed(3)}`);
  check("Best-of-64 KL rounds to 3.17", med("Best-of-64", "kl").toFixed(2) === "3.17");
}

console.log("\nthe DPO sentence");
if (says("DPO takes the best gold of anything here, +2.125")) {
  const methods = [...new Set(rows.map((r) => r.method))];
  const best = methods.reduce((a, b) => (med(a, "gold") >= med(b, "gold") ? a : b));
  check("DPO has the largest gold of any method", best === "DPO", `largest is ${best}`);
  check("DPO gold rounds to 2.125", med("DPO", "gold").toFixed(3) === "2.125",
        `gold ${med("DPO", "gold").toFixed(3)}`);
}

console.log("\nthe RLOO and GRPO sentence");
if (says("RLOO and GRPO both stop near +0.60 gold at a KL around 6, " +
         "short of the +1.320 PPO reaches")) {
  for (const m of ["RLOO", "GRPO"]) {
    const g = med(m, "gold");
    const k = med(m, "kl");
    check(`${m} gold is near 0.60`, Math.abs(g - 0.6) < 0.05, `gold ${g.toFixed(3)}`);
    check(`${m} KL is around 6`, Math.abs(k - 6) < 1.5, `KL ${k.toFixed(2)}`);
    check(`${m} falls short of PPO beta=0.01`, g < med("PPO (beta=0.01)", "gold"),
          `${g.toFixed(3)} against ${med("PPO (beta=0.01)", "gold").toFixed(3)}`);
  }
}

console.log("\nthe monotone proxy sentence");
if (says("The proxy rises monotonically the whole way, from +9.004 to +9.966")) {
  const sweep = ["PPO (beta=0.2)", "PPO (beta=0.05)", "PPO (beta=0.01)", "PPO (beta=0.0)"];
  const proxies = sweep.map((m) => med(m, "proxy"));
  const rising = proxies.every((v, i) => i === 0 || v > proxies[i - 1]);
  check("proxy rises at every step of the sweep", rising,
        proxies.map((v) => v.toFixed(3)).join(" -> "));
  check("the sweep runs from 9.004 to 9.966",
        proxies[0].toFixed(3) === "9.004" && proxies[3].toFixed(3) === "9.966");
  // the same sweep on gold has to do the opposite, or there is no result here
  const golds = sweep.map((m) => med(m, "gold"));
  check("gold peaks inside the sweep and then falls",
        Math.max(...golds) === golds[2] && golds[3] < golds[0],
        golds.map((v) => v.toFixed(3)).join(" -> "));
}

console.log("\nthe collapse sentence");
if (says("The gold peaks at +1.320 and then collapses to -1.426, which is worse " +
         "than the reference policy it started from")) {
  check("the collapsed run is below the reference",
        med("PPO (beta=0.0)", "gold") < med("SFT (reference)", "gold"),
        `${med("PPO (beta=0.0)", "gold").toFixed(3)} against ` +
        `${med("SFT (reference)", "gold").toFixed(3)}`);
}

if (failures > 0) {
  console.log(`\n${failures} claims in README.md are not supported by results/methods.csv`);
  process.exit(1);
}
console.log("\nJavaScript: every claim written in words holds against results/methods.csv");
