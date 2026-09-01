//! Recompute the PPO kernels in Rust, by a different formulation than the
//! Python and the C use.
//!
//! rlhf/ppo.py computes GAE as a backward recursion, and verify/kernels.c
//! writes the same recursion again. Two implementations of one recursion do
//! not catch an error in the recursion itself. This uses the closed form
//!
//!     A_t = sum_{l=0}^{T-1-t} (gamma * lam)^l * delta_{t+l}
//!
//! which is the definition GAE is derived from, evaluated forward with no
//! carried state. If the recursion in ppo.py had an off by one in the value
//! bootstrap, this would not share it.
//!
//! Then, because Rust can afford what Python could not, it runs the two
//! formulations against each other on a large number of random sequences,
//! with a small xorshift generator so nothing outside the standard library is
//! needed. The golden vectors cover eight sequences; this covers the shape of
//! the input space around them.

use std::env;
use std::fs;
use std::process::exit;

const TOL: f64 = 1e-12;
const RANDOM_CASES: usize = 200_000;

/// xorshift64*, so the random cross check needs no crate and no rand.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform on [-1, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

struct Table {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn load(path: &str) -> Table {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| { eprintln!("cannot read {path}: {e}"); exit(2) });
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header: Vec<String> = lines
            .next()
            .unwrap_or_else(|| { eprintln!("{path} is empty"); exit(2) })
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        let rows: Vec<Vec<String>> = lines
            .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
            .collect();
        if rows.is_empty() {
            eprintln!("{path} has a header and no rows");
            exit(2);
        }
        Table { header, rows }
    }

    /// Columns are resolved by name so an upstream column cannot shift what
    /// this reads.
    fn col(&self, name: &str) -> usize {
        match self.header.iter().position(|h| h == name) {
            Some(i) => i,
            None => { eprintln!("no column {name}"); exit(2) }
        }
    }

    fn num(&self, row: usize, col: usize) -> f64 {
        self.rows[row][col].parse().unwrap_or_else(|_| {
            eprintln!("row {} column {} is not a number: {:?}",
                      row + 2, self.header[col], self.rows[row][col]);
            exit(2)
        })
    }
}

/// GAE by the closed form, no carried state between timesteps.
fn gae_closed_form(rewards: &[f64], values: &[f64], gamma: f64, lam: f64) -> Vec<f64> {
    let t = rewards.len();
    let delta: Vec<f64> = (0..t)
        .map(|i| {
            let next_v = if i + 1 < t { values[i + 1] } else { 0.0 };
            rewards[i] + gamma * next_v - values[i]
        })
        .collect();
    (0..t)
        .map(|i| {
            let mut acc = 0.0;
            let mut w = 1.0;
            for d in delta.iter().skip(i) {
                acc += w * d;
                w *= gamma * lam;
            }
            acc
        })
        .collect()
}

/// The backward recursion, kept only to be compared against the closed form.
fn gae_recursive(rewards: &[f64], values: &[f64], gamma: f64, lam: f64) -> Vec<f64> {
    let t = rewards.len();
    let mut adv = vec![0.0; t];
    let mut last = 0.0;
    for i in (0..t).rev() {
        let next_v = if i + 1 < t { values[i + 1] } else { 0.0 };
        let delta = rewards[i] + gamma * next_v - values[i];
        last = delta + gamma * lam * last;
        adv[i] = last;
    }
    adv
}

fn check_gae(root: &str) -> usize {
    let t = Table::load(&format!("{root}/verify/golden/gae.csv"));
    let (c_case, c_gamma, c_lam, c_beta) =
        (t.col("case"), t.col("gamma"), t.col("lam"), t.col("kl_beta"));
    let (c_score, c_t, c_kl) = (t.col("score"), t.col("t"), t.col("kl_tok"));
    let (c_reward, c_value, c_adv, c_ret) =
        (t.col("reward"), t.col("value"), t.col("adv"), t.col("ret"));

    let mut bad = 0usize;
    let mut worst: f64 = 0.0;
    let mut cases = 0usize;
    let mut start = 0usize;
    while start < t.rows.len() {
        let case = t.rows[start][c_case].clone();
        let mut end = start;
        while end < t.rows.len() && t.rows[end][c_case] == case {
            end += 1;
        }
        let n = end - start;
        let (gamma, lam) = (t.num(start, c_gamma), t.num(start, c_lam));
        let (beta, score) = (t.num(start, c_beta), t.num(start, c_score));

        let mut kl = vec![0.0; n];
        let (mut want_r, mut want_a, mut want_ret) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut values = vec![0.0; n];
        for r in start..end {
            let i = t.num(r, c_t) as usize;
            if i >= n {
                eprintln!("case {case}: step index {i} outside a run of {n} rows");
                exit(2);
            }
            kl[i] = t.num(r, c_kl);
            values[i] = t.num(r, c_value);
            want_r[i] = t.num(r, c_reward);
            want_a[i] = t.num(r, c_adv);
            want_ret[i] = t.num(r, c_ret);
        }

        // the reward model score lands on the final token only
        let rewards: Vec<f64> = (0..n)
            .map(|i| -beta * kl[i] + if i + 1 == n { score } else { 0.0 })
            .collect();
        let adv = gae_closed_form(&rewards, &values, gamma, lam);

        for i in 0..n {
            for (what, got, want) in [
                ("reward", rewards[i], want_r[i]),
                ("adv", adv[i], want_a[i]),
                ("ret", adv[i] + values[i], want_ret[i]),
            ] {
                let d = (got - want).abs();
                worst = worst.max(d);
                if d > TOL {
                    println!("  FAIL case {case} t {i} {what}: rust {got:.17} golden {want:.17} |d| {d:.3e}");
                    bad += 1;
                }
            }
        }
        println!("  case {case}  T {n:2}  gamma {gamma:.4} lam {lam:.4} beta {beta:.4}   ok");
        cases += 1;
        start = end;
    }
    println!("GAE by the closed form: {cases} cases, {bad} disagreements, worst |d| {worst:.1e}");
    bad
}

fn check_losses(root: &str) -> usize {
    let inp = Table::load(&format!("{root}/verify/golden/ppo_inputs.csv"));
    let out = Table::load(&format!("{root}/verify/golden/ppo_loss.csv"));
    let (i_case, i_i) = (inp.col("case"), inp.col("i"));
    let (i_lp, i_olp, i_adv) = (inp.col("lp"), inp.col("old_lp"), inp.col("adv"));
    let (i_v, i_ov, i_ret) = (inp.col("v"), inp.col("old_v"), inp.col("ret"));
    let (o_case, o_clip, o_vclip, o_n) =
        (out.col("case"), out.col("clip"), out.col("vf_clip"), out.col("n"));
    let (o_frac, o_pg, o_vf) =
        (out.col("frac_ratio_clipped"), out.col("pg_loss"), out.col("vf_loss"));

    let mut bad = 0usize;
    let mut worst: f64 = 0.0;
    for r in 0..out.rows.len() {
        let case = out.rows[r][o_case].clone();
        let (clip, vclip) = (out.num(r, o_clip), out.num(r, o_vclip));
        let n = out.num(r, o_n) as usize;

        let mut rows: Vec<usize> = (0..inp.rows.len())
            .filter(|&k| inp.rows[k][i_case] == case)
            .collect();
        rows.sort_by_key(|&k| inp.num(k, i_i) as usize);
        if rows.len() != n {
            println!("  FAIL case {case}: ppo_loss.csv says n={n}, ppo_inputs.csv has {}",
                     rows.len());
            bad += 1;
            continue;
        }

        let (mut pg, mut vf) = (0.0, 0.0);
        let mut clipped = 0usize;
        for &k in &rows {
            let ratio = (inp.num(k, i_lp) - inp.num(k, i_olp)).exp();
            let a = inp.num(k, i_adv);
            // written as a branch rather than min(ratio*a, clamp(ratio)*a), so
            // a sign error in the Python clamp would not be reproduced here
            let unclipped = ratio * a;
            let surrogate = if ratio < 1.0 - clip {
                let alt = (1.0 - clip) * a;
                if alt < unclipped { alt } else { unclipped }
            } else if ratio > 1.0 + clip {
                let alt = (1.0 + clip) * a;
                if alt < unclipped { alt } else { unclipped }
            } else {
                unclipped
            };
            pg -= surrogate;

            let (v, ov, ret) = (inp.num(k, i_v), inp.num(k, i_ov), inp.num(k, i_ret));
            let dv = (v - ov).clamp(-vclip, vclip);
            let e1 = (v - ret) * (v - ret);
            let e2 = (ov + dv - ret) * (ov + dv - ret);
            vf += 0.5 * e1.max(e2);
            if !(1.0 - clip..=1.0 + clip).contains(&ratio) {
                clipped += 1;
            }
        }
        pg /= n as f64;
        vf /= n as f64;

        for (what, got, want) in [
            ("pg_loss", pg, out.num(r, o_pg)),
            ("vf_loss", vf, out.num(r, o_vf)),
            ("frac_ratio_clipped", clipped as f64 / n as f64, out.num(r, o_frac)),
        ] {
            let d = (got - want).abs();
            worst = worst.max(d);
            if d > TOL {
                println!("  FAIL case {case} {what}: rust {got:.17} golden {want:.17} |d| {d:.3e}");
                bad += 1;
            }
        }
        println!("  case {case}  n {n:3}  clip {clip:.2}  {clipped} of {n} ratios clipped   pg {pg:+.12}  vf {vf:.12}");
    }
    println!("clipped surrogate: {} cases, {bad} disagreements, worst |d| {worst:.1e}",
             out.rows.len());
    bad
}

/// The part Python could not afford: the two formulations of GAE on a large
/// random sample of sequences, including the corners lam=0 and lam=1.
fn random_cross_check() -> usize {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut worst: f64 = 0.0;
    let mut bad = 0usize;
    for case in 0..RANDOM_CASES {
        let t = 1 + (rng.next_u64() % 40) as usize;
        let gamma = match case % 4 {
            0 => 1.0,
            1 => 0.99,
            2 => 0.5 + 0.5 * (rng.unit() + 1.0) / 2.0,
            _ => 1.0,
        };
        let lam = match case % 5 {
            0 => 1.0,
            1 => 0.0,
            _ => (rng.unit() + 1.0) / 2.0,
        };
        let values: Vec<f64> = (0..t).map(|_| rng.unit() * 3.0).collect();
        let mut rewards: Vec<f64> = (0..t).map(|_| rng.unit() * 0.1).collect();
        rewards[t - 1] += rng.unit() * 10.0;

        let a = gae_recursive(&rewards, &values, gamma, lam);
        let b = gae_closed_form(&rewards, &values, gamma, lam);
        for i in 0..t {
            // relative, because a long sequence with gamma=lam=1 accumulates a
            // large advantage and an absolute tolerance would be meaningless
            let scale = a[i].abs().max(1.0);
            let d = (a[i] - b[i]).abs() / scale;
            worst = worst.max(d);
            if d > 1e-11 {
                println!("  FAIL random case {case} t {i}: {} vs {}", a[i], b[i]);
                bad += 1;
            }
        }
    }
    println!("recursion against closed form on {RANDOM_CASES} random sequences: \
              worst relative |d| {worst:.1e}, {bad} disagreements");
    bad
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let bad = check_gae(&root) + check_losses(&root) + random_cross_check();
    if bad > 0 {
        println!("\n{bad} values disagree");
        exit(1);
    }
    println!("\nRust reproduces every golden value and the two GAE formulations agree");
}
