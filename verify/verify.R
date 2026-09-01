# Statistical check of the claims the README makes in words rather than digits.
#
# scripts/check_numbers.py says of itself that it "checks quoted figures against
# results/, not claims written in words". Those claims are the argument of the
# repository, so they are the ones worth testing:
#
#   the proxy rises the whole way            monotone in KL, positive slope
#   the gold peaks and then collapses        a negative quadratic term with the
#                                            turning point inside the observed
#                                            range of drift
#   agreement is 0.68 to 0.76 near the       recomputed from the curve, with the
#   reference and drops as low as 0.41       claimed bounds read out of README.md
#
# The turnover is also given a cluster bootstrap over the twelve runs, because
# checkpoints inside one run are not independent draws and treating 180 of them
# as if they were would make any interval here far too narrow.
#
# Base R only, so CI needs nothing beyond r-base-core.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

DRAWS <- 20000
curve <- read.csv(file.path(root, "results", "overopt-curve.csv"))
readme <- paste(readLines(file.path(root, "README.md"), warn = FALSE), collapse = " ")
failures <- 0

# A file that failed to parse cleanly would otherwise turn up much later as an
# NA inside a comparison, with an error that says nothing useful.
needed <- c("beta", "seed", "step", "kl", "gold", "proxy", "rm_agreement")
missing <- setdiff(needed, names(curve))
if (length(missing) > 0) {
    cat(sprintf("  FAIL: results/overopt-curve.csv has no column %s\n",
                paste(missing, collapse = ", ")))
    quit(status = 1)
}
if (any(is.na(curve[needed]))) {
    cat("  FAIL: results/overopt-curve.csv has missing values in a column this needs\n")
    quit(status = 1)
}

fail <- function(fmt, ...) {
    cat(sprintf(paste0("  FAIL: ", fmt, "\n"), ...))
    failures <<- failures + 1
}

# ---- the claimed agreement bounds, read out of the README ------------------
m <- regmatches(readme,
                regexpr("agrees with gold on [0-9.]+ to [0-9.]+ of pairs", readme))
if (length(m) != 1) {
    fail("README no longer states the agreement range in the expected form")
    lo <- NA; hi <- NA
} else {
    nums <- as.numeric(regmatches(m, gregexpr("[0-9.]+", m))[[1]])
    lo <- nums[1]; hi <- nums[2]
}
m2 <- regmatches(readme, regexpr("as low as [0-9.]+", readme))
worst_claim <- if (length(m2) == 1)
    as.numeric(regmatches(m2, gregexpr("[0-9.]+", m2))[[1]]) else NA

cat("agreement between the reward model and gold\n")
# step 0 is the reference policy itself, which is the distribution the
# preference comparisons were drawn from
ref <- unique(curve$rm_agreement[curve$step == 0])
cat(sprintf("  one value per seed: %s\n",
            paste(sprintf("%.4f", sort(ref)), collapse = ", ")))
if (length(ref) != length(unique(curve$seed)))
    fail("expected one reference agreement per seed, got %d for %d seeds",
         length(ref), length(unique(curve$seed)))
if (!is.na(lo)) {
    if (abs(round(min(ref), 2) - lo) > 1e-9 || abs(round(max(ref), 2) - hi) > 1e-9)
        fail("README claims %.2f to %.2f, the curve gives %.2f to %.2f",
             lo, hi, round(min(ref), 2), round(max(ref), 2))
    else
        cat(sprintf("  rounds to %.2f to %.2f, as the README claims\n",
                    round(min(ref), 2), round(max(ref), 2)))
}

worst <- min(curve$rm_agreement)
at <- curve[which.min(curve$rm_agreement), ]
cat(sprintf("  worst anywhere on the curve: %.4f, at beta %.2f and KL %.1f\n",
            worst, at$beta, at$kl))
if (!is.na(worst_claim) && abs(round(worst, 2) - worst_claim) > 1e-9)
    fail("README says agreement falls as low as %.2f, the curve gives %.2f",
         worst_claim, round(worst, 2))
if (at$beta != 0)
    fail("the README attributes the worst agreement to a zero penalty run, this one has beta %.2f",
         at$beta)

# ---- the proxy climbs, the gold turns over ---------------------------------
x <- sqrt(curve$kl)
cat("\nthe proxy against drift\n")
fit_p <- lm(curve$proxy ~ x)
slope <- summary(fit_p)$coefficients["x", ]
rho <- cor(curve$proxy, curve$kl, method = "spearman")
cat(sprintf("  slope %+.4f (t = %.1f), Spearman rho %+.3f\n",
            slope[1], slope[3], rho))
if (slope[1] <= 0 || slope[3] < 3)
    fail("the proxy does not rise with drift, which the README's whole argument needs")
if (rho <= 0.5)
    fail("the proxy is not monotone in KL, Spearman rho %.3f", rho)

cat("\nthe gold against drift\n")
fit_g <- lm(curve$gold ~ x + I(x^2))
q <- summary(fit_g)$coefficients["I(x^2)", ]
b <- coef(fit_g)
peak <- -b[2] / (2 * b[3])
cat(sprintf("  quadratic term %+.4f (t = %.1f, p = %.1e)\n", q[1], q[3], q[4]))
cat(sprintf("  turning point at sqrt(KL) = %.2f, observed range %.2f to %.2f\n",
            peak, min(x), max(x)))
if (q[1] >= 0 || q[4] > 0.01)
    fail("no significant downward curvature in the gold, so nothing turns over")
if (peak <= min(x) || peak >= max(x))
    fail("the fitted peak is outside the observed range, so the curve does not turn over inside it")

# ---- cluster bootstrap over the twelve runs --------------------------------
# A run is one (beta, seed). Its fifteen checkpoints are the same policy a few
# steps apart, so the run is the resampling unit and not the checkpoint.
curve$run <- paste(curve$beta, curve$seed)
runs <- unique(curve$run)
cat(sprintf("\ncluster bootstrap, %d draws, resampling %d runs of %d checkpoints\n",
            DRAWS, length(runs), nrow(curve) / length(runs)))
quad <- numeric(DRAWS)
peaks <- numeric(DRAWS)
for (i in seq_len(DRAWS)) {
    idx <- unlist(lapply(sample(runs, length(runs), replace = TRUE),
                         function(r) which(curve$run == r)))
    xs <- sqrt(curve$kl[idx])
    cf <- tryCatch(coef(lm(curve$gold[idx] ~ xs + I(xs^2))), error = function(e) rep(NA, 3))
    quad[i] <- cf[3]
    peaks[i] <- -cf[2] / (2 * cf[3])
}
qi <- quantile(quad, c(0.025, 0.975), names = FALSE, na.rm = TRUE)
pk <- quantile(peaks, c(0.025, 0.975), names = FALSE, na.rm = TRUE)
cat(sprintf("  quadratic term  95%% interval %+.4f to %+.4f\n", qi[1], qi[2]))
cat(sprintf("  turning point   95%% interval %.2f to %.2f\n", pk[1], pk[2]))
frac <- mean(quad < 0, na.rm = TRUE)
cat(sprintf("  downward curvature in %.1f%% of draws\n", 100 * frac))
if (qi[2] >= 0)
    fail("the interval for the quadratic term includes zero, so the turnover is not established")

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR reproduces the agreement figures and the turnover survives a cluster bootstrap\n")
