/* Recompute the PPO kernels in C from the golden vectors.
 *
 * Three things are checked, all of them written in rlhf/ppo.py and all of them
 * load bearing for the overoptimization result:
 *
 *   token level KL penalty   reward_t = -beta * kl_t, plus the reward model
 *                            score added at the final token only
 *   GAE                      the backward recursion over the token sequence
 *   clipped surrogate        the PPO policy loss and the clipped value loss
 *
 * The point is not speed. It is that a mistake in the Python would have to be
 * repeated identically here, and in the Rust, to survive. Columns are resolved
 * by name, so a column added upstream cannot silently shift what this reads.
 *
 * Reads the CSVs under verify/golden, exits non-zero on the first disagreement past the
 * tolerance.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-12
#define LINE 4096
#define MAXT 64
#define MAXN 256

static int column_of(const char *header, const char *name)
{
    char buf[LINE];
    strncpy(buf, header, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';

    int i = 0;
    for (char *tok = strtok(buf, ",\r\n"); tok; tok = strtok(NULL, ",\r\n"), i++)
        if (strcmp(tok, name) == 0)
            return i;
    return -1;
}

static const char *field(const char *line, int index)
{
    static char out[256];
    int col = 0;
    const char *p = line;
    while (col < index) {
        p = strchr(p, ',');
        if (!p)
            return "";
        p++;
        col++;
    }
    const char *end = strchr(p, ',');
    size_t n = end ? (size_t)(end - p) : strlen(p);
    if (n >= sizeof out)
        n = sizeof out - 1;
    memcpy(out, p, n);
    out[n] = '\0';
    char *nl = strpbrk(out, "\r\n");
    if (nl)
        *nl = '\0';
    return out;
}

static double num(const char *line, int index)
{
    return strtod(field(line, index), NULL);
}

static double worst = 0.0;

static int close_enough(const char *what, int cse, int t, double got, double want)
{
    const double d = fabs(got - want);
    if (d > worst)
        worst = d;
    if (d <= TOL)
        return 1;
    printf("  FAIL case %d t %d %s: C %.17g golden %.17g  |d| %.3e\n",
           cse, t, what, got, want, d);
    return 0;
}

/* ---- GAE and the token level KL penalty ---------------------------------- */
static int check_gae(const char *root)
{
    char path[1024], line[LINE], header[LINE];
    snprintf(path, sizeof path, "%s/verify/golden/gae.csv", root);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    if (!fgets(header, sizeof header, f)) { fclose(f); return -1; }

    const int c_case = column_of(header, "case"), c_gamma = column_of(header, "gamma");
    const int c_lam = column_of(header, "lam"), c_beta = column_of(header, "kl_beta");
    const int c_score = column_of(header, "score"), c_t = column_of(header, "t");
    const int c_kl = column_of(header, "kl_tok"), c_reward = column_of(header, "reward");
    const int c_value = column_of(header, "value"), c_adv = column_of(header, "adv");
    const int c_ret = column_of(header, "ret");
    if (c_case < 0 || c_gamma < 0 || c_lam < 0 || c_beta < 0 || c_score < 0 ||
        c_t < 0 || c_kl < 0 || c_reward < 0 || c_value < 0 || c_adv < 0 || c_ret < 0) {
        fprintf(stderr, "gae.csv is missing a column this check needs\n");
        fclose(f);
        return -1;
    }

    double kl[MAXT], val[MAXT], want_r[MAXT], want_a[MAXT], want_ret[MAXT];
    double gamma = 0, lam = 0, beta = 0, score = 0;
    int cur = -1, T = 0, bad = 0, cases = 0;

    /* One trailing flush after the loop, so the last case is not dropped. */
    for (int done = 0; !done; ) {
        char *got_line = fgets(line, sizeof line, f);
        int this_case = -1;
        if (got_line && line[0] != '\n' && line[0] != '\0')
            this_case = (int)num(line, c_case);
        else
            done = 1;

        if ((done || this_case != cur) && cur >= 0) {
            double adv[MAXT], last = 0.0;
            for (int t = T - 1; t >= 0; t--) {
                const double next_v = (t + 1 < T) ? val[t + 1] : 0.0;
                const double delta = want_r[t] + gamma * next_v - val[t];
                last = delta + gamma * lam * last;
                adv[t] = last;
            }
            for (int t = 0; t < T; t++) {
                /* the reward model score lands on the final token only */
                const double r = -beta * kl[t] + (t == T - 1 ? score : 0.0);
                bad += !close_enough("reward", cur, t, r, want_r[t]);
                bad += !close_enough("adv", cur, t, adv[t], want_a[t]);
                bad += !close_enough("ret", cur, t, adv[t] + val[t], want_ret[t]);
            }
            printf("  case %d  T %2d  gamma %.4g lam %.4g beta %.4g   ok\n",
                   cur, T, gamma, lam, beta);
            cases++;
            T = 0;
        }
        if (done)
            break;

        if (this_case != cur) {
            cur = this_case;
            gamma = num(line, c_gamma);
            lam = num(line, c_lam);
            beta = num(line, c_beta);
            score = num(line, c_score);
        }
        const int t = (int)num(line, c_t);
        if (t >= MAXT) { fprintf(stderr, "sequence longer than MAXT\n"); fclose(f); return -1; }
        kl[t] = num(line, c_kl);
        val[t] = num(line, c_value);
        want_r[t] = num(line, c_reward);
        want_a[t] = num(line, c_adv);
        want_ret[t] = num(line, c_ret);
        if (t + 1 > T)
            T = t + 1;
    }
    fclose(f);
    if (cases == 0) { fprintf(stderr, "no cases in gae.csv\n"); return -1; }
    printf("GAE and KL shaping: %d cases, %d disagreements\n", cases, bad);
    return bad;
}

/* ---- the clipped PPO surrogate ------------------------------------------- */
static int check_losses(const char *root)
{
    char path[1024], line[LINE], header[LINE];
    snprintf(path, sizeof path, "%s/verify/golden/ppo_inputs.csv", root);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    if (!fgets(header, sizeof header, f)) { fclose(f); return -1; }

    const int c_case = column_of(header, "case"), c_i = column_of(header, "i");
    const int c_lp = column_of(header, "lp"), c_olp = column_of(header, "old_lp");
    const int c_adv = column_of(header, "adv"), c_v = column_of(header, "v");
    const int c_ov = column_of(header, "old_v"), c_ret = column_of(header, "ret");
    if (c_case < 0 || c_i < 0 || c_lp < 0 || c_olp < 0 || c_adv < 0 ||
        c_v < 0 || c_ov < 0 || c_ret < 0) {
        fprintf(stderr, "ppo_inputs.csv is missing a column this check needs\n");
        fclose(f);
        return -1;
    }

    static double lp[16][MAXN], olp[16][MAXN], adv[16][MAXN];
    static double v[16][MAXN], ov[16][MAXN], ret[16][MAXN];
    static int count[16];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\0')
            continue;
        const int c = (int)num(line, c_case), i = (int)num(line, c_i);
        if (c < 0 || c >= 16 || i < 0 || i >= MAXN) {
            fprintf(stderr, "case %d index %d out of range\n", c, i);
            fclose(f);
            return -1;
        }
        lp[c][i] = num(line, c_lp);   olp[c][i] = num(line, c_olp);
        adv[c][i] = num(line, c_adv); v[c][i] = num(line, c_v);
        ov[c][i] = num(line, c_ov);   ret[c][i] = num(line, c_ret);
        if (i + 1 > count[c])
            count[c] = i + 1;
    }
    fclose(f);

    snprintf(path, sizeof path, "%s/verify/golden/ppo_loss.csv", root);
    f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    if (!fgets(header, sizeof header, f)) { fclose(f); return -1; }
    const int l_case = column_of(header, "case"), l_clip = column_of(header, "clip");
    const int l_vclip = column_of(header, "vf_clip"), l_n = column_of(header, "n");
    const int l_frac = column_of(header, "frac_ratio_clipped");
    const int l_pg = column_of(header, "pg_loss"), l_vf = column_of(header, "vf_loss");
    if (l_case < 0 || l_clip < 0 || l_vclip < 0 || l_n < 0 || l_frac < 0 ||
        l_pg < 0 || l_vf < 0) {
        fprintf(stderr, "ppo_loss.csv is missing a column this check needs\n");
        fclose(f);
        return -1;
    }

    int bad = 0, cases = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\0')
            continue;
        const int c = (int)num(line, l_case);
        const double clip = num(line, l_clip), vclip = num(line, l_vclip);
        const int n = (int)num(line, l_n);
        if (c < 0 || c >= 16 || n != count[c]) {
            printf("  FAIL case %d: ppo_loss.csv says n=%d, ppo_inputs.csv has %d\n",
                   c, n, c >= 0 && c < 16 ? count[c] : -1);
            bad++;
            continue;
        }

        double pg = 0.0, vf = 0.0;
        int clipped = 0;
        for (int i = 0; i < n; i++) {
            const double ratio = exp(lp[c][i] - olp[c][i]);
            double r_clamped = ratio;
            if (r_clamped < 1 - clip) r_clamped = 1 - clip;
            if (r_clamped > 1 + clip) r_clamped = 1 + clip;
            if (ratio < 1 - clip || ratio > 1 + clip)
                clipped++;
            const double a = ratio * adv[c][i], b = r_clamped * adv[c][i];
            pg += -(a < b ? a : b);

            double dv = v[c][i] - ov[c][i];
            if (dv < -vclip) dv = -vclip;
            if (dv > vclip) dv = vclip;
            const double e1 = v[c][i] - ret[c][i], e2 = ov[c][i] + dv - ret[c][i];
            const double s1 = e1 * e1, s2 = e2 * e2;
            vf += 0.5 * (s1 > s2 ? s1 : s2);
        }
        pg /= n;
        vf /= n;

        bad += !close_enough("pg_loss", c, -1, pg, num(line, l_pg));
        bad += !close_enough("vf_loss", c, -1, vf, num(line, l_vf));
        bad += !close_enough("frac_ratio_clipped", c, -1,
                             (double)clipped / n, num(line, l_frac));
        printf("  case %d  n %3d  clip %.2g  %2d of %d ratios clipped   "
               "pg %+.12f  vf %.12f\n", c, n, clip, clipped, n, pg, vf);
        cases++;
    }
    fclose(f);
    if (cases == 0) { fprintf(stderr, "no cases in ppo_loss.csv\n"); return -1; }
    printf("clipped surrogate: %d cases, %d disagreements\n", cases, bad);
    return bad;
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    const int a = check_gae(root);
    if (a < 0)
        return 2;
    const int b = check_losses(root);
    if (b < 0)
        return 2;
    if (a + b) {
        printf("\n%d values disagree with the golden vectors\n", a + b);
        return 1;
    }
    printf("\nC reproduces every golden value, worst |d| %.1e, tolerance %.0e\n",
           worst, TOL);
    return 0;
}
