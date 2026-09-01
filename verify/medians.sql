-- Recompute the overoptimization table in the README from results/methods.csv.
--
-- The README publishes one row per KL penalty. Each cell is the median over the
-- three seeds of the matching column in results/methods.csv, and until now the
-- only thing that computed those medians was experiments/overopt.py, whose
-- output is what the README was written from. Nothing checked the two agreed.
--
-- This emits the markdown rows exactly as they should appear. verify/verify.sh
-- looks for each of them in README.md, so a cell that drifts stops being found.
--
-- Median with an even count is the mean of the two middle values, which is why
-- the row_number filter takes (n+1)/2 and (n+2)/2 under integer division: with
-- three seeds both land on the same row, and the code stays correct if a
-- fourth seed is ever added.
.mode csv
.import results/methods.csv m
.headers off
.mode list

WITH long(method, col, val) AS (
    SELECT method, 'kl',    CAST(kl    AS REAL) FROM m
    UNION ALL SELECT method, 'proxy', CAST(proxy AS REAL) FROM m
    UNION ALL SELECT method, 'gold',  CAST(gold  AS REAL) FROM m
    UNION ALL SELECT method, 'motif', CAST(motif AS REAL) FROM m
    UNION ALL SELECT method, 'hoard', CAST(hoard AS REAL) FROM m
),
ranked AS (
    SELECT method, col, val,
           ROW_NUMBER() OVER (PARTITION BY method, col ORDER BY val) AS rn,
           COUNT(*)     OVER (PARTITION BY method, col)              AS n
    FROM long
),
med AS (
    SELECT method, col, AVG(val) AS v
    FROM ranked
    WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
    GROUP BY method, col
),
wide AS (
    SELECT method,
           MAX(CASE WHEN col = 'kl'    THEN v END) AS kl,
           MAX(CASE WHEN col = 'proxy' THEN v END) AS proxy,
           MAX(CASE WHEN col = 'gold'  THEN v END) AS gold,
           MAX(CASE WHEN col = 'motif' THEN v END) AS motif,
           MAX(CASE WHEN col = 'hoard' THEN v END) AS hoard
    FROM med GROUP BY method
),
labelled AS (
    SELECT CASE method
               WHEN 'SFT (reference)' THEN 'reference'
               WHEN 'PPO (beta=0.2)'  THEN '0.2'
               WHEN 'PPO (beta=0.05)' THEN '0.05'
               WHEN 'PPO (beta=0.01)' THEN '0.01'
               WHEN 'PPO (beta=0.0)'  THEN '0.0'
           END AS label,
           CASE method
               WHEN 'SFT (reference)' THEN 0 WHEN 'PPO (beta=0.2)'  THEN 1
               WHEN 'PPO (beta=0.05)' THEN 2 WHEN 'PPO (beta=0.01)' THEN 3
               WHEN 'PPO (beta=0.0)'  THEN 4
           END AS ord,
           kl, proxy, gold, motif, hoard
    FROM wide
)
-- The README writes a negative as a unicode minus, so this does too, and the
-- signed columns carry an explicit plus.
SELECT '| ' || label
    || ' | ' || printf('%.2f', kl)
    || ' | ' || replace(printf('%+.3f', proxy), '-', char(8722))
    || ' | ' || replace(printf('%+.3f', gold),  '-', char(8722))
    || ' | ' || printf('%.2f', motif)
    || ' | ' || printf('%.2f', hoard)
    || ' |'
FROM labelled WHERE label IS NOT NULL ORDER BY ord;
