#!/usr/bin/env bash
# Recompute what the README publishes, in every language here, and require
# agreement.
#
# Every number in this repository came out of one implementation: the sweep in
# experiments/overopt.py writes results/*.csv, bench/figures.py reads them back,
# and the README was written from the same output. Nothing checked that the
# aggregation was right, because everything downstream read the same numbers. An
# error in rlhf/ppo.py would appear in the results, the figures and the prose
# together and look consistent.
#
# These are independent implementations. For the PPO kernels they work from
# golden vectors exported by verify/export_golden.py; for the published tables
# they work from the per seed file results/methods.csv; for the claims written in
# words they work from the same file and the README text.
#
# Each is skipped with a clear message if its toolchain is absent, so this runs
# on a laptop with only some of them. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# The SQL has no assertion of its own: it prints the five rows of the
# overoptimization table as they should read, and the comparison is here.
# sqlite3 reads stdin, which inside a script is the rest of this script, so the
# redirect from /dev/null is not optional. Its CSV output is CRLF, hence the tr.
check_sql () {
    local rows n=0 bad=0
    rows=$(sqlite3 -init verify/medians.sql :memory: "" < /dev/null 2>/dev/null | tr -d '\r')
    [ -n "$rows" ] || { echo "sqlite produced nothing"; return 1; }
    # bold markers are formatting, not part of the number
    local readme; readme=$(tr -d '*' < README.md)
    while IFS= read -r row; do
        [ -n "$row" ] || continue
        n=$((n + 1))
        if printf '%s\n' "$readme" | grep -qF -- "$row"; then
            printf '  in README: %s\n' "$row"
        else
            printf '  MISSING from README: %s\n' "$row"
            bad=$((bad + 1))
        fi
    done <<< "$rows"
    if [ "$n" -ne 5 ]; then
        echo "expected 5 table rows from SQL, got $n"
        return 1
    fi
    [ "$bad" -eq 0 ] || { echo "$bad of $n rows do not match the README table"; return 1; }
    echo "SQL reproduces all $n rows of the overoptimization table"
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -o "$tmp/kernels" verify/kernels.c -lm \
        || return 1
    "$tmp/kernels" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/ppokernel && cargo run --release --quiet -- "$root" ); }

# The golden vectors are only evidence if they still come out of rlhf/ppo.py.
# This regenerates them somewhere else and requires the files to be identical,
# which is what stops a change to the kernel from leaving stale vectors behind
# that the C and the Rust then happily agree with.
check_python () {
    # the repository's own venv if it has one, otherwise whatever python3 is
    local py=./.venv/bin/python
    [ -x "$py" ] || py=python3
    "$py" -c 'import torch' 2>/dev/null || {
        echo "torch is not importable, so the golden vectors cannot be regenerated"
        return 2
    }
    "$py" verify/export_golden.py --check || return 1

    # the test count the README publishes
    local want got
    want=$(grep -Eo 'tests/ +[0-9]+ tests' README.md | grep -Eo '[0-9]+' | head -1)
    got=$("$py" -m pytest tests/ -q --collect-only 2>/dev/null \
          | grep -Eo '^[0-9]+ tests collected' | grep -Eo '^[0-9]+')
    if [ -z "$got" ]; then
        echo "  could not collect the test suite, skipping the count"
    elif [ "$want" != "$got" ]; then
        echo "  README says tests/ has $want tests, pytest collects $got"
        return 1
    else
        echo "  README says tests/ has $want tests, pytest collects $got"
    fi
}

# check_python needs a reason to be skipped rather than failed when torch is
# absent, which is the one case where the toolchain test above is not enough.
run_python () {
    printf '\n=== %s ===\n' "Python, golden vectors still match rlhf/ppo.py"
    if ! command -v python3 >/dev/null 2>&1; then
        printf 'skipped: python3 is not installed\n'
        skip=$((skip + 1)); return
    fi
    check_python
    case $? in
        0) pass=$((pass + 1)) ;;
        2) skip=$((skip + 1)) ;;
        *) fail=$((fail + 1)) ;;
    esac
}

run "SQL, the published table from the per seed file"  sqlite3 check_sql
run "C, the PPO kernels against golden vectors"        cc      check_c
run "Rust, GAE by the closed form and a random sweep"  cargo   check_rust
run "Go, file structure and the table recomputed"      go      check_go
run "R, the claims that are statistical"               Rscript Rscript verify/verify.R "$root"
run "JavaScript, the claims written in words"          node    node verify/claims.js "$root"
run_python

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
