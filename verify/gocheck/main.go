// Structural validation of every results file, and a second independent
// recomputation of the README's overoptimization table.
//
// results/*.csv is the evidence for every number in the README, and the golden
// vectors under verify/golden are the evidence for the C and Rust checks.
// Nothing checked that any of them are well formed. A truncated write, a
// column that drifted, a NaN out of a division, a seed silently missing from
// the sweep: all of it would be invisible until someone read the table.
//
// The medians here are computed by sorting, not by SQL's window functions, and
// compared against the table as it is actually written in README.md. Between
// this and verify/medians.sql the published table is recomputed twice from the
// per seed file by two implementations that share no code.
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Cells in the README table, so the comparison is against the printed text and
// not against a rounding this program chose for itself.
var tableCols = []struct {
	name   string
	places int
	signed bool
}{
	{"kl", 2, false}, {"proxy", 3, true}, {"gold", 3, true},
	{"motif", 2, false}, {"hoard", 2, false},
}

var labelOf = map[string]string{
	"SFT (reference)": "reference",
	"PPO (beta=0.2)":  "0.2",
	"PPO (beta=0.05)": "0.05",
	"PPO (beta=0.01)": "0.01",
	"PPO (beta=0.0)":  "0.0",
}

var tableOrder = []string{"reference", "0.2", "0.05", "0.01", "0.0"}

type table struct {
	header []string
	rows   [][]string
}

func readCSV(path string) (*table, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) < 2 {
		return nil, fmt.Errorf("only %d rows", len(rows))
	}
	return &table{header: rows[0], rows: rows[1:]}, nil
}

func (t *table) col(name string) int {
	for i, h := range t.header {
		if h == name {
			return i
		}
	}
	return -1
}

func (t *table) get(row int, name string) string {
	i := t.col(name)
	if i < 0 || i >= len(t.rows[row]) {
		return ""
	}
	return t.rows[row][i]
}

// validate reports every structural problem in one file rather than the first,
// so a broken run is diagnosed in one pass.
func validate(path string) []string {
	var problems []string
	t, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}

	seen := map[string]bool{}
	for _, h := range t.header {
		if strings.TrimSpace(h) == "" {
			problems = append(problems, "a column has an empty name")
		}
		if seen[h] {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}

	// A column is numeric if every cell in it parses. NaN and Inf parse in Go,
	// so they are rejected explicitly rather than left to the parser.
	for j, name := range t.header {
		numeric := true
		for _, row := range t.rows {
			if j >= len(row) {
				continue
			}
			if _, err := strconv.ParseFloat(strings.TrimSpace(row[j]), 64); err != nil {
				numeric = false
				break
			}
		}
		if !numeric {
			continue
		}
		for i, row := range t.rows {
			v, _ := strconv.ParseFloat(strings.TrimSpace(row[j]), 64)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is %s", i+2, name, row[j]))
			}
		}
	}

	for i, row := range t.rows {
		for j, cell := range row {
			if strings.TrimSpace(cell) == "" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is empty", i+2, t.header[j]))
			}
		}
	}
	return problems
}

func median(xs []float64) float64 {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func format(v float64, places int, signed bool) string {
	s := strconv.FormatFloat(v, 'f', places, 64)
	if signed && !strings.HasPrefix(s, "-") {
		s = "+" + s
	}
	// the README writes a negative as a unicode minus
	return strings.Replace(s, "-", "−", 1)
}

// parseREADMETable pulls the overoptimization table out of the README as it is
// written, keyed by the first cell of each row.
func parseREADMETable(readme string) map[string][]string {
	out := map[string][]string{}
	for _, line := range strings.Split(readme, "\n") {
		s := strings.TrimSpace(line)
		if !strings.HasPrefix(s, "|") || !strings.HasSuffix(s, "|") {
			continue
		}
		parts := strings.Split(strings.Trim(s, "|"), "|")
		if len(parts) != 6 {
			continue
		}
		cells := make([]string, 0, 6)
		for _, p := range parts {
			cells = append(cells, strings.TrimSpace(strings.ReplaceAll(p, "*", "")))
		}
		for _, want := range tableOrder {
			if cells[0] == want {
				out[want] = cells[1:]
			}
		}
	}
	return out
}

func fail(problems *[]string, format string, args ...any) {
	*problems = append(*problems, fmt.Sprintf(format, args...))
}

// die stops on a file this program cannot read at all, after printing whatever
// it already found, so a ragged file is diagnosed alongside the rest.
func die(problems []string, format string, args ...any) {
	for _, p := range problems {
		fmt.Printf("  - %s\n", p)
	}
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(2)
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	var problems []string

	// ---- structure of every CSV the repository publishes or verifies against
	var files []string
	for _, pat := range []string{"results/*.csv", "verify/golden/*.csv"} {
		found, err := filepath.Glob(filepath.Join(*root, pat))
		if err != nil {
			fmt.Fprintf(os.Stderr, "bad glob %s: %v\n", pat, err)
			os.Exit(2)
		}
		files = append(files, found...)
	}
	if len(files) == 0 {
		fmt.Fprintf(os.Stderr, "no CSVs found under %s\n", *root)
		os.Exit(2)
	}
	sort.Strings(files)
	fmt.Printf("validating %d files\n", len(files))
	for _, path := range files {
		rel, _ := filepath.Rel(*root, path)
		for _, p := range validate(path) {
			fail(&problems, "%s: %s", rel, p)
		}
	}

	// ---- the sweep is complete: every method seen once per seed
	methods, err := readCSV(filepath.Join(*root, "results", "methods.csv"))
	if err != nil {
		die(problems, "results/methods.csv: %v", err)
	}
	seen := map[string]int{}
	seeds, names := map[string]bool{}, map[string]bool{}
	for i := range methods.rows {
		m, s := methods.get(i, "method"), methods.get(i, "seed")
		seen[m+"|"+s]++
		seeds[s] = true
		names[m] = true
	}
	for k, n := range seen {
		if n != 1 {
			fail(&problems, "results/methods.csv has %d rows for %s", n, k)
		}
	}
	if len(methods.rows) != len(seeds)*len(names) {
		fail(&problems, "results/methods.csv has %d rows, not %d methods x %d seeds",
			len(methods.rows), len(names), len(seeds))
	}
	fmt.Printf("results/methods.csv: %d rows, %d methods x %d seeds\n",
		len(methods.rows), len(names), len(seeds))

	// ---- the curve has as many checkpoints as the README says it does
	curve, err := readCSV(filepath.Join(*root, "results", "overopt-curve.csv"))
	if err != nil {
		die(problems, "results/overopt-curve.csv: %v", err)
	}
	readmeBytes, err := os.ReadFile(filepath.Join(*root, "README.md"))
	if err != nil {
		die(problems, "README.md: %v", err)
	}
	readme := string(readmeBytes)
	m := regexp.MustCompile(`(?i)the same (\d+) checkpoints`).FindStringSubmatch(readme)
	if m == nil {
		fail(&problems, "README.md no longer states a checkpoint count")
	} else {
		want, _ := strconv.Atoi(m[1])
		if len(curve.rows) != want {
			fail(&problems, "README says %d checkpoints, overopt-curve.csv has %d",
				want, len(curve.rows))
		}
		fmt.Printf("results/overopt-curve.csv: %d rows, README says %s\n",
			len(curve.rows), m[1])
	}
	keys := map[string]bool{}
	for i := range curve.rows {
		k := curve.get(i, "beta") + "|" + curve.get(i, "seed") + "|" + curve.get(i, "step")
		if keys[k] {
			fail(&problems, "results/overopt-curve.csv repeats beta/seed/step %s", k)
		}
		keys[k] = true
	}

	// ---- run-meta.json describes the sweep that actually ran
	metaBytes, err := os.ReadFile(filepath.Join(*root, "results", "run-meta.json"))
	if err != nil {
		die(problems, "results/run-meta.json: %v", err)
	}
	var meta struct {
		Seeds    []int     `json:"seeds"`
		Betas    []float64 `json:"betas"`
		PPOSteps int       `json:"ppo_steps"`
	}
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		die(problems, "results/run-meta.json: %v", err)
	}
	if len(meta.Seeds) != len(seeds) {
		fail(&problems, "run-meta.json lists %d seeds, methods.csv has %d",
			len(meta.Seeds), len(seeds))
	}
	betas := map[string]bool{}
	maxStep := -1
	for i := range curve.rows {
		betas[curve.get(i, "beta")] = true
		if s, err := strconv.Atoi(curve.get(i, "step")); err == nil && s > maxStep {
			maxStep = s
		}
	}
	if len(meta.Betas) != len(betas) {
		fail(&problems, "run-meta.json lists %d betas, the curve has %d",
			len(meta.Betas), len(betas))
	}
	if maxStep >= meta.PPOSteps {
		fail(&problems, "the curve reaches step %d but run-meta.json says %d steps",
			maxStep, meta.PPOSteps)
	}
	fmt.Printf("results/run-meta.json: %d seeds, %d betas, %d steps, curve tops out at %d\n",
		len(meta.Seeds), len(meta.Betas), meta.PPOSteps, maxStep)

	// ---- the README table, recomputed by sorting
	published := parseREADMETable(readme)
	fmt.Println("recomputing the overoptimization table from results/methods.csv")
	for _, label := range tableOrder {
		cells, ok := published[label]
		if !ok {
			fail(&problems, "no row %q in the README table", label)
			continue
		}
		got := make([]string, 0, len(tableCols))
		for k, c := range tableCols {
			var vals []float64
			for i := range methods.rows {
				if labelOf[methods.get(i, "method")] != label {
					continue
				}
				v, err := strconv.ParseFloat(methods.get(i, c.name), 64)
				if err != nil {
					fail(&problems, "results/methods.csv column %s is not a number: %v",
						c.name, err)
					continue
				}
				vals = append(vals, v)
			}
			if len(vals) == 0 {
				fail(&problems, "no rows in methods.csv for %q", label)
				got = append(got, "?")
				continue
			}
			s := format(median(vals), c.places, c.signed)
			got = append(got, s)
			if s != cells[k] {
				fail(&problems, "row %q column %s: recomputed %s, README says %s",
					label, c.name, s, cells[k])
			}
		}
		fmt.Printf("  %-10s %s\n", label, strings.Join(got, "  "))
	}

	if len(problems) > 0 {
		fmt.Printf("\n%d problems:\n", len(problems))
		for _, p := range problems {
			fmt.Printf("  - %s\n", p)
		}
		os.Exit(1)
	}
	fmt.Printf("\nGo: %d files well formed, and the README table matches the medians exactly\n",
		len(files))
}
