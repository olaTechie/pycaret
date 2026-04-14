# mltoolkit Three-Agent Review — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Orchestrate three Opus reviewers (data scientist, public-health analyst, lead) to produce a prioritized gap analysis of the `mltoolkit-plugin` vs. its current spec, PyCaret's capability surface, and ML-paper reporting needs.

**Architecture:** Single-session orchestration driven from the main agent. Pre-fetch a PyCaret capability reference via Context7 to give all reviewers the same ruler. Spawn DS + PH reviewers in parallel (two `Agent` calls in one message), then the lead sequentially. Reviewers write their own reports to disk and return short confirmations; the lead merges findings into a single `findings.jsonl`. Main agent then verifies the five deliverable files and prints a summary. No plugin code changes this session.

**Tech Stack:** Claude Code `Agent` tool (general-purpose subagent, `model: opus`), Context7 MCP (`resolve-library-id`, `query-docs`), filesystem via `Write` / `Read` / `Bash`.

**Spec:** `mltoolkit-plugin/docs/superpowers/specs/2026-04-14-mltoolkit-three-agent-review-design.md`

---

## File Structure

All paths are relative to the plugin root: `mltoolkit-plugin/`.

**Created by this plan (all under `docs/superpowers/reviews/`):**

| File | Written by | Purpose |
|---|---|---|
| `2026-04-14-pycaret-capability-reference.md` | Main agent (Task 2) | Shared PyCaret feature map from Context7 |
| `2026-04-14-review-data-scientist.md` | DS reviewer agent (Task 3) | Technical-rigor review |
| `2026-04-14-review-public-health-analyst.md` | PH analyst agent (Task 3) | Applied + equity + reporting review |
| `2026-04-14-lead-synthesis.md` | Lead reviewer agent (Task 4) | Exec summary + prioritized backlog |
| `2026-04-14-findings.jsonl` | Lead reviewer agent (Task 4) | Deduped machine-readable findings |

**Read by this plan (reviewers only — not modified):**

- `mltoolkit-plugin/README.md`
- `mltoolkit-plugin/agents/ml-pipeline.md`
- All nine `mltoolkit-plugin/skills/*/SKILL.md`
- All files under `mltoolkit-plugin/skills/*/references/` and `mltoolkit-plugin/skills/*/scripts/`
- `mltoolkit-plugin/tests/` (test files + `test_references.sh`)
- `mltoolkit-plugin/scripts/check-env.sh`
- `mltoolkit-plugin/pytest.ini`

No plugin code is modified. The plan's "implementation" is orchestration + file production, so classical TDD does not apply — instead each task has explicit verification commands and pass/fail criteria.

---

## Task 1: Pre-flight verification

**Files:**
- Read: all required plugin inputs listed in spec "Required Reads"
- No writes this task

- [ ] **Step 1: Confirm reviews directory exists**

Run: `ls mltoolkit-plugin/docs/superpowers/reviews/`
Expected: directory listing (may be empty). If the command errors with "No such file or directory", run:

```bash
mkdir -p mltoolkit-plugin/docs/superpowers/reviews
```

- [ ] **Step 2: Enumerate required plugin files and fail fast if any are missing**

Run:

```bash
cd mltoolkit-plugin && \
for f in \
  README.md \
  agents/ml-pipeline.md \
  pytest.ini \
  scripts/check-env.sh \
  skills/setup/SKILL.md \
  skills/classify/SKILL.md \
  skills/regress/SKILL.md \
  skills/cluster/SKILL.md \
  skills/anomaly/SKILL.md \
  skills/compare/SKILL.md \
  skills/tune/SKILL.md \
  skills/eda/SKILL.md \
  skills/package/SKILL.md ; do
  [ -f "$f" ] || echo "MISSING: $f"
done
```

Expected: no output (every file exists). If any `MISSING:` line appears, **abort the plan** and surface the failure. Do not proceed to Task 2.

- [ ] **Step 3: Confirm Context7 MCP tools are loadable**

Use the `ToolSearch` tool with query `select:mcp__context7__resolve-library-id,mcp__context7__query-docs`.
Expected: both tool schemas returned in the result. If the search returns empty, **abort** and tell the user Context7 is not available.

- [ ] **Step 4: No commit this task** (read-only verification)

---

## Task 2: Build the PyCaret capability reference

**Files:**
- Create: `mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-pycaret-capability-reference.md`

- [ ] **Step 1: Resolve the PyCaret library ID**

Call `mcp__context7__resolve-library-id` with `libraryName: "pycaret"` and `question: "PyCaret API surface: setup, compare_models, tune_model, plot_model, interpret_model, calibrate_model, finalize_model, predict_model, save_model, deploy_model, check_fairness, dashboard, get_leaderboard, and per-task modules for classification, regression, clustering, anomaly"`.

Expected: one or more matches. Pick the best per the Context7 rules (exact name match, highest code-snippet count, High/Medium source reputation). Record the chosen ID (e.g. `/pycaret/pycaret`) — it will be reused across every `query-docs` call below.

- [ ] **Step 2: Query docs for the core top-level API**

Call `mcp__context7__query-docs` with the chosen library ID and question:

> "For each of these PyCaret functions, give me: (1) purpose, (2) signature with key arguments, (3) what it returns, (4) which task modules expose it. Functions: setup, compare_models, create_model, tune_model, ensemble_model, blend_models, stack_models, calibrate_model, plot_model, interpret_model, evaluate_model, predict_model, finalize_model, save_model, load_model, deploy_model, get_leaderboard, check_fairness, dashboard, pull, get_config, set_config, get_metrics, add_metric, remove_metric."

Save the response. Do not truncate — we need the full capability surface.

- [ ] **Step 3: Query docs for each task module**

Make four additional `query-docs` calls (one per module) against the same library ID, each asking for: "List all models in the model zoo (id, display name, underlying estimator), all available plots in `plot_model`, all tuning backends supported, any task-specific helpers not in the top-level API. Also list the metrics this module tracks by default." Modules:

1. `pycaret.classification`
2. `pycaret.regression`
3. `pycaret.clustering`
4. `pycaret.anomaly`

Save each response.

- [ ] **Step 4: Write the capability reference file**

Compose a markdown file with this structure and write it to `mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-pycaret-capability-reference.md`:

```markdown
# PyCaret Capability Reference (fetched 2026-04-14)

**Source:** Context7 MCP — library ID `<chosen-id>`
**Purpose:** Shared ruler for the three-agent review of mltoolkit-plugin. Reviewers measure gaps against this document, not their own recall.

## 1. Top-level API
<one subsection per function from Step 2, with purpose / signature / returns / available-in modules>

## 2. Classification module
### 2.1 Model zoo
### 2.2 Plots available via plot_model
### 2.3 Tuning backends
### 2.4 Default metrics
### 2.5 Task-specific helpers

## 3. Regression module
<same 5 subsections>

## 4. Clustering module
<same 5 subsections>

## 5. Anomaly detection module
<same 5 subsections>

## 6. Cross-cutting capabilities
- Fairness / bias auditing (check_fairness)
- Interpretability (interpret_model, SHAP integration)
- Experiment logging / MLflow
- Deployment (deploy_model targets)
- Dashboards and leaderboards

## 7. Notable gaps / caveats
<anything Context7 flags as deprecated, version-specific, or missing — e.g. "time series has its own module pycaret.time_series not covered here">
```

Every subsection must contain actual content extracted from the Context7 responses. No "TBD" or "see upstream docs" placeholders — if Context7 didn't return info for a slot, write an explicit `**Not returned by Context7 query.**` line so reviewers know it's absent, not forgotten.

- [ ] **Step 5: Verify the file is non-empty and has all expected sections**

Run:

```bash
wc -l mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-pycaret-capability-reference.md
grep -c '^## ' mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-pycaret-capability-reference.md
```

Expected: line count > 100, H2-heading count ≥ 7 (sections 1–7 above).

- [ ] **Step 6: Commit**

```bash
cd mltoolkit-plugin
git add docs/superpowers/reviews/2026-04-14-pycaret-capability-reference.md
git commit -m "$(cat <<'EOF'
docs(mltoolkit): PyCaret capability reference for three-agent review

Fetched via Context7. Shared ruler for the DS, public-health, and lead
reviewers — keeps gap measurements consistent across reports.
EOF
)"
```

---

## Task 3: Spawn the two domain reviewers in parallel

**Files:**
- Create: `mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-review-data-scientist.md` (written by DS agent)
- Create: `mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-review-public-health-analyst.md` (written by PH agent)

**Important:** Both agents are spawned in a **single message** with two `Agent` tool-use blocks — this is what makes them run in parallel. Do not send them in sequence.

- [ ] **Step 1: Send one message containing both Agent calls**

Agent call #1 — Data scientist reviewer:

- `description`: "DS review of mltoolkit plugin"
- `subagent_type`: "general-purpose"
- `model`: "opus"
- `prompt` (verbatim — this is the complete, self-contained prompt; do not paraphrase):

```
You are the DATA SCIENTIST REVIEWER in a three-agent review of the
mltoolkit-plugin (a Claude Code plugin that emits sklearn-native ML
code for classification, regression, clustering, and anomaly
detection).

YOUR LENS — technical rigor:
- Correctness: leakage, CV strategy, preprocessing order inside
  pipelines, metric selection, stratification, seed handling.
- Model-zoo completeness vs. PyCaret (use the capability reference).
- Tuning coverage: search strategies, early stopping, budget controls,
  and in particular whether the Optuna backend claimed in README.md
  actually exists in code.
- Interpretability: SHAP, permutation importance, PDP/ICE, calibration
  curves, learning curves.
- Deliverable packaging quality (skills/package): reproducibility,
  environment pinning, inference parity between the in-session
  session.py and the packaged artifact (tier A/B/C).

READ, in order (absolute paths relative to the repository root
mltoolkit-plugin/):
  README.md
  agents/ml-pipeline.md
  skills/setup/SKILL.md and skills/setup/references/**
  skills/classify/SKILL.md and skills/classify/references/**
  skills/regress/SKILL.md and skills/regress/references/**
  skills/cluster/SKILL.md and skills/cluster/references/**
  skills/anomaly/SKILL.md and skills/anomaly/references/**
  skills/compare/SKILL.md and skills/compare/references/**
  skills/tune/SKILL.md and skills/tune/references/**
  skills/eda/SKILL.md and skills/eda/references/**
  skills/package/SKILL.md and skills/package/references/**
  tests/ (all files)
  scripts/check-env.sh
  pytest.ini
  docs/superpowers/reviews/2026-04-14-pycaret-capability-reference.md

CONSTRAINTS:
- Read-only. Do NOT edit any plugin file. You may ONLY write to
  docs/superpowers/reviews/2026-04-14-review-data-scientist.md.
- Every finding must cite evidence as "<path>:<line or symbol> — <what
  you observed>". No vibes.
- If you cannot complete the review, still write the report and use
  Section 1 (Summary) to explain why. Do not silently exit.

OUTPUT — write exactly one markdown file at
docs/superpowers/reviews/2026-04-14-review-data-scientist.md with the
following sections:

  1. Summary (≤200 words)
  2. Current-spec gaps (where the plugin fails its own stated goals)
  3. PyCaret-target gaps (missing capabilities vs. capability reference)
  4. Paper-reporting gaps (what's missing for ML-paper workflows)
  5. Prioritized recommendations (P0 / P1 / P2 with rationale)
  6. Open questions for the follow-up session
  7. Findings — a fenced ```jsonl code block containing one JSON
     object per finding, one per line. Required fields per object:
       id          (string, e.g. "DS-001")
       reviewer    ("data-scientist")
       category    (short tag, e.g. "tuning", "leakage", "packaging")
       severity    ("P0" | "P1" | "P2")
       tags        (subset of ["current-spec-gap",
                              "pycaret-target-gap",
                              "paper-reporting-gap"])
       affected_skill  (e.g. "mltoolkit:tune" or "agents/ml-pipeline")
       evidence    (string — "<path>:<line or symbol> — <observation>")
       recommendation (string — concrete next action)

RETURN to the caller: a ≤50-word confirmation containing the output
path and a one-line status. The caller already knows the template;
do not repeat findings in your return value.
```

Agent call #2 — Public-health data analyst:

- `description`: "Public-health analyst review of mltoolkit"
- `subagent_type`: "general-purpose"
- `model`: "opus"
- `prompt` (verbatim):

```
You are the PUBLIC-HEALTH DATA ANALYST REVIEWER in a three-agent
review of the mltoolkit-plugin (a Claude Code plugin that emits
sklearn-native ML code for classification, regression, clustering,
and anomaly detection).

YOUR LENS — applied epi/clinical fit + equity + paper reporting:
- Applied workflow fit: Table 1 generation, stratified and subgroup
  metrics, missing-data handling (MAR/MCAR/MNAR awareness), rare-
  outcome handling, class imbalance for low-prevalence outcomes,
  calibration in the small.
- ETHICS / BIAS / EQUITY AUDIT (first-class, not bundled): group
  fairness metrics, disparate performance across protected
  attributes, representational harms, documentation of protected-
  attribute handling, dataset provenance / consent signals.
- Reporting-guideline alignment: TRIPOD+AI, STARD 2015 (diagnostic),
  CONSORT-AI (interventional), PROBAST (risk-of-bias).
- Paper-ready outputs: methods-section scaffolding, results tables,
  calibration + decision-curve plots, supplementary reproducibility
  appendix.

READ, in order (absolute paths relative to the repository root
mltoolkit-plugin/):
  README.md
  agents/ml-pipeline.md
  skills/setup/SKILL.md and skills/setup/references/**
  skills/classify/SKILL.md and skills/classify/references/**
  skills/regress/SKILL.md and skills/regress/references/**
  skills/cluster/SKILL.md and skills/cluster/references/**
  skills/anomaly/SKILL.md and skills/anomaly/references/**
  skills/compare/SKILL.md and skills/compare/references/**
  skills/tune/SKILL.md and skills/tune/references/**
  skills/eda/SKILL.md and skills/eda/references/**
  skills/package/SKILL.md and skills/package/references/**
  tests/ (all files)
  scripts/check-env.sh
  pytest.ini
  docs/superpowers/reviews/2026-04-14-pycaret-capability-reference.md

CONSTRAINTS:
- Read-only. Do NOT edit any plugin file. You may ONLY write to
  docs/superpowers/reviews/2026-04-14-review-public-health-analyst.md.
- Every finding must cite evidence as "<path>:<line or symbol> — <what
  you observed>".
- If you cannot complete the review, still write the report and use
  Section 1 (Summary) to explain why. Do not silently exit.

OUTPUT — write exactly one markdown file at
docs/superpowers/reviews/2026-04-14-review-public-health-analyst.md
with the following sections:

  1. Summary (≤200 words)
  2. Current-spec gaps
  3. PyCaret-target gaps
  4. Paper-reporting gaps
  5. Ethics / bias / equity audit (DEDICATED section, required)
  6. Prioritized recommendations (P0 / P1 / P2 with rationale)
  7. Open questions for the follow-up session
  8. Findings — a fenced ```jsonl code block containing one JSON
     object per finding, one per line. Required fields:
       id          (string, e.g. "PH-001")
       reviewer    ("public-health-analyst")
       category    (short tag, e.g. "fairness", "tripod-ai",
                    "missing-data", "calibration")
       severity    ("P0" | "P1" | "P2")
       tags        (subset of ["current-spec-gap",
                              "pycaret-target-gap",
                              "paper-reporting-gap"])
       affected_skill  (e.g. "mltoolkit:classify")
       evidence    (string — "<path>:<line or symbol> — <observation>")
       recommendation (string — concrete next action)

RETURN to the caller: a ≤50-word confirmation containing the output
path and a one-line status.
```

- [ ] **Step 2: Verify both reviewer files exist and are non-empty**

Run:

```bash
for f in \
  mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-review-data-scientist.md \
  mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-review-public-health-analyst.md ; do
  if [ ! -s "$f" ]; then echo "EMPTY OR MISSING: $f"; fi
done
```

Expected: no output.

- [ ] **Step 3: Verify each report has the required sections**

Run:

```bash
grep -E '^#{1,3} ' mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-review-data-scientist.md
grep -E '^#{1,3} ' mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-review-public-health-analyst.md
```

Expected DS headings cover: Summary, Current-spec gaps, PyCaret-target gaps, Paper-reporting gaps, Prioritized recommendations, Open questions, Findings.

Expected PH headings additionally cover: Ethics / bias / equity audit (section 5).

If any required section is missing from either file, abort and surface the gap — do **not** proceed to the lead.

- [ ] **Step 4: Verify each report contains a parseable JSONL findings block**

Run:

```bash
python3 <<'PY'
import json, re, sys
for path in [
  "mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-review-data-scientist.md",
  "mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-review-public-health-analyst.md",
]:
  text = open(path).read()
  m = re.search(r"```jsonl\s*\n(.*?)```", text, re.DOTALL)
  if not m:
    print(f"NO JSONL BLOCK: {path}"); sys.exit(1)
  n = 0
  for i, line in enumerate(m.group(1).splitlines(), 1):
    line = line.strip()
    if not line: continue
    try:
      obj = json.loads(line)
    except json.JSONDecodeError as e:
      print(f"INVALID JSON at {path} line {i}: {e}"); sys.exit(1)
    for k in ("id","reviewer","category","severity","tags","affected_skill","evidence","recommendation"):
      if k not in obj:
        print(f"MISSING FIELD '{k}' at {path} line {i}"); sys.exit(1)
    n += 1
  print(f"OK: {path} — {n} findings")
PY
```

Expected: two `OK:` lines, no errors. If errors, abort and surface — the lead cannot merge malformed JSONL.

- [ ] **Step 5: Commit the two reviewer reports**

```bash
cd mltoolkit-plugin
git add docs/superpowers/reviews/2026-04-14-review-data-scientist.md \
        docs/superpowers/reviews/2026-04-14-review-public-health-analyst.md
git commit -m "$(cat <<'EOF'
docs(mltoolkit): data-scientist and public-health reviewer reports

Parallel independent reviews against current spec, PyCaret capability
reference, and ML-paper reporting needs. Findings emitted as JSONL
blocks for the lead synthesizer to dedupe.
EOF
)"
```

---

## Task 4: Spawn the lead reviewer

**Files:**
- Create: `mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-lead-synthesis.md`
- Create: `mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-findings.jsonl`

- [ ] **Step 1: Send one Agent call for the lead reviewer**

- `description`: "Lead reviewer synthesis"
- `subagent_type`: "general-purpose"
- `model`: "opus"
- `prompt` (verbatim):

```
You are the LEAD REVIEWER in a three-agent review of the
mltoolkit-plugin. Two parallel reviewers have completed their
reports; your job is to synthesize.

READ, in order (paths relative to repository root):
  docs/superpowers/reviews/2026-04-14-pycaret-capability-reference.md
  docs/superpowers/reviews/2026-04-14-review-data-scientist.md
  docs/superpowers/reviews/2026-04-14-review-public-health-analyst.md

You may also read any plugin file referenced in the evidence of a
finding if you need to adjudicate — but do not conduct your own
independent review. Your role is synthesis, not re-review.

YOUR JOB:
1. Extract the JSONL findings blocks from both reviewer reports.
2. Dedupe by evidence + affected_skill. When two reviewers flag the
   same issue, merge into a single record and list both original IDs
   in a new "source_ids" field (array of strings).
3. Resolve disagreements. If the two reviewers assign different
   severities or recommendations, pick one, add a "disagreement_note"
   field explaining what you picked and why.
4. Re-tag each finding. Every finding must carry ≥1 tag from
   {"current-spec-gap","pycaret-target-gap","paper-reporting-gap"}.
5. Re-assign severity using: blast radius × effort × strategic fit
   with the new PyCaret/paper target. Severities must be
   P0 | P1 | P2.
6. Assign fresh consolidated IDs ("LEAD-001", "LEAD-002", ...).

CONSTRAINTS:
- Read-only on plugin files and reviewer reports. You may ONLY write
  to docs/superpowers/reviews/2026-04-14-lead-synthesis.md and
  docs/superpowers/reviews/2026-04-14-findings.jsonl.
- No finding may be orphaned — every DS-xxx and PH-xxx source ID must
  appear in the source_ids of at least one lead finding, OR be
  explicitly listed in a "Superseded findings" subsection of the
  synthesis with a one-line justification.

OUTPUT 1 — write
docs/superpowers/reviews/2026-04-14-lead-synthesis.md with:

  1. Executive summary (≤500 words) covering: key themes, biggest
     risks to the current spec, biggest gaps vs. PyCaret, biggest
     gaps vs. ML-paper reporting, and top three recommended next
     actions.
  2. Prioritized backlog table — columns: ID | severity | tags |
     affected_skill | one-line recommendation. Sort by severity then
     ID.
  3. Disagreements resolved — bullet list, one per merged finding
     where DS and PH disagreed; note what each said and what you
     chose.
  4. Superseded findings (if any) — list of source IDs dropped with
     one-line reason each.
  5. Cross-references — a table mapping each source ID (DS-xxx and
     PH-xxx) to the consolidated LEAD-xxx that absorbed it.

OUTPUT 2 — write
docs/superpowers/reviews/2026-04-14-findings.jsonl containing one
JSON object per line, one per consolidated finding. Required fields:
    id              ("LEAD-xxx")
    source_ids      (array of original "DS-xxx"/"PH-xxx" IDs)
    category        (short tag)
    severity        ("P0" | "P1" | "P2")
    tags            (non-empty subset of the three allowed tags)
    affected_skill  (e.g. "mltoolkit:tune")
    evidence        (string — may combine both reviewers' evidence)
    recommendation  (string)
    disagreement_note (string, OPTIONAL — only if DS and PH disagreed)

RETURN to the caller: a ≤50-word confirmation with both output paths,
the total consolidated-finding count, and the count of disagreements
resolved.
```

- [ ] **Step 2: Verify both lead outputs exist and are non-empty**

Run:

```bash
for f in \
  mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-lead-synthesis.md \
  mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-findings.jsonl ; do
  if [ ! -s "$f" ]; then echo "EMPTY OR MISSING: $f"; fi
done
```

Expected: no output.

- [ ] **Step 3: Verify lead synthesis sections**

Run:

```bash
grep -E '^#{1,3} ' mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-lead-synthesis.md
```

Expected: headings for Executive summary, Prioritized backlog, Disagreements resolved, Superseded findings (may be empty-bodied but must be present), Cross-references.

- [ ] **Step 4: Validate `findings.jsonl` schema and executive-summary word budget**

Run:

```bash
python3 <<'PY'
import json, re, sys

jsonl = "mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-findings.jsonl"
synth = "mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-lead-synthesis.md"
ds    = "mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-review-data-scientist.md"
ph    = "mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-review-public-health-analyst.md"

allowed_tags = {"current-spec-gap","pycaret-target-gap","paper-reporting-gap"}
allowed_sev  = {"P0","P1","P2"}
required = ("id","source_ids","category","severity","tags","affected_skill","evidence","recommendation")

lead = []
for i, line in enumerate(open(jsonl), 1):
  line = line.strip()
  if not line: continue
  obj = json.loads(line)
  for k in required:
    assert k in obj, f"line {i} missing {k}"
  assert obj["severity"] in allowed_sev, f"line {i} bad severity"
  assert obj["tags"], f"line {i} empty tags"
  assert set(obj["tags"]).issubset(allowed_tags), f"line {i} bad tag"
  assert obj["id"].startswith("LEAD-"), f"line {i} bad id prefix"
  lead.append(obj)
print(f"findings.jsonl OK — {len(lead)} consolidated findings")

# Orphan check: every DS-xxx and PH-xxx from source reviewer reports
# must appear in some lead.source_ids OR in the synthesis's
# "Superseded findings" section.
def ids_from(path, prefix):
  text = open(path).read()
  m = re.search(r"```jsonl\s*\n(.*?)```", text, re.DOTALL)
  return {json.loads(l)["id"] for l in m.group(1).splitlines() if l.strip()}

ds_ids = ids_from(ds, "DS-")
ph_ids = ids_from(ph, "PH-")
absorbed = {sid for f in lead for sid in f["source_ids"]}
synth_text = open(synth).read()
orphans = []
for sid in (ds_ids | ph_ids):
  if sid in absorbed: continue
  if sid in synth_text: continue   # mentioned in superseded list
  orphans.append(sid)
if orphans:
  print("ORPHANS (not absorbed and not superseded):", orphans)
  sys.exit(1)
print("No orphan findings.")

# Exec summary word budget
m = re.search(r"#{1,3}\s*Executive summary.*?\n(.*?)(?=\n#{1,3}\s|\Z)",
              synth_text, re.DOTALL | re.IGNORECASE)
if not m:
  print("NO EXECUTIVE SUMMARY SECTION"); sys.exit(1)
words = len(m.group(1).split())
print(f"Executive summary: {words} words")
if words > 500:
  print("OVER BUDGET (>500 words)"); sys.exit(1)
PY
```

Expected output: three OK lines, exit 0. Any failure → surface and stop; do not commit a malformed synthesis.

- [ ] **Step 5: Commit the lead outputs**

```bash
cd mltoolkit-plugin
git add docs/superpowers/reviews/2026-04-14-lead-synthesis.md \
        docs/superpowers/reviews/2026-04-14-findings.jsonl
git commit -m "$(cat <<'EOF'
docs(mltoolkit): lead synthesis and consolidated findings

Deduplicated DS and public-health findings into a single prioritized
backlog. findings.jsonl is the machine-readable input for the
follow-up enhancement-plan session.
EOF
)"
```

---

## Task 5: Post-flight summary for the user

**Files:**
- No writes. Terminal output only.

- [ ] **Step 1: Print the deliverable paths and counts**

Run:

```bash
python3 <<'PY'
import json, re, pathlib
base = pathlib.Path("mltoolkit-plugin/docs/superpowers/reviews")
files = [
  "2026-04-14-pycaret-capability-reference.md",
  "2026-04-14-review-data-scientist.md",
  "2026-04-14-review-public-health-analyst.md",
  "2026-04-14-lead-synthesis.md",
  "2026-04-14-findings.jsonl",
]
print("Deliverables:")
for f in files:
  p = base / f
  size = p.stat().st_size if p.exists() else 0
  print(f"  {p}  ({size} bytes)")

lead = [json.loads(l) for l in open(base / files[-1]) if l.strip()]
by_sev = {"P0":0,"P1":0,"P2":0}
by_tag = {"current-spec-gap":0,"pycaret-target-gap":0,"paper-reporting-gap":0}
disagreements = 0
for f in lead:
  by_sev[f["severity"]] += 1
  for t in f["tags"]: by_tag[t] = by_tag.get(t,0)+1
  if f.get("disagreement_note"): disagreements += 1

print(f"\nConsolidated findings: {len(lead)}")
print(f"By severity: {by_sev}")
print(f"By tag:      {by_tag}")
print(f"Disagreements resolved by lead: {disagreements}")
PY
```

Expected: paths listed, severity and tag breakdowns printed, disagreement count printed.

- [ ] **Step 2: Suggest next step to the user**

Print the following suggestion to the user:

> "Review complete. To turn this into an enhancement plan, run `/superpowers:writing-plans` against `mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-lead-synthesis.md` (and point it at `findings.jsonl` for the machine-readable backlog). That session will produce a concrete plan for the PyCaret-parity + ML-paper-reporting enhancements."

- [ ] **Step 3: No commit this task** (nothing written).

---

## Out of Scope

- Acting on findings (follow-up session).
- Plugin code edits.
- Reviewer retries or multi-round adjudication.
- Implementing PyCaret-parity or paper-reporting features.
