#!/usr/bin/env bash
# Setup GitHub labels using GitHub CLI (gh)
# Usage: bash .github/setup_labels.sh [owner/repo]

set -euo pipefail

REPO_ARG="${1:-}"

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: GitHub CLI (gh) is not installed."
  echo "Install it via 'brew install gh' and authenticate with 'gh auth login'."
  exit 1
fi

GH_CMD="gh label create"
EXTRA_ARGS="--force"

if [ -n "$REPO_ARG" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --repo $REPO_ARG"
fi

create_label() {
  local name="$1"
  local color="$2"
  local desc="$3"
  echo "Setting label: '$name' (#$color)..."
  gh label create "$name" --color "$color" --description "$desc" $EXTRA_ARGS
}

# Type labels
create_label "type: eda" "0E8A16" "Exploratory Data Analysis & Data Quality"
create_label "type: feature" "1D76DB" "Feature engineering or pipeline transformation"
create_label "type: experiment" "5319E7" "Model experiment, tuning, or benchmarking"
create_label "type: bug" "D93F0B" "Pipeline error, broken fetch, or test failure"
create_label "type: task" "006B75" "General engineering task, CI/CD, or maintenance"
create_label "type: docs" "0075CA" "Documentation, wiki, or presentation slides"

# Area labels
create_label "area: data" "BFDADC" "Raw data, downloads, storage, and ingestion"
create_label "area: features" "C2E0C6" "Feature engineering and data transformations"
create_label "area: modeling" "D4C5F9" "Model architecture, loss functions, uncertainty"
create_label "area: eval" "FEF2C0" "Metrics, cross-validation, benchmarking"
create_label "area: infra" "F9D0C4" "Makefile, environments, tests, dependencies"

# Priority labels
create_label "prio: p0-urgent" "B60205" "Blocks the team or critical milestone"
create_label "prio: p1-high" "E99695" "High priority for current milestone"
create_label "prio: p2-medium" "FBCA04" "Normal priority"
create_label "prio: p3-low" "FEF2C0" "Nice-to-have / backlog refinement"

# Status labels
create_label "status: blocked" "6A0DAD" "Blocked by external dependency or another issue"
create_label "status: needs-review" "FBCA04" "PR is waiting on teammate review"
create_label "status: needs-run" "C5DEF5" "Awaiting long-running computation or data rebuild"

echo "All labels configured successfully!"
