# Spec Delta: training-observability

## Purpose

Make the quality of a training run observable: record what happened, show it as a chart, and serve that chart live while the run is in progress, without depending on an external tracking service.

## ADDED Requirements

### Requirement: Durable per-run metrics

The system SHALL persist the full metric history of a training run — per-step loss and every other metric the trainer emits, plus the hyperparameters the run used — to a per-run file under a gitignored artifacts directory. The file SHALL be written outside the checkpoint directory, which training clears at the start of every run, and SHALL be updated after every logging step rather than only at the end. Writes SHALL be atomic so a concurrent reader never observes a partial file.

#### Scenario: A completed run leaves a record

- **WHEN** a training run finishes
- **THEN** a run file exists naming the run, its hyperparameters, and the full metric history, and it survives the next run being started

#### Scenario: A run in progress is readable

- **WHEN** a reader loads the run file while training is still going
- **THEN** it parses successfully and contains the steps logged so far

#### Scenario: An interrupted run keeps what it had

- **WHEN** a run is killed part way through
- **THEN** the metrics logged before the interruption remain on disk

### Requirement: Metrics chart

The system SHALL render recorded runs as a self-contained HTML chart, using no plotting library, no JavaScript and no network requests, so it can be opened directly from disk. The chart SHALL show one panel per metric family — loss, token accuracy, entropy and gradient norm — sharing an epoch axis, and SHALL overlay multiple runs on the same axes with a legend identifying each. Where a run has held-out metrics, they SHALL be drawn distinctly from training metrics and the best held-out value SHALL be marked.

#### Scenario: Charting a single run

- **WHEN** the chart command runs against a directory containing one run
- **THEN** it writes an HTML file showing that run's metrics, readable in a browser with no external assets

#### Scenario: Comparing runs

- **WHEN** several runs exist
- **THEN** all of them appear on the same axes, distinguished by colour, with each legend entry naming the run and summarising its result

#### Scenario: No runs recorded yet

- **WHEN** the chart command runs and no run files exist
- **THEN** it exits non-zero with a message explaining how to produce one

### Requirement: Live metrics server

The system SHALL provide a local HTTP server that re-renders the metrics chart on each request and instructs the browser to refresh periodically, so a training run can be watched as it progresses. The server SHALL bind to the loopback interface only, SHALL use a different port from the browser chat application so both can run at once, and SHALL remain available when no runs exist yet or when a run file cannot be rendered.

#### Scenario: Watching a run

- **WHEN** the server is running and a training run is in progress
- **THEN** reloading the page shows the metrics logged so far

#### Scenario: Started before any run

- **WHEN** the server is started with no runs recorded
- **THEN** it serves a page explaining that no runs exist rather than failing

#### Scenario: Port already in use

- **WHEN** the configured port is occupied
- **THEN** the server exits non-zero with a message naming the port and suggesting an alternative

### Requirement: Recovering metrics from a training log

The system SHALL be able to build a chart from a saved training stdout log as well as from a run file, so runs recorded before metrics persistence existed, or run outside the pipeline, can still be charted.

#### Scenario: Charting from a stdout log

- **WHEN** the chart command is pointed at a text file containing the per-step metric lines the trainer prints
- **THEN** it extracts those metrics and renders them as a run
