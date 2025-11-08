# PolarPandas Improvement Plan

## Overview
PolarPandas delivers a large pandas-compatible surface atop Polars but exhibits critical quality gaps and misaligned documentation. This plan captures prioritized improvements derived from a project-wide review of packaging, core abstractions, utilities, tooling, and tests.

## Priority 1 — Critical Fixes
- **Broken CLI entry point**: `pyproject.toml` declares `polarpandas.cli:main`, yet no `cli.py` module exists. Either implement the CLI (with tests) or remove the console script before the next release to prevent installation failures.
- **LazyFrame eager materialization**: `LazyFrame.__repr__`, `__str__`, `__len__`, `columns`, etc. call `.collect()`, eliminating laziness and risking large-scale memory use. Redesign repr/info accessors to rely on `collect_schema()` or partial materialization only when explicitly requested.
- **Datetime helpers correctness**: `date_range` and `to_datetime` return naïve placeholder sequences; adopt Polars-native implementations (`pl.datetime_range`, `pl.Series.str.strptime`) with pandas-compatible parameter handling and add regression tests.

## Priority 2 — High-Impact Improvements
- **Modularize monolithic classes**: Split the 13k-line `frame.py` and 6k-line `series.py` into focused submodules (construction, arithmetic, reductions, indexing) with shared helper utilities to reduce duplication and simplify maintenance.
- **Tighten LazyFrame joins**: Harden `LazyFrame.join` to accept both polarpandas and raw Polars inputs, ensuring `other_df_lazy` is always defined and enriching type hints to prevent runtime NameErrors.
- **Robust binning utilities**: Replace the hand-rolled logic in `utils.cut`/`operations.qcut` with Polars-native binning, fix label padding bugs, and align behavior with pandas (edge inclusion, NA handling).
- **Index-aware Series ops**: Align Series arithmetic/comparisons on index labels and preserve metadata (`name`, `index`) to match pandas semantics and avoid silent misalignment.

## Priority 3 — Medium-Term Enhancements
- **Dependency hygiene**: Reconcile duplication between `pyproject.toml` extra groups and `requirements_dev.txt`, remove unused extras (e.g., `ty`), and document the rationale for version pins vs. the minimal `requirements.txt`.
- **Tooling coherence**: Standardize on a single linter (`ruff` vs `flake8`). Update `tox.ini`, CI workflows, and contributor docs accordingly; wire mypy/ruff enforcement into automated checks.
- **Testing depth**: Introduce shared fixtures/factories, expand behavioral tests around index preservation, inplace mutations, datetime utilities, and optional dependency paths. Increase coverage of `_index_manager`, `utils`, and `datetime`.
- **IndexManager cleanup**: Simplify `preserve_index_inplace` length checks and add tests that confirm index retention across reshape operations.

## Documentation Realignment
- Audit README/TODO/KNOWN_LIMITATIONS claims (coverage, feature completeness, optional dependency support) against reality, trimming or updating marketing copy. Automate export of the pandas API matrix to keep documentation synchronized with actual capability.

## Quick Wins
- Centralize common dtype tuples/constants used across aggregators instead of repeating inline lists.
- Move repeated `import polars as pl` statements to module scope to streamline code and facilitate patching/mocking.
- Add smoke tests for the eventual CLI, optional dependencies, and lazy representations.

## Execution Roadmap
1. **Stabilize Critical Paths** (Weeks 1–2)  
   Address CLI entry point, lazy materialization, and datetime helper correctness with dedicated regression tests.
2. **Refactor Core Modules** (Weeks 3–5)  
   Break down `frame.py`/`series.py`, extract shared utilities, and harden LazyFrame join logic.
3. **Enhance Utilities & Tooling** (Weeks 6–7)  
   Rewrite binning utilities, align dependency/tooling configurations, and clean up IndexManager.
4. **Expand Test & Doc Coverage** (Weeks 8–9)  
   Add behavioral tests, update documentation, and integrate automated compatibility matrices.

Progress should be tracked in issues linked to these milestones, with CI gates updated as fixes land.

