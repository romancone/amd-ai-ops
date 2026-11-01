---
name: benchmark-report-review
description: Review benchmark imports, coverage and report migrations while preserving measurement meaning and access to results. Use for dataset or viewer changes, not routine text or CSS edits.
---

# Benchmark Report Review

Read [AGENTS.md](../../../AGENTS.md) for rules and permissions and
[MAINTAINERS.md](../../../MAINTAINERS.md) for current file ownership.

1. Identify the canonical dataset, import inputs, generated outputs and manual
   pages from the ownership map; inspect the working tree and index separately
2. Confirm workload identity and metric definitions before matching records
3. Reconcile approved configurations with saved attempts, imported records and
   visible selections; classify gaps as missing runs, missing imports or filters
   Reuse exact matches before proposing additional measurements
4. Apply the bounded change using the rules in AGENTS.md
5. Compare result and image-answer IDs before and after a migration, not just
   counts; account for approved removals and ensure imports cannot restore them
   Confirm retained answers and previews remain reachable in the replacement
6. Verify within the allowed scope; when authorized, check generated summaries,
   selector behavior and image previews, including local and published paths
   Otherwise distinguish static inspection from unverified runtime behavior

Report discrepancies and verification limits briefly; update the existing
pending plan rather than creating another tracking document.
