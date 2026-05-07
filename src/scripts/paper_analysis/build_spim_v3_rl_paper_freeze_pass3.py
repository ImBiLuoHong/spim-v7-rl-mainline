from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
PASS1_ROOT = ARTIFACTS_ROOT / "paper_analysis" / "20260420_080826"
PASS2_ROOT = ARTIFACTS_ROOT / "paper_analysis_pass2" / "20260420_084219"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-freeze pass-3 manuscript assembly bundle.")
    parser.add_argument("--pass1-root", type=str, default=str(PASS1_ROOT))
    parser.add_argument("--pass2-root", type=str, default=str(PASS2_ROOT))
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(raw: str) -> Path:
    if raw:
        path = Path(raw)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = ARTIFACTS_ROOT / "paper_freeze_pass3" / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_claim_ledger(pass1_root: Path, pass2_root: Path) -> pd.DataFrame:
    p1_comp = pd.read_csv(pass1_root / "comparison_table.csv")
    p1_tax = pd.read_csv(pass1_root / "case_taxonomy.csv")
    p1_diff = pd.read_csv(pass1_root / "difficulty_bucket_tables.csv")
    p1_pol = pd.read_csv(pass1_root / "policy_behavior_tables.csv")
    p2_comp = pd.read_csv(pass2_root / "complementarity_tables.csv")
    p2_slot = pd.read_csv(pass2_root / "slot_role_tendency_tables.csv")
    p2_front = pd.read_csv(pass2_root / "frontier_composition_tables.csv")

    strongest = p1_comp[p1_comp["method_id"] == "strongest_rl"].iloc[0]
    greedy = p1_comp[p1_comp["method_id"] == "posterior_greedy"].iloc[0]
    entropy_q4 = p1_diff[(p1_diff["feature"] == "initial_posterior_entropy") & (p1_diff["bucket"] == "q4")].iloc[0]
    policy_overall = p1_pol[p1_pol["slice"] == "overall"].iloc[0]
    subset = p2_comp[p2_comp["slice"] == "overall_subset"].set_index("strategy")
    slot_rl1 = p2_slot[(p2_slot["slice"] == "overall_subset") & (p2_slot["strategy"] == "strongest_rl") & (p2_slot["slot_index"] == 1)].iloc[0]
    slot_rl3 = p2_slot[(p2_slot["slice"] == "overall_subset") & (p2_slot["strategy"] == "strongest_rl") & (p2_slot["slot_index"] == 3)].iloc[0]
    frontier_target = p2_front[(p2_front["table_scope"] == "targeted_slot_frontier_source") & (p2_front["slice"] == "overall_subset")].set_index("frontier_source")

    rows: List[Dict[str, Any]] = [
        {
            "claim_id": "C1",
            "claim_text": "Strongest RL beats Greedy Posterior and all four heuristic baselines under the aligned held-out val/B30/SPIM-v3 exact-hit contract.",
            "status": "proven",
            "supporting_evidence_locator": "comparison_table.csv rows method_id=strongest_rl,posterior_greedy,posterior_thompson_sampling,posterior_entropy_drop,posterior_cover_shrink,posterior_disagreement_split",
            "supporting_artifacts": json.dumps([
                str(pass1_root / "comparison_table.csv"),
                str(pass1_root / "analysis_summary.md"),
                str(pass1_root / "run_manifest.json"),
            ]),
            "recommended_figure_or_table": "Main Table 1",
            "safe_wording": f"Under the aligned held-out val/B30/SPIM-v3 exact-hit contract, Strongest RL achieved Success@B30={strongest['success_at_B30']:.6f}, exceeding Greedy Posterior ({greedy['success_at_B30']:.6f}) and all other heuristic baselines.",
            "avoid_wording": "RL universally dominates all reasonable posterior-based policies under any evaluation setup.",
        },
        {
            "claim_id": "C2",
            "claim_text": "RL gains appear early in the budget curve, not only at the end.",
            "status": "proven",
            "supporting_evidence_locator": "success_vs_budget.csv rows method_id=strongest_rl,posterior_greedy at sample_budget=1,4,10,20,30",
            "supporting_artifacts": json.dumps([
                str(pass1_root / "success_vs_budget.csv"),
                str(pass1_root / "success_vs_budget.png"),
            ]),
            "recommended_figure_or_table": "Main Figure 1",
            "safe_wording": "The advantage of RL is already visible in the early budget regime and is not solely a late-budget effect.",
            "avoid_wording": "RL only helps by making the final few selections more efficient.",
        },
        {
            "claim_id": "C3",
            "claim_text": "RL gains concentrate more in hard states (high entropy / large support / low margin).",
            "status": "proven",
            "supporting_evidence_locator": "difficulty_bucket_tables.csv rows feature=initial_posterior_entropy,bucket=q4 and feature=initial_candidate_count,bucket=largest_support; hard_state_mechanism_tables.csv rows slice=hard_q4_entropy sample_budget=1,5,10,20,30",
            "supporting_artifacts": json.dumps([
                str(pass1_root / "difficulty_bucket_tables.csv"),
                str(pass2_root / "hard_state_mechanism_tables.csv"),
                str(pass2_root / "hard_state_success_budget.png"),
            ]),
            "recommended_figure_or_table": "Main Figure 2",
            "safe_wording": f"The relative advantage of RL is larger in harder initial states, especially in the highest-entropy bucket where RL reached {entropy_q4['rl_success_rate']:.6f} vs Greedy Posterior {entropy_q4['greedy_success_rate']:.6f}.",
            "avoid_wording": "RL only matters in impossible cases or uniformly helps every regime equally.",
        },
        {
            "claim_id": "C4",
            "claim_text": "RL is not simply copying Greedy's 3-set choices.",
            "status": "proven",
            "supporting_evidence_locator": "policy_behavior_tables.csv row slice=overall; complementarity_tables.csv rows analysis_type=fullpanel_observational_overlap",
            "supporting_artifacts": json.dumps([
                str(pass1_root / "policy_behavior_tables.csv"),
                str(pass1_root / "policy_overlap.png"),
                str(pass2_root / "complementarity_tables.csv"),
            ]),
            "recommended_figure_or_table": "Main Figure 3 or Appendix Figure A1",
            "safe_wording": f"RL differs substantially from Greedy Posterior at the set level, with an exact 3-set match rate of only {policy_overall['exact_action_set_match_rate']:.6f} and mean Jaccard overlap {policy_overall['mean_jaccard_overlap']:.6f}.",
            "avoid_wording": "RL is just Greedy Posterior with noise or trivial reranking.",
        },
        {
            "claim_id": "C5",
            "claim_text": "RL gains are more consistent with second/third-pick completion and set-level complementarity than with just a better first pick.",
            "status": "partially proven",
            "supporting_evidence_locator": "complementarity_tables.csv rows slice=overall_subset strategy=strongest_rl,posterior_greedy,rl1_g23,g1_rl23,rl12_g3,g12_rl3; run_manifest.json targeted_subset_case_count=24",
            "supporting_artifacts": json.dumps([
                str(pass2_root / "complementarity_tables.csv"),
                str(pass2_root / "complementarity_hybrid_success.png"),
                str(pass2_root / "analysis_summary_pass2.md"),
                str(pass2_root / "run_manifest.json"),
            ]),
            "recommended_figure_or_table": "Appendix Table A1",
            "safe_wording": f"On the bounded 24-case hard subset, replacing later picks changes outcomes more than replacing only the first pick (e.g. RL1+G23={subset.at['rl1_g23','success_rate']:.6f}, G1+RL23={subset.at['g1_rl23','success_rate']:.6f}), which is more consistent with later-pick completion than with a pure first-pick explanation.",
            "avoid_wording": "We proved that the RL advantage is entirely caused by second/third-pick complementarity.",
        },
        {
            "claim_id": "C6",
            "claim_text": "Slot-level tendency exists: earlier picks are more exploitative, later picks more complementary/dispersed.",
            "status": "partially proven",
            "supporting_evidence_locator": "slot_role_tendency_tables.csv rows slice=overall_subset strategy=strongest_rl slot_index=1,2,3",
            "supporting_artifacts": json.dumps([
                str(pass2_root / "slot_role_tendency_tables.csv"),
                str(pass2_root / "slot_role_tendency.png"),
            ]),
            "recommended_figure_or_table": "Appendix Figure A2",
            "safe_wording": f"Slot statistics suggest an exploit-first, then-complement tendency: RL slot 1 has mean posterior rank {slot_rl1['posterior_rank_mean']:.3f}, while slot 3 has mean posterior rank {slot_rl3['posterior_rank_mean']:.3f} and larger distance to previously selected nodes.",
            "avoid_wording": "The three action slots correspond to three fixed semantic heads or roles.",
        },
        {
            "claim_id": "C7",
            "claim_text": "Fixed three semantic roles emerged.",
            "status": "not proved",
            "supporting_evidence_locator": "analysis_summary_pass2.md non-claim note; slot_role_tendency_tables.csv only shows descriptive tendencies",
            "supporting_artifacts": json.dumps([
                str(pass2_root / "slot_role_tendency_tables.csv"),
                str(pass2_root / "analysis_summary_pass2.md"),
            ]),
            "recommended_figure_or_table": "Do not use as a positive claim",
            "safe_wording": "We observe slot-level tendencies but do not claim fixed semantic roles.",
            "avoid_wording": "The policy clearly learned three stable semantic heads.",
        },
        {
            "claim_id": "C8",
            "claim_text": "Strong novelty-frontier or mixed-frontier behavior is the main RL mechanism.",
            "status": "not proved",
            "supporting_evidence_locator": "frontier_composition_tables.csv rows table_scope=targeted_slot_frontier_source slice=overall_subset frontier_source=posterior,disagreement,novelty,fill",
            "supporting_artifacts": json.dumps([
                str(pass2_root / "frontier_composition_tables.csv"),
                str(pass2_root / "frontier_composition.png"),
                str(pass2_root / "analysis_summary_pass2.md"),
            ]),
            "recommended_figure_or_table": "Appendix limitation note only",
            "safe_wording": f"The selected-node frontier composition remains strongly posterior-dominated in the targeted replay subset (posterior={frontier_target.at['posterior','fraction']:.6f}); this does not support a strong mixed-frontier mechanism claim.",
            "avoid_wording": "RL wins primarily because it mixes posterior, disagreement, and novelty frontiers in a balanced way.",
        },
        {
            "claim_id": "C9",
            "claim_text": "Exact hit is the main metric and relaxed-radius metrics are secondary only.",
            "status": "proven",
            "supporting_evidence_locator": "comparison_table.csv primary endpoint rows; relaxed_metric_tables.csv rows method_id=strongest_rl,posterior_greedy at radius_hops=0,1,2 and sample_budget=30",
            "supporting_artifacts": json.dumps([
                str(pass1_root / "comparison_table.csv"),
                str(pass1_root / "relaxed_metric_tables.csv"),
                str(pass1_root / "analysis_summary.md"),
            ]),
            "recommended_figure_or_table": "Methods paragraph + Appendix Figure A3",
            "safe_wording": "Exact node-level Success@B30 is the primary endpoint; relaxed radius curves are reported as secondary context only.",
            "avoid_wording": "Relaxed radius success is the real task objective and can replace exact hit.",
        },
        {
            "claim_id": "C10",
            "claim_text": "The paired-case improvement is real and not just an aggregate-rate artifact.",
            "status": "proven",
            "supporting_evidence_locator": "comparison_table.csv rows method_id=strongest_rl,posterior_greedy; case_taxonomy.csv rows taxonomy_group=rl_unique_win,both_hit_rl_earlier,both_hit_greedy_earlier,greedy_unique_win,both_fail",
            "supporting_artifacts": json.dumps([
                str(pass1_root / "comparison_table.csv"),
                str(pass1_root / "case_taxonomy.csv"),
            ]),
            "recommended_figure_or_table": "Main Table 1 + Main Figure 3",
            "safe_wording": f"RL shows 41 RL-only wins versus 16 Greedy-only wins, with exact McNemar p={strongest['mcnemar_exact_p']:.6g}, and a substantial both-hit-but-RL-earlier bucket ({int(p1_tax[p1_tax['taxonomy_group']=='both_hit_rl_earlier'].iloc[0]['case_count'])} cases).",
            "avoid_wording": "The advantage is only a tiny average improvement without case-level support.",
        },
        {
            "claim_id": "C11",
            "claim_text": "Hybrid replay sometimes outperforming the original RL checkpoint means the original RL policy is suboptimal on the full benchmark.",
            "status": "not proved",
            "supporting_evidence_locator": "complementarity_tables.csv overall_subset synthetic hybrid rows; run_manifest.json targeted_subset_case_count=24",
            "supporting_artifacts": json.dumps([
                str(pass2_root / "complementarity_tables.csv"),
                str(pass2_root / "run_manifest.json"),
            ]),
            "recommended_figure_or_table": "Reviewer-defense memo only",
            "safe_wording": "The hybrid replay is bounded-subset mechanism evidence with synthetic prefix/suffix combinations; it should not be interpreted as a new headline benchmark or as proof of global suboptimality.",
            "avoid_wording": "The original RL checkpoint is clearly inferior to the synthetic hybrids in general.",
        },
        {
            "claim_id": "C12",
            "claim_text": "The hard-state story is consistent with an early ambiguity-reduction advantage.",
            "status": "partially proven",
            "supporting_evidence_locator": "hard_state_mechanism_tables.csv rows slice=hard_q4_entropy_early_rounds round_index=1,2,3 and slice=hard_q4_entropy sample_budget=1,5,10,30",
            "supporting_artifacts": json.dumps([
                str(pass2_root / "hard_state_mechanism_tables.csv"),
                str(pass2_root / "hard_state_success_budget.png"),
            ]),
            "recommended_figure_or_table": "Main Figure 2",
            "safe_wording": "The hard-state evidence is consistent with an early ambiguity-reduction advantage: RL reduces entropy slightly faster and increases top-3 mass earlier in the highest-entropy subset.",
            "avoid_wording": "We proved the precise causal mechanism is early ambiguity reduction.",
        },
    ]
    return pd.DataFrame(rows)


def final_figure_table_plan_md(pass1_root: Path, pass2_root: Path) -> str:
    return f"""# Final Figure/Table Plan

## Main Paper

### Table 1. Main Comparison Under The Aligned Held-Out Contract
- Purpose: establish the headline result that Strongest RL beats Greedy Posterior and all four heuristic baselines under the locked val/B30/SPIM-v3 exact-hit contract.
- Supporting claims: `C1`, `C10`
- Why main paper: this is the headline empirical result and must appear early.
- Source files:
  - `{pass1_root / 'comparison_table.csv'}`

### Figure 1. Success vs Budget
- Purpose: show that the RL gain appears early in the budget curve rather than only at the end.
- Supporting claims: `C2`
- Why main paper: this is the strongest direct rebuttal to a “late-stage only” interpretation.
- Source files:
  - `{pass1_root / 'success_vs_budget.csv'}`
  - `{pass1_root / 'success_vs_budget.png'}`

### Figure 2. Hard-State Analysis
- Purpose: show that RL gains concentrate in difficult states and appear early there as well.
- Supporting claims: `C3`, `C12`
- Why main paper: this gives the cleanest mechanism-facing story without overclaiming.
- Source files:
  - `{pass2_root / 'hard_state_mechanism_tables.csv'}`
  - `{pass2_root / 'hard_state_success_budget.png'}`
  - `{pass1_root / 'difficulty_bucket_tables.csv'}`

### Figure 3. Paired Case Breakdown
- Purpose: show where the gain comes from at the case level: RL-only wins, both-hit-but-RL-earlier, Greedy-only wins, and shared failures.
- Supporting claims: `C10`
- Why main paper: this is more directly interpretable than the policy-overlap plot for a compact paper.
- Source files:
  - `{pass1_root / 'case_taxonomy.csv'}`
  - `{pass1_root / 'case_taxonomy.png'}`

## Appendix

### Table A1. Bounded Hybrid Replay Complementarity Analysis
- Purpose: show why the mechanism claim must stay cautious and why the evidence is more consistent with later-pick completion than with a pure first-pick explanation.
- Supporting claims: `C5`, `C11`
- Why appendix: useful mechanism evidence, but bounded-subset and synthetic by design.
- Source files:
  - `{pass2_root / 'complementarity_tables.csv'}`
  - `{pass2_root / 'complementarity_hybrid_success.png'}`

### Figure A1. Policy Difference / Overlap
- Purpose: show that RL is not simply copying Greedy’s selected 3-sets.
- Supporting claims: `C4`
- Why appendix: important but secondary to the main paired taxonomy figure.
- Source files:
  - `{pass1_root / 'policy_behavior_tables.csv'}`
  - `{pass1_root / 'policy_overlap.png'}`

### Figure A2. Slot-Level Tendency
- Purpose: show the exploit-first / then-complement tendency while explicitly avoiding fixed-role language.
- Supporting claims: `C6`, `C7`
- Why appendix: mechanism nuance, not headline evidence.
- Source files:
  - `{pass2_root / 'slot_role_tendency_tables.csv'}`
  - `{pass2_root / 'slot_role_tendency.png'}`

### Figure A3. Relaxed Radius Curves
- Purpose: show that the relaxed-radius metrics are supportive secondary evidence while keeping exact hit primary.
- Supporting claims: `C9`
- Why appendix: secondary metric, not the main task.
- Source files:
  - `{pass1_root / 'relaxed_metric_tables.csv'}`
  - `{pass1_root / 'relaxed_radius_curves.png'}`

### Figure A4. Representative Case Panels
- Purpose: provide concrete examples of RL-only wins, RL-earlier wins, Greedy-only wins, and shared failures.
- Supporting claims: `C10`, `C12`
- Why appendix: useful for reviewers and collaborators, but too detailed for a compact main paper.
- Source files:
  - `{pass2_root / 'representative_case_figures/figure_manifest.csv'}`
  - representative PNG panels under `{pass2_root / 'representative_case_figures'}`

### Table A2. Frontier Composition Boundary Check
- Purpose: document that a strong novelty-frontier or mixed-frontier mechanism is not supported.
- Supporting claims: `C8`
- Why appendix: this is primarily a boundary-control artifact.
- Source files:
  - `{pass2_root / 'frontier_composition_tables.csv'}`
  - `{pass2_root / 'frontier_composition.png'}`

## Optional Supplement

### Supplement S1. Full Observational Overlap Detail
- Purpose: preserve the full-panel early-divergence evidence without overloading the appendix.
- Supporting claims: `C4`, `C5`
- Source files:
  - `{pass2_root / 'complementarity_observational_case_rows.csv'}`
  - full-panel observational rows inside `{pass2_root / 'complementarity_tables.csv'}`

### Supplement S2. Targeted Replay Raw Rows
- Purpose: provide raw targeted replay case and slot tables for auditability.
- Supporting claims: `C5`, `C6`, `C11`
- Source files:
  - `{pass2_root / 'targeted_hybrid_replay_case_rows.csv'}`
  - `{pass2_root / 'targeted_hybrid_replay_slot_rows.csv'}`
"""


def experiment_section_en(pass1_root: Path, pass2_root: Path) -> str:
    return f"""# Experiment Section Draft

## Evaluation Setup

We evaluate all headline results on a held-out validation panel under a fixed `B30` budget, corresponding to 10 rounds with 3 node queries per round. The posterior core is kept fixed to SPIM-v3 throughout the main comparison. The strongest RL policy is evaluated using the strict held-out runner, and the heuristic baselines are taken from the aligned teacher5 comparison artifact. The main contract is therefore shared in split (`val`), budget (`B30`), posterior family (`hsr_soft_scenario_posterior_v3`), and success definition (exact source-node hit within budget). The authoritative artifact references are `{pass1_root / 'run_manifest.json'}` and `{pass2_root / 'run_manifest.json'}`.

## Compared Methods

The main comparison includes Strongest RL, Greedy Posterior, Thompson Sampling, Entropy Reduction, Cover Shrink, and Disagreement Split. Our intended interpretation is stage-wise: SPIM-v3 provides the posterior / belief core, while RL learns a budgeted 3-action set-level sampling policy on top of that core rather than replacing posterior inference itself.

## Main Metric And Secondary Metrics

The primary metric is exact held-out node-level Success@B30. This exact-hit metric remains the headline endpoint. We additionally report the success-versus-budget curve over budgets 1 to 30, paired win/loss statistics versus Greedy Posterior, and secondary relaxed-radius success curves. The relaxed-radius metrics are reported only as supplementary evidence and are not used to replace the primary exact-hit result.

## Main Results

Under the aligned held-out val/B30/SPIM-v3 exact-hit contract, Strongest RL achieves Success@B30 of 0.923375, compared with 0.899127 for Greedy Posterior, 0.803104 for Thompson Sampling, 0.554801 for Entropy Reduction, 0.512124 for Cover Shrink, and 0.482056 for Disagreement Split (`comparison_table.csv`). Thus, the strongest RL policy outperforms Greedy Posterior and all four heuristic baselines under the same evaluation contract.

## Paired-Case Analysis

The aggregate gain is also supported by paired evidence against Greedy Posterior. RL records 41 RL-only wins versus 16 Greedy-only wins, for a net flip of +25 and an exact McNemar p-value of 0.00126356 (`comparison_table.csv`). The paired taxonomy further shows that the gain is not driven solely by a small number of unique wins: in addition to 41 RL-only wins, there are 209 cases in which both methods succeed but RL reaches the source materially earlier, compared with 54 cases in which Greedy is earlier (`case_taxonomy.csv`).

## Hard-State Analysis

The relative gain of RL is larger in more difficult initial states. In the highest-entropy bucket, RL reaches 0.856589 compared with 0.806202 for Greedy Posterior (`difficulty_bucket_tables.csv`). The hard-state budget curves further show that this advantage is already visible in early budgets, rather than emerging only near budget exhaustion (`hard_state_mechanism_tables.csv`, `hard_state_success_budget.png`). Together, these results support the claim that RL is particularly beneficial in harder, more ambiguous states.

## Mechanism Analysis

Mechanism evidence should be stated cautiously. First, RL is not simply copying Greedy Posterior at the set level: the exact 3-set match rate is only 0.053838 and the mean Jaccard overlap is 0.221139 (`policy_behavior_tables.csv`). Second, full-panel observational overlap analysis shows that RL and Greedy typically diverge very early, especially in RL-only-win cases (`complementarity_tables.csv`). Third, bounded hybrid replay on a fixed 24-case hard subset suggests that the gain is more consistent with later-pick completion and set-level complementarity than with a pure first-pick advantage: on that subset, RL1+G23 reaches 0.708333, whereas G1+RL23 and RL12+G3 each reach 0.875000 (`complementarity_tables.csv`). However, because this replay analysis is synthetic and subset-bounded, it should be interpreted as mechanism evidence rather than as a new benchmark.

## Secondary Relaxed-Metric Analysis

Secondary relaxed-radius curves are consistent with the primary exact-hit result: RL remains stronger than Greedy Posterior at radius 0, 1, and 2 (`relaxed_metric_tables.csv`). These curves provide practical context but do not replace the exact-hit primary endpoint.

## Boundary-Control Language

Two forms of overclaim must be avoided. First, the current evidence does not justify fixed three-role semantic interpretations of the three selected points. The slot-level results support only a tendency, not stable semantic heads (`slot_role_tendency_tables.csv`). Second, the current evidence does not support a strong novelty-frontier or mixed-frontier mechanism claim: the targeted frontier-composition analysis remains strongly posterior-dominated (`frontier_composition_tables.csv`)."""


def collaborator_note_cn(pass1_root: Path, pass2_root: Path) -> str:
    return f"""# 协作说明（中文）

## 可以自信主张的内容

- 可以明确主张：在锁定的 `held-out val / B30 / SPIM-v3 / exact node hit` 合同下，最强 RL 优于 Greedy Posterior 和其余 4 个 heuristic baseline。直接证据是 `{pass1_root / 'comparison_table.csv'}`。
- 可以明确主张：RL 的优势不是只在最后预算段才出现，预算曲线前段已经拉开。直接证据是 `{pass1_root / 'success_vs_budget.csv'}` 和 `{pass1_root / 'success_vs_budget.png'}`。
- 可以明确主张：RL 的优势更集中在 hard states，尤其是高 entropy / 大 support / 低 margin 的状态。直接证据是 `{pass1_root / 'difficulty_bucket_tables.csv'}` 和 `{pass2_root / 'hard_state_mechanism_tables.csv'}`。
- 可以明确主张：RL 不是简单复制 Greedy 的 3-set。直接证据是 `{pass1_root / 'policy_behavior_tables.csv'}`。

## 需要谨慎表述的内容

- 关于 complementarity，只能说“现有证据更支持 later-pick completion / set-level complementarity，而不是单纯更好的 first pick”。这里必须强调证据来自 bounded hybrid replay 子集，而不是新的 full-panel benchmark。直接证据是 `{pass2_root / 'complementarity_tables.csv'}`。
- 关于 slot-level 结构，只能说“存在 tendency：前面更 exploitative，后面更 dispersed / complementary”，不能说“学出了三个固定角色”。直接证据是 `{pass2_root / 'slot_role_tendency_tables.csv'}`。
- 关于 hard-state 机制，只能说“与更早的 ambiguity reduction 一致”，不能写成已经证明了严格因果机制。直接证据是 `{pass2_root / 'hard_state_mechanism_tables.csv'}`。

## 不应该主张的内容

- 不要说“固定 three semantic heads 已经出现”。
- 不要说“RL 的主机制已经被证明是 novelty-frontier / mixed-frontier selection”。
- 不要把 bounded hybrid replay 结果写成新的 headline benchmark。
- 不要把 relaxed radius metric 提升成主指标；主指标始终是 exact Success@B30。

## 建议的汇报 / defense 叙事

建议把故事压缩成四层：

1. `posterior core` 没变，SPIM-v3 仍然是 stage-1 belief core。
2. RL 的贡献发生在 stage-2：在固定 belief 之上学一个预算受限的 3 点 set-level sampling policy。
3. headline result 是 exact held-out Success@B30 提升，并且这个提升在 early budget 和 hard states 中更明显。
4. 机制层面只给 controlled story：RL 明显不同于 Greedy，且更像在后续 pick completion 上获益；slot-level 只到 tendency，不到 fixed roles；frontier-mixing 没有被支持。"""


def captions_en() -> str:
    return """# Caption Suggestions (English)

## Main Comparison Table
Table 1. Performance under the aligned held-out val/B30/SPIM-v3 exact-hit contract. Strongest RL improves Success@B30 over Greedy Posterior and all four posterior-heuristic baselines while preserving the same posterior core and evaluation contract.

## Success-vs-Budget Figure
Figure 1. Cumulative success rate as a function of budget from 1 to 30. The advantage of RL appears early in the budget curve rather than only at the end of the episode.

## Hard-State Figure
Figure 2. Hard-state analysis on the highest-entropy subset. RL maintains a larger success advantage over Greedy Posterior in difficult states, with separation already visible in early budgets and early-round belief trajectories.

## Paired Taxonomy Figure
Figure 3. Paired RL-versus-Greedy case taxonomy. The overall gain combines RL-only wins with a substantial both-hit-but-RL-earlier bucket, rather than depending on a small number of isolated flips.

## Policy-Overlap Figure
Figure A1. Set-level overlap between RL and Greedy Posterior. The low exact set-match rate and modest Jaccard overlap show that RL is not simply reproducing Greedy’s 3-set decisions.

## Relaxed-Radius Figure
Figure A2. Relaxed radius-based success curves reported as secondary evidence. Exact hit remains the primary endpoint; relaxed radius provides additional practical context without replacing the main metric or the locked held-out contract.

## Case-Study Figure
Figure A3. Representative case studies comparing RL and Greedy Posterior over the first few rounds. These examples illustrate where RL helps, where it fails, and why the mechanism claims must remain controlled.
"""


def captions_cn() -> str:
    return """# 图表标题建议（中文）

- 表 1：`在对齐的 held-out val/B30/SPIM-v3 exact-hit 合同下的主结果比较。Strongest RL 在保持相同 posterior core 的前提下优于 Greedy Posterior 和其余四个 heuristic baseline。`
- 图 1：`预算 1 到 30 下的累计成功率曲线。RL 的优势在早期预算段已经出现，而不是仅在预算末端累积。`
- 图 2：`最高熵子集上的 hard-state 分析。RL 在困难状态中的优势更明显，并且在早期预算和早期 belief 轨迹中已可见。`
- 图 3：`RL 与 Greedy Posterior 的 paired taxonomy。整体增益既来自 RL-only wins，也来自大量 both-hit-but-RL-earlier 的案例。`
- 图 A1：`RL 与 Greedy Posterior 的 3-set overlap。低 exact match 和有限 Jaccard overlap 表明 RL 并非简单复制 Greedy 的选点集合。`
- 图 A2：`作为次级证据报告的 relaxed radius 成功率曲线。主指标仍然是 exact Success@B30。`
- 图 A3：`代表案例图。展示 RL 与 Greedy 在前几轮 3-set、belief trajectory 和最终结果上的差异。`
"""


def discussion_en(pass2_root: Path) -> str:
    return f"""# Discussion, Limitations, And Non-Claims

## Interpretation of the RL Gain

The strongest headline claim is empirical rather than speculative: under the aligned held-out val/B30/SPIM-v3 exact-hit contract, the strongest RL policy outperforms Greedy Posterior and the other posterior-based heuristics. The budget curves and paired-case analysis indicate that this advantage appears early and is not merely a late-budget artifact. The hard-state analysis further suggests that the gain is concentrated in more ambiguous initial states. Mechanistically, the evidence supports a controlled interpretation: RL differs materially from Greedy at the set level and the bounded hybrid replay is more consistent with later-pick completion than with a pure first-pick explanation.

## Limitations of the Current Mechanism Evidence

The mechanism evidence remains bounded. The hybrid replay analysis is based on a fixed 24-case hard subset with synthetic prefix/suffix combinations and should therefore be interpreted as supportive mechanism evidence rather than as a new benchmark. The slot-level analysis reveals only statistical tendencies, not fixed semantic heads. Moreover, the frontier-composition analysis does not support a strong novelty-frontier or mixed-frontier account; the targeted replay remains strongly posterior-dominated (`{pass2_root / 'frontier_composition_tables.csv'}`).

## What Is Explicitly Not Claimed

We do not claim that three fixed semantic roles emerged. We do not claim that the main mechanism is novelty-frontier selection or balanced multi-frontier mixing. We do not claim that the bounded hybrid replay supersedes the original strongest RL checkpoint on the full benchmark. Finally, we do not replace exact Success@B30 with relaxed-radius success; the relaxed metrics remain secondary only."""


def reviewer_memo(pass1_root: Path, pass2_root: Path) -> str:
    return f"""# Reviewer Defense Memo

## Attack 1. “Is RL just benefiting from a different metric?”
- Answer: No. The headline comparison is locked to the aligned held-out val/B30/SPIM-v3 exact-hit contract, and exact Success@B30 remains the primary endpoint.
- Supporting artifacts:
  - `{pass1_root / 'comparison_table.csv'}`
  - `{pass1_root / 'analysis_summary.md'}`
- Caveat: relaxed-radius metrics are reported only as secondary context.

## Attack 2. “Is the improvement only tiny or only visible at the very end?”
- Answer: No. The advantage is visible in the early budget curve and is supported by paired-case evidence, including a large both-hit-but-RL-earlier bucket.
- Supporting artifacts:
  - `{pass1_root / 'success_vs_budget.csv'}`
  - `{pass1_root / 'case_taxonomy.csv'}`
- Caveat: the absolute effect size is still moderate rather than transformative, so the writing should stay controlled.

## Attack 3. “Is the mechanism story overclaimed?”
- Answer: The manuscript package explicitly separates proven, partially proven, and not-proved claims. The mechanism story is limited to set-level difference, early hard-state advantage, and slot-level tendency.
- Supporting artifacts:
  - `{pass2_root / 'analysis_summary_pass2.md'}`
  - `{pass2_root / 'slot_role_tendency_tables.csv'}`
  - `{pass2_root / 'frontier_composition_tables.csv'}`
- Caveat: fixed semantic roles and strong mixed-frontier claims are explicitly avoided.

## Attack 4. “Why exact hit as the main metric and relaxed radius only secondary?”
- Answer: Exact hit is the strict task-aligned endpoint and was locked as the main metric throughout the analysis. Relaxed-radius curves provide practical context but do not replace the primary objective.
- Supporting artifacts:
  - `{pass1_root / 'comparison_table.csv'}`
  - `{pass1_root / 'relaxed_metric_tables.csv'}`
- Caveat: secondary metrics can be mentioned as supportive, not as the basis of the main claim.

## Attack 5. “Why not claim novelty-frontier behavior?”
- Answer: Because the targeted frontier-composition analysis does not support that claim. The selected-node composition remains strongly posterior-dominated.
- Supporting artifacts:
  - `{pass2_root / 'frontier_composition_tables.csv'}`
  - `{pass2_root / 'frontier_composition.png'}`
- Caveat: the full-panel slate counts do show small disagreement/novelty components, but not enough to justify a strong mechanism claim.

## Attack 6. “Why does hybrid replay sometimes outperform the original RL checkpoint on the subset?”
- Answer: The hybrid replay is a bounded-subset mechanism probe using synthetic prefix/suffix combinations. It is not a new held-out benchmark and should not be interpreted as proving global suboptimality of the original RL checkpoint.
- Supporting artifacts:
  - `{pass2_root / 'complementarity_tables.csv'}`
  - `{pass2_root / 'run_manifest.json'}`
- Caveat: this is exactly why the complementarity claim remains only partially proven.
"""


def analysis_summary_pass3(pass1_root: Path, pass2_root: Path) -> str:
    return f"""# Pass-3 Paper Freeze Summary

## Scope
- This pass does not add new experiments, new training, or new broad evaluation.
- It freezes the paper story by converting the verified pass-1 and pass-2 outputs into a claim-evidence ledger, a final figure/table plan, and manuscript-ready text assets.

## Headline Safe Claims
- [proven] Strongest RL beats Greedy Posterior and all four heuristic baselines under the aligned held-out val/B30/SPIM-v3 exact-hit contract.
- [proven] RL gains appear early in the budget curve rather than only near budget exhaustion.
- [proven] RL gains are larger in harder states.
- [proven] RL is not simply copying Greedy’s 3-set selections.

## Claims That Must Remain Cautious
- [partially proven] The gain is more consistent with later-pick completion and set-level complementarity than with a pure first-pick explanation.
- [partially proven] Slot-level tendency exists, but fixed semantic roles are not supported.
- [partially proven] The hard-state evidence is consistent with an early ambiguity-reduction interpretation, but not a proved causal mechanism.

## Claims To Avoid
- [not proven] Fixed three semantic heads.
- [not proven] Strong novelty-frontier or mixed-frontier main mechanism.
- [not proven] Any interpretation that treats the bounded hybrid replay as a new headline benchmark.

## Source Bundles
- Pass 1: `{pass1_root}`
- Pass 2: `{pass2_root}`
"""


def main() -> None:
    args = parse_args()
    pass1_root = Path(args.pass1_root)
    pass2_root = Path(args.pass2_root)
    output_dir = ensure_output_dir(str(args.output_dir))

    ledger = build_claim_ledger(pass1_root, pass2_root)
    ledger.to_csv(output_dir / "claim_evidence_ledger.csv", index=False)

    (output_dir / "final_figure_table_plan.md").write_text(final_figure_table_plan_md(pass1_root, pass2_root), encoding="utf-8")
    (output_dir / "experiment_section_draft_en.md").write_text(experiment_section_en(pass1_root, pass2_root), encoding="utf-8")
    (output_dir / "collaborator_note_cn.md").write_text(collaborator_note_cn(pass1_root, pass2_root), encoding="utf-8")
    (output_dir / "caption_suggestions_en.md").write_text(captions_en(), encoding="utf-8")
    (output_dir / "caption_suggestions_cn.md").write_text(captions_cn(), encoding="utf-8")
    (output_dir / "discussion_limitations_nonclaims_en.md").write_text(discussion_en(pass2_root), encoding="utf-8")
    (output_dir / "reviewer_defense_memo.md").write_text(reviewer_memo(pass1_root, pass2_root), encoding="utf-8")
    (output_dir / "analysis_summary_pass3.md").write_text(analysis_summary_pass3(pass1_root, pass2_root), encoding="utf-8")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "pass1_root": str(pass1_root),
        "pass2_root": str(pass2_root),
        "generated_outputs": [
            "claim_evidence_ledger.csv",
            "final_figure_table_plan.md",
            "experiment_section_draft_en.md",
            "collaborator_note_cn.md",
            "caption_suggestions_en.md",
            "caption_suggestions_cn.md",
            "discussion_limitations_nonclaims_en.md",
            "reviewer_defense_memo.md",
            "analysis_summary_pass3.md",
            "run_manifest.json",
        ],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
