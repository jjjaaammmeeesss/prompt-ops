"""End-to-end smoke test with mock LLM: 验证完整管道（不调真实 API）。"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Mock LLM client ─────────────────────────────────────

def make_mock_client():
    """创建 mock OpenAI 客户端，返回合理 JSON 响应。

    不依赖 system prompt 内容匹配（太脆弱），而是根据 response_format 判断。
    """
    client = MagicMock()
    call_count = [0]  # mutable counter

    def mock_create(*, model, messages, response_format=None, temperature=None, timeout=None, **kwargs):
        """返回通用但有效的 JSON，让 pipeline 能跑通。"""
        resp = MagicMock()
        choice = MagicMock()
        call_count[0] += 1

        if response_format and response_format.get("type") == "json_object":
            # 通用 JSON 响应：包含所有常见字段
            choice.message.content = json.dumps({
                # PerceptionAgent fields
                "emotion_track": "家长表现出明显的愤怒和沮丧",
                "belief_diagnosis": "家长认为孩子故意对抗",
                "child_state": "孩子沉默回避，不愿意交流",
                "relation_pattern": "长期的命令-反抗循环",
                "positive_moment": "",
                "positive_moment_category": "none",
                "response_need": "needs_diagnostic",
                "signals": {
                    "has_generalization": True,
                    "has_labeling": False,
                    "has_conflict_escalation": True,
                    "has_safety_emergency": False,
                    "child_core_need_unmet": True,
                    "parent_emotion_overload": True,
                },
                # MasterAgent fields
                "route_a_insight": "家长情绪过载导致泛化指责",
                "route_b_insight": "感知分析确认：孩子核心需求未被回应",
                "story_arc": "冲突升级中",
                "main_contradiction": "家长用指责表达关心，孩子用沉默表达抗议",
                "direction": "diagnostic",
                "tone_reasoning": "家长存在情绪盲区",
                "should_popup": True,
                "is_redundant_with_previous": False,
                "contradiction_flag": "",
                # ProductionAgent fields
                "popup_insight": "【诊断式弹窗 mock】你反复提到'你又这样'——看起来你已经受够了。"
                               "但孩子听到的不是'关心'而是'否定'。",
                "popup_suggestion": "试着换一种说法表达关心",
                "tone": "diagnostic",
                # Expert opinion fields
                "evidence": "家长说'你又这样'，表现出泛化倾向",
                "confidence": 0.7,
                "reasoning": "家长情绪过载但未识别孩子的需求",
                # Speaker fields
                "final_tone": "diagnostic",
                "resolution": "majority",
                # Validator fields
                "is_consistent": True,
                "actual_tone_in_draft": "diagnostic",
                "mismatch_evidence": "",
                "suggestion": "",
                # Deep Review fields
                "deep_tone": "diagnostic",
                "is_correction_needed": False,
                "narrative_impact": "",
                # M6/M7 judge fields
                "score": 4,
                "strength": "抓住了核心矛盾",
                "weakness": "表达可以更具体",
                "verdict": "adopt",
                # Proposer fields
                "failure_analysis": {
                    "prompt": ["C10-002"],
                    "judge": [],
                    "dataset": [],
                    "search": [],
                    "model": ["C11-009"]
                },
                "edits": [{
                    "target_file": "harness.md",
                    "before": "## Expert emotion",
                    "after": "## Expert emotion\n\n核心区分标准已更新",
                    "reason": "添加更明确的区分标准",
                    "affected_cases": ["C10-002"]
                }],
                "rationale": "添加 diagnostic vs empowering 的核心区分标准",
                "risks": "可能过于细化"
            }, ensure_ascii=False)
        else:
            choice.message.content = "Mock response"

        resp.choices = [choice]
        return resp

    client.chat.completions.create = mock_create
    call_count_ref = call_count
    return client


# ── 测试 ────────────────────────────────────────────────

def test_senate_adapter():
    """测试 Senate 适配器 process_window。"""
    print("Testing SenateAdapter...")
    from meta_optimize.strategy_adapters import SenateAdapter

    client = make_mock_client()
    adapter = SenateAdapter(client, model="mock")

    result = adapter.process_window(
        "小明，你又没写作业！我跟你说过多少遍了？你这样以后怎么办？",
        family="test_case"
    )

    assert result.should_popup, "Senate should popup"
    assert result.tone == "diagnostic", f"Expected diagnostic, got {result.tone}"
    assert result.popup_text, "Should have popup text"
    assert "[Senate vote:" in result.route_a_insight, f"route_a should show votes: {result.route_a_insight}"
    print(f"  ✅ Senate: tone={result.tone}, popup={result.popup_text[:60]}...")


def test_teacher_student_adapter():
    """测试 Teacher-Student 适配器 process_window。"""
    print("Testing TeacherStudentAdapter...")
    from meta_optimize.strategy_adapters import TeacherStudentAdapter

    client = make_mock_client()
    adapter = TeacherStudentAdapter(client, model="mock")

    result = adapter.process_window(
        "小明，你又没写作业！我跟你说过多少遍了？你这样以后怎么办？"
        "你现在不努力，将来怎么竞争？我不是在批评你，我是为你好。"
        "你看看别人家的孩子，哪个像你这样？",
        family="test_case"
    )

    assert result.should_popup, "TS should popup"
    assert result.tone == "diagnostic", f"Expected diagnostic, got {result.tone}"
    assert result.popup_text, "Should have popup text"
    assert "TS feedback" in result.route_b_insight, f"route_b should have feedback: {result.route_b_insight}"
    print(f"  ✅ Teacher-Student: tone={result.tone}, popup={result.popup_text[:60]}...")


def test_saga_adapter():
    """测试 SAGA 适配器 process_window。"""
    print("Testing SAGAAdapter...")
    from meta_optimize.strategy_adapters import SAGAAdapter

    client = make_mock_client()
    adapter = SAGAAdapter(client, model="mock")

    result = adapter.process_window(
        "小明，你又没写作业！我跟你说过多少遍了？你这样以后怎么办？"
        "你现在不努力，将来怎么竞争？我不是在批评你，我是为你好。"
        "你看看别人家的孩子，哪个像你这样？",
        family="test_case"
    )

    assert result.should_popup, "SAGA should popup"
    assert result.tone == "diagnostic", f"Expected diagnostic, got {result.tone}"
    assert result.popup_text, "Should have popup text"
    assert "Deep Review" in result.route_b_insight, f"route_b should mention Deep Review: {result.route_b_insight}"
    print(f"  ✅ SAGA: tone={result.tone}, popup={result.popup_text[:60]}...")


def test_evaluate_with_strategy():
    """测试 evaluate_with_strategy 使用 mock 客户端。"""
    print("Testing evaluate_with_strategy (1 case)...")
    from meta_optimize.run_meta_optimize import evaluate_with_strategy

    client = make_mock_client()

    # 仅评估一个 case（C10-001）
    report = evaluate_with_strategy(
        client, "mock", "senate",
        harness_dir=str(Path(__file__).parent / "candidates/senate/baseline"),
        eval_entries=[("C10-001", None)],
        verbose=True,
    )

    assert report.results, "Should have results"
    assert report.overall_score > 0, f"Overall should be > 0: {report.overall_score}"
    print(f"  ✅ evaluate: overall={report.overall_score:.3f}, "
          f"M5={report.aggregate_m5:.1%}, n_results={len(report.results)}")


def test_full_pipeline():
    """测试完整管道：evaluate → propose → apply → re-evaluate。"""
    print("\nTesting full pipeline (evaluate → propose → apply → re-evaluate)...")

    from meta_optimize.candidate_store import (
        load_trace, load_metrics, build_candidate_from_report,
        ParentRef,
    )
    from meta_optimize.proposer import heuristic_propose
    from meta_optimize.meta_loop import apply_edits_to_harness
    from meta_optimize.run_meta_optimize import evaluate_with_strategy

    client = make_mock_client()
    strategy = "senate"
    harness_dir = Path(__file__).parent / "candidates" / strategy / "baseline"

    # 1. Baseline
    report1 = evaluate_with_strategy(
        client, "mock", strategy, str(harness_dir),
        eval_entries=[("C10-001", None)],
    )
    harness_md = (harness_dir / "harness.md").read_text(encoding="utf-8")

    d1 = build_candidate_from_report(strategy, "baseline", report1, harness_md=harness_md)
    print(f"  基线: overall={report1.overall_score:.3f}, M5={report1.aggregate_m5:.1%}")

    # 2. Propose
    trace = load_trace(d1)
    metrics = load_metrics(d1)
    proposal = heuristic_propose(strategy, "candidate_001", trace, harness_md, metrics)
    print(f"  Proposer: {len(proposal.edits)} edits, rationale={proposal.rationale[:60]}...")

    # 3. Apply
    mutated = harness_md
    for edit in proposal.edits:
        mutated = apply_edits_to_harness(mutated, [edit])

    # 写入临时 harness
    tmp_dir = Path(__file__).parent / "candidates" / strategy / "candidate_001"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "harness.md").write_text(mutated, encoding="utf-8")

    # 4. Re-evaluate
    report2 = evaluate_with_strategy(
        client, "mock", strategy, str(tmp_dir),
        eval_entries=[("C10-001", None)],
    )

    d2 = build_candidate_from_report(
        strategy, "candidate_001", report2,
        harness_md=mutated,
        proposal=proposal.to_markdown(),
        parent=ParentRef(candidate_id="baseline", strategy=strategy, overall_score=report1.overall_score),
    )
    print(f"  候选: overall={report2.overall_score:.3f}, M5={report2.aggregate_m5:.1%}")

    # 5. Keep/discard
    from auto_evolve.optimizer import should_keep
    kept, reason = should_keep(report1, report2)
    print(f"  Keep: {kept}, Reason: {reason}")

    # Cleanup
    import shutil
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("  ✅ Full pipeline complete")


def test_comparison():
    """测试 L2 对比报告生成。"""
    print("\nTesting L2 comparison...")
    from meta_optimize.comparison import (
        explore_traces, build_case_matrix,
        generate_comparison_report, render_markdown_report,
    )
    from meta_optimize.candidate_store import CANDIDATES_DIR

    strategies = ["senate", "teacher_student", "saga"]
    all_traces = explore_traces(strategies)

    if not any(all_traces.values()):
        print("  ⚠ No traces yet (no evaluations run), skipping matrix build")
        return

    matrix = build_case_matrix(all_traces)
    report = generate_comparison_report(all_traces, matrix)
    md = render_markdown_report(report)

    print(f"  Matrix: {len(matrix)} cases")
    print(f"  Recommendations: {report.recommendation[:100]}...")
    print("  ✅ Comparison OK")


# ── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Meta-Optimizer E2E Smoke Test (Mock LLM)")
    print("=" * 60)

    all_passed = True

    try:
        test_senate_adapter()
    except Exception as e:
        print(f"  ❌ Senate failed: {e}")
        all_passed = False
        import traceback; traceback.print_exc()

    try:
        test_teacher_student_adapter()
    except Exception as e:
        print(f"  ❌ Teacher-Student failed: {e}")
        all_passed = False
        import traceback; traceback.print_exc()

    try:
        test_saga_adapter()
    except Exception as e:
        print(f"  ❌ SAGA failed: {e}")
        all_passed = False
        import traceback; traceback.print_exc()

    try:
        test_evaluate_with_strategy()
    except Exception as e:
        print(f"  ❌ evaluate_with_strategy failed: {e}")
        all_passed = False
        import traceback; traceback.print_exc()

    try:
        test_full_pipeline()
    except Exception as e:
        print(f"  ❌ Full pipeline failed: {e}")
        all_passed = False
        import traceback; traceback.print_exc()

    try:
        test_comparison()
    except Exception as e:
        print(f"  ❌ Comparison failed: {e}")
        all_passed = False
        import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if all_passed else 1)
