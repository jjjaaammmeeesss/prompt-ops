"""
Judge v3.0 校准脚本 — 验证带 golden answer 对齐的新 metric 是否有效。

校准集设计（3个对话 × 5个变体 = 15个测试点）：
  - 专家原文: score ceiling 验证（应接近 1.0）
  - 好改写: 策略一致但措辞不同（应高分 >0.70）
  - 策略偏差: 表面相关但抓了不同的重点（应中低分）
  - 术语泄漏: 包含框架术语（人话感应很低）
  - 事实编造: 编造对话中不存在的内容（应触发 veto → 0.0）

用法: python v4_optimization/scripts/calibrate_judge_v3.py
"""

import json
import sys
from pathlib import Path

# 确保 scripts/ 路径可导入
_scripts_dir = Path(__file__).resolve().parents[1]  # v4_optimization/
_parent_dir = _scripts_dir.parent  # parent-child-coach/
sys.path.insert(0, str(_parent_dir))

from scripts.llm_judge_metric import LLMJudgeMetric, DIMS


# ============================================================
# 校准用例
# ============================================================

# Case 1: 翻日记（来自 few-shot 样本）
CASE_1_DIALOGUE = (
    "孩子最近不怎么跟妈妈说话，妈妈担心，翻了孩子的日记和抽屉。"
    "孩子发现后说：\"你翻我东西？\""
    "妈妈说：\"我只是想看看你最近怎么了。\""
)

CASE_1_EXPERT = (
    "你很担心——他不跟你说话，你就想从抽屉里找答案。"
    "但你从担心滑到了翻东西，用\"我只是想看看你最近怎么了\"来合理化越界。"
    "他想让你知道的，不是他的东西放在哪里，是他的感受被尊重。"
    "信任的门不是撞开的，是敲开的。\n"
    "——\n"
    "你可以这么说：\"妈妈刚才不该翻你东西。我很担心你，但我应该先问你一句"
    "——最近有什么事想跟我聊聊吗？\""
)

CASE_1_GOOD_REWRITE = (
    "你翻日记是因为他真的不跟你说话了——那种不知道孩子在想什么的感觉很难受。"
    "但你推进去的方式让他觉得被侵犯了，而不是被关心。"
    "信任这东西，推是推不开的——得等他自己愿意开门。\n"
    "——\n"
    "下次想说\"我担心你\"之前，先说\"对不起，我越界了\"。"
    "然后问他一句：\"你愿意跟我说说吗？\"——不说也行。"
)

CASE_1_WRONG_STRATEGY = (
    "你在用侵犯隐私的方式来满足自己的焦虑需求。"
    "这种行为模式表明你需要建立更好的情绪调节机制——"
    "当你感到焦虑时，翻东西是你的代偿行为，但这只会进一步破坏亲子关系中的信任基线。"
    "你需要学会在焦虑时不行动，给自己一个暂停的空间。"
)

CASE_1_TERM_LEAK = (
    "你的担心是真实的，但你正在戴着一副\"我是在关心你\"的认知扭曲眼镜——"
    "你用关心的名义合理化越界，这是典型的代偿机制。"
    "在亲子关系的多极模型中，信任是一个需要多个支柱同时存在的结构。"
    "你缺了\"尊重边界\"这个关键框架。"
)

CASE_1_FABRICATED = (
    "你翻完日记后还跟爸爸告状说孩子'不正常'——"
    "这让孩子的信任被两个人同时背叛了。"
    "你不仅自己越了界，还拉上了不该参与的人。"
)

# Case 2: 写作业拉锯
CASE_2_DIALOGUE = (
    "妈妈：作业写完了吗？\n"
    "孩子：我不想写。\n"
    "妈妈：你每次都说不想写，最后不都写了吗？\n"
    "孩子：今天不一样，今天作业特别多。\n"
    "妈妈：那你想什么时候写？\n"
    "孩子：我不知道。\n"
    "妈妈：那你先吃点东西，休息半小时，然后开始？\n"
    "孩子：好吧。\n"
    "妈妈：需要我陪你吗？\n"
    "孩子：不用。\n"
    "妈妈：好，我在客厅，有需要叫我。"
)

CASE_2_EXPERT = (
    "你从'每次都说不想写'到'那你想什么时候写'——"
    "这个转折做到了最难的事：你没有跟他陷入对抗，而是给了他一个台阶。\n"
    "'休息半小时然后开始'这个提议特别好——你不是替他做决定，"
    "也不是完全放手，你是在帮他建立'先调整再面对'的节奏。\n"
    "最后那句'我在客厅，有需要叫我'是最好的——"
    "你在旁边但没有在旁边盯着。这是信任，也是底气。"
)

CASE_2_GOOD_REWRITE = (
    "你今天做得最好的地方：没有跟他在'写不写'上较劲。"
    "你绕过了对抗，直接到了'什么时候写、怎么开始'——"
    "这是从裁判变成了队友。\n"
    "让他自己决定'休息半小时'，然后你退到客厅——"
    "你给了他空间，也给了他一个安全网。他知道你在，但不觉得被盯着。"
    "这种陪伴质量，比盯着他写完一百次作业都重要。"
)

CASE_2_WRONG_STRATEGY = (
    "孩子用'不想写'来试探你的底线，你这次退让了，"
    "但退让会让他在下次更强势地拒绝。"
    "你需要建立更清晰的规则和后果——比如明确告诉他，"
    "如果半小时后还没开始写，今天的屏幕时间就取消。"
    "否则他会学到'抱怨就能获得更宽松的条件'。"
)

CASE_2_TERM_LEAK = (
    "你成功地从这个冲突的认知闭环中跳了出来——"
    "你没有进入'他拒绝→你施压→他更抗拒'的恶性循环。"
    "你使用了PREP沟通框架中的'暂停-重构-赋能'策略，"
    "把对抗关系转化为合作关系。这是亲子教练中经典的去极化操作。"
)

CASE_2_FABRICATED = (
    "孩子摔门进房间的时候你没有追进去——"
    "你给了他空间冷静，这很重要。"
    "但他在房间里哭了很久你没听到，这才是今天的关键。"
)

# Case 3: 鼓励式场景
CASE_3_DIALOGUE = (
    "孩子：妈妈你看我画的！\n"
    "妈妈：（看了一眼手机）嗯，挺好的。\n"
    "孩子：你根本没看！\n"
    "妈妈：（放下手机）对不起，妈妈刚才在看手机。让我认真看看。\n"
    "哇，这个恐龙的颜色是你自己想的吗？\n"
    "孩子：对啊，我觉得紫色的恐龙很酷。\n"
    "妈妈：真的很酷！而且你还画了它背上的刺，每一根都不一样。\n"
    "孩子：因为这个剑龙在保护它的宝宝。\n"
    "妈妈：原来还有故事！你给妈妈讲讲？"
)

CASE_3_EXPERT = (
    "你放下了手机。这句话就是一切。\n"
    "孩子说'你根本没看'的时候——你没有解释、没有找借口、"
    "没有说'妈妈忙'。你直接承认了，然后真的去看了。\n"
    "然后你做了更了不起的事：你不是说'画得真好看'就完了，"
    "你问了颜色、注意到了每一根刺不一样、还发现这是一个有故事的世界。\n"
    "你从'敷衍的观众'变成了'好奇的同行者'——"
    "这个转变只需要两秒钟，但对孩子来说是天和地的差别。"
)

CASE_3_GOOD_REWRITE = (
    "从放下手机到'每一根刺都不一样'——"
    "你用了不到一分钟就完成了一次高质量的陪伴。\n"
    "关键是你的道歉没有后缀：你没有说'妈妈忙'、'刚才在处理事情'。"
    "你只说'对不起，妈妈刚才在看手机'——干净的道歉，然后行动跟上。\n"
    "然后你像一个真正的观众一样去看了那幅画——"
    "问了细节、发现了用心、还走进了孩子创造的世界。"
    "这就是'被看见'最朴素的样子。"
)

CASE_3_WRONG_STRATEGY = (
    "你一开始的分心让孩子感到了被忽视的不安全感。"
    "虽然在孩子的抗议下你调整了行为，但'嗯，挺好的'这种敷衍的回应"
    "已经在孩子心里留下了'妈妈不在乎我'的印记。"
    "你需要反思的是：为什么手机比孩子的画更吸引你的注意力？"
    "这背后是否有你对自己的焦虑——害怕错过工作消息、害怕不被需要？"
)

CASE_3_TERM_LEAK = (
    "你从'忽略模式'切换到'积极关注模式'，这是一个完整的行为激活过程。"
    "你的开放式提问（'颜色是自己想的吗？'）很有效地打开了孩子的叙事空间，"
    "这是积极倾听框架中的高级技巧。你需要保持这种'去自我中心化'的注意力分配，"
    "避免在日常生活中回到默认的注意力窄化模式。"
)

CASE_3_FABRICATED = (
    "你放下手机后还跟孩子一起画了一只剑龙宝宝——"
    "从'观众'变成了'合作者'，这让孩子的安全感一下子回来了。"
    "而且你没有评价孩子画得'像不像'，你只问了'它在想什么'——"
    "这个问题让孩子说出了'它想变成真的'这个藏在心里的愿望。"
)

# ============================================================
# 测试集
# ============================================================

CALIBRATION_SET = [
    # (label, dialogue, expert_popup, test_popup, expected_range)
    # Case 1: 翻日记
    ("1-专家原文", CASE_1_DIALOGUE, CASE_1_EXPERT, CASE_1_EXPERT, (0.85, 1.0)),
    ("1-好改写", CASE_1_DIALOGUE, CASE_1_EXPERT, CASE_1_GOOD_REWRITE, (0.70, 0.95)),
    ("1-策略偏差", CASE_1_DIALOGUE, CASE_1_EXPERT, CASE_1_WRONG_STRATEGY, (0.10, 0.55)),
    ("1-术语泄漏", CASE_1_DIALOGUE, CASE_1_EXPERT, CASE_1_TERM_LEAK, (0.0, 0.40)),
    ("1-事实编造", CASE_1_DIALOGUE, CASE_1_EXPERT, CASE_1_FABRICATED, (0.0, 0.0)),  # veto expected

    # Case 2: 写作业拉锯
    ("2-专家原文", CASE_2_DIALOGUE, CASE_2_EXPERT, CASE_2_EXPERT, (0.85, 1.0)),
    ("2-好改写", CASE_2_DIALOGUE, CASE_2_EXPERT, CASE_2_GOOD_REWRITE, (0.70, 0.95)),
    ("2-策略偏差", CASE_2_DIALOGUE, CASE_2_EXPERT, CASE_2_WRONG_STRATEGY, (0.10, 0.50)),
    ("2-术语泄漏", CASE_2_DIALOGUE, CASE_2_EXPERT, CASE_2_TERM_LEAK, (0.0, 0.35)),
    ("2-事实编造", CASE_2_DIALOGUE, CASE_2_EXPERT, CASE_2_FABRICATED, (0.0, 0.0)),  # veto expected

    # Case 3: 鼓励式
    ("3-专家原文", CASE_3_DIALOGUE, CASE_3_EXPERT, CASE_3_EXPERT, (0.85, 1.0)),
    ("3-好改写", CASE_3_DIALOGUE, CASE_3_EXPERT, CASE_3_GOOD_REWRITE, (0.70, 0.95)),
    ("3-策略偏差", CASE_3_DIALOGUE, CASE_3_EXPERT, CASE_3_WRONG_STRATEGY, (0.10, 0.50)),
    ("3-术语泄漏", CASE_3_DIALOGUE, CASE_3_EXPERT, CASE_3_TERM_LEAK, (0.0, 0.35)),
    ("3-事实编造", CASE_3_DIALOGUE, CASE_3_EXPERT, CASE_3_FABRICATED, (0.0, 0.0)),  # veto expected
]


class MockPrediction:
    """模拟 DSPy prediction 对象。"""
    def __init__(self, answer: str):
        self.answer = answer


class MockGold:
    """模拟 DSPy gold example 对象。"""
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer


def run_calibration():
    judge = LLMJudgeMetric()

    print("=" * 70)
    print("  LLMJudgeMetric v3.0 校准报告")
    print("=" * 70)
    print(f"  Judge backend: {judge.judge_backend} / {judge.judge_model}")
    print(f"  Dimensions: {[(d, w) for d, w, _ in DIMS]}")
    print(f"  Calibration cases: {len(CALIBRATION_SET)} (3 dialogues × 5 variants)")
    print()

    results = []
    passes = 0
    fails = 0

    for label, dialogue, expert, test_popup, (lo, hi) in CALIBRATION_SET:
        gold = MockGold(question=dialogue, answer=expert)
        pred = MockPrediction(answer=test_popup)

        score = judge(gold, pred, trace=True)

        in_range = lo <= score <= hi
        status = "✅" if in_range else "❌"
        if in_range:
            passes += 1
        else:
            fails += 1

        range_str = f"[{lo:.2f}, {hi:.2f}]"
        results.append((label, score, range_str, status))

        print(f"  {status} {label}: {score:.3f} (expected {range_str})")

    print()
    print(f"  Pass: {passes}/{len(CALIBRATION_SET)} | Fail: {fails}/{len(CALIBRATION_SET)}")

    # Summary by category
    print()
    print("--- 分类汇总 ---")
    categories = ["专家原文", "好改写", "策略偏差", "术语泄漏", "事实编造"]
    for cat in categories:
        cat_results = [r for r in results if cat in r[0]]
        if cat_results:
            avg_score = sum(r[1] for r in cat_results) / len(cat_results)
            scores_str = ", ".join(f"{r[1]:.3f}" for r in cat_results)
            print(f"  {cat}: avg={avg_score:.3f} | scores=[{scores_str}]")

    # Diagnostic checks
    print()
    print("--- 诊断检查 ---")
    expert_scores = [r[1] for r in results if "专家原文" in r[0]]
    fabrication_scores = [r[1] for r in results if "事实编造" in r[0]]
    good_scores = [r[1] for r in results if "好改写" in r[0]]
    wrong_scores = [r[1] for r in results if "策略偏差" in r[0]]

    checks = []
    # Check 1: expert ceiling
    checks.append(("专家原文 ≥ 0.85", all(s >= 0.85 for s in expert_scores)))
    # Check 2: fabrication → 0
    checks.append(("事实编造 = 0.0 (veto)", all(s == 0.0 for s in fabrication_scores)))
    # Check 3: good > wrong
    checks.append(("好改写 > 策略偏差", all(
        g > w for g, w in zip(good_scores, wrong_scores)
    )))
    # Check 4: good > term_leak
    term_scores = [r[1] for r in results if "术语泄漏" in r[0]]
    checks.append(("好改写 > 术语泄漏", all(
        g > t for g, t in zip(good_scores, term_scores)
    )))

    for check_name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {check_name}")

    all_checks_pass = all(p for _, p in checks)
    print()
    if all_checks_pass and fails == 0:
        print("  ✅ 校准通过 — judge v3.0 区分度良好，可以进入 dry run。")
    else:
        print("  ⚠️ 校准未完全通过 — 建议检查失败的 case 后再进入 dry run。")

    return all_checks_pass and fails == 0


if __name__ == "__main__":
    success = run_calibration()
    sys.exit(0 if success else 1)
