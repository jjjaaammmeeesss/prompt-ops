"""v2.1 回归验证 — 5 个已知样本（ws08/ws09 专家打标窗口）。

验证 v2.1 的五个改动是否生效：
  - 行业惯例豁免（ws08丁 → 应安静）
  - 自然收束豁免（ws08戊、ws09丁 → 应安静）
  - 框架转换检测（ws09甲 → 应弹）
  - 决策权边界识别（ws09乙 → 应弹）

Usage:
    python regression_v21.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results" / "regression"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GEN_MODEL = os.environ.get("GEN_MODEL", "deepseek/deepseek-v4-pro")
GEN_API_BASE = os.environ.get("GEN_API_BASE", "https://api.deepseek.com/v1")
GEN_API_KEY = os.environ.get("GEN_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))
_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.3"))

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/glm-5.2")
JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", "https://qianfan.baidubce.com/v2")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", os.environ.get("GLM_API_KEY", ""))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("regression")


# ── 回归用例 ──

REGRESSION_CASES = [
    {
        "id": "regression_ws08_ding_industry_norm",
        "source": "ws08 丁窗口 · S4-合同条款谈判 句28-36",
        "expect": "安静",
        "expect_reason": "行业惯例豁免：甲方让乙方整合合同是标准商务动作，不应弹窗",
        "dialogue": (
            "可以 反正委员会里面你我和双方领导都在 不会让你单方面决定 那回到质保金 三个月后系统运行稳定 怎么定义稳定\n"
            "核心功能月度可用率百分之九十九点五以上 非核心功能百分之九十九以上 出现重大故障在四小时内响应修复\n"
            "这些标准可以 但是需要一个独立的监控平台 双方都能看到数据\n"
            "我们可以在系统里加一个监控面板 实时显示可用率 数据无法篡改\n"
            "好 还有一个问题 如果你们中途因为自身原因终止合作 怎么处理\n"
            "如果因为我们原因终止 已交付的代码和文档归你们所有 我们配合交接 已付款不退还 未付款不再支付\n"
            "那如果是因为我们这边的原因终止呢\n"
            "如果因你们原因终止 已交付的仍归你们 但已付款不退还 未交付部分按实际完成的工作量结算\n"
            "合理 那基本条款差不多了 你回去把上面这几个点整合到新合同里 我们下周签字"
        ),
    },
    {
        "id": "regression_ws08_wu_one_liner",
        "source": "ws08 戊窗口 · S4-合同条款谈判 句37",
        "expect": "安静",
        "expect_reason": "一句话收尾，纯确认下一步，不应弹窗",
        "dialogue": "好 我周四之前发你修订版",
    },
    {
        "id": "regression_ws09_jia_frame_shift",
        "source": "ws09 甲窗口 · S5-商务谈判与逼单 句1-9",
        "expect": "弹窗",
        "expect_reason": "框架转换检测：客户把'值不值'转成'超不超'，应弹且需点出框架转换",
        "expect_keywords": ["预算", "框"],
        "dialogue": (
            "李总 上次您说回去跟财务商量 有结果了吗\n"
            "商量了 方案我们认可 但价格方面财务觉得太高 我们今年的软件预算一共五十万 你一家就要吃掉三十五万 其他部门就什么都别干了\n"
            "三十五万三年 平均一年不到十二万 你们销售团队二十人 每人月成本八千 用系统后效率提升百分之二十 每月省三个人 一年回本\n"
            "你这个账算得是理想情况 实际上新系统上线员工要学习 要磨合 效率可能不升反降\n"
            "所以我们的合同里包含了两个月的驻场辅导 实施顾问跟着你们销售跑业务 帮每个人把系统用到日常里去\n"
            "驻场两个月不另外收费\n"
            "包含在实施费里了 实施费五万 加上三年的软件费三十万 总共三十五万\n"
            "那你能不能把实施费免了 三十五万降到三十万\n"
            "实施费免不了 顾问工资差旅都是成本 但我送你一个管理驾驶舱定制开发 按你们指标体系做一套大屏 平时给老板看数据用的"
        ),
    },
    {
        "id": "regression_ws09_yi_decision_boundary",
        "source": "ws09 乙窗口 · S5-商务谈判与逼单 句10-17",
        "expect": "弹窗",
        "expect_reason": "决策权边界识别：客户亮底牌'老板预算三十万'，应弹且识别为权限信号而非谈判筹码",
        "expect_keywords": ["老板", "预算", "三十万"],
        "dialogue": (
            "那个东西我们自己也能做吧 用Excel就行了\n"
            "Excel的数据滞后 我们大屏实时更新 做完以后季度会直接投屏 不用提前花一天做PPT 你上次说老板每季度会前熬一夜做数据\n"
            "是 我是熬了 但是这个功能值五万吗\n"
            "管理驾驶舱的定制开发我们对外报价是八万 算我送你的 抵实施费了 这样总价还是三十五万\n"
            "不行 我老板给我的预算就是三十万 你必须帮我想办法\n"
            "那这样 我把付款节奏拉长 你今年只需要付十万 明年付十万 后年付十万 这样不超你今年的预算 但是总价还是三十万\n"
            "等于你把后面两年的压力丢给以后的我了 万一明年预算更紧呢\n"
            "明年销售团队扩到三十人 系统价值更明显 到时申请预算比现在有说服力 我可在合同加一条 明年预算没批 你们可暂停使用 不产生额外费用"
        ),
    },
    {
        "id": "regression_ws09_ding_natural_close",
        "source": "ws09 丁窗口 · S5-商务谈判与逼单 句26-28",
        "expect": "安静",
        "expect_reason": "自然收束豁免：'好，这单我们定了'是干净收尾，不应弹窗",
        "dialogue": (
            "不用了 你说迁移以后没问题我就信你 但是合同里写的你得做到\n"
            "一定做到 我们维护了四十多家客户 没有一家因为系统稳定性解约的\n"
            "好 这单我们定了"
        ),
    },
]


# ── 生成 ──

def generate_popup(system_prompt: str, dialogue: str) -> str | None:
    import litellm

    user_msg = f"当前对话：\n{dialogue}"
    kwargs: dict = dict(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=_TEMPERATURE,
        max_tokens=4096,
        timeout=180,
    )
    if GEN_API_BASE:
        kwargs["api_base"] = GEN_API_BASE
    if GEN_API_KEY:
        kwargs["api_key"] = GEN_API_KEY

    for attempt in range(3):
        try:
            resp = litellm.completion(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:
            logger.warning("生成尝试 %d/3 失败: %s", attempt + 1, e)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    if not text or len(text) <= 10:
        return None
    if text.strip() == "安好":
        return None
    for prefix in ["弹窗：", "弹窗:", "【弹窗】", "输出："]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


# ── Judge（复用现有五维评审） ──

def _judge_popup(dialogue: str, popup_text: str) -> dict:
    import litellm

    prompt = f"""你是一位苛刻但公正的沟通教练评审。下面是一段两人对话，以及一个"沟通现场弹窗"AI 生成的弹窗。

弹窗的设计意图：作为一个独立第三方观察者，帮对话双方看见此刻还没看到的东西——纯洞察，不给建议，不站队任何一方，不称"你"而用场景临时功能角色。

【对话】
{dialogue}

【弹窗】
{popup_text}

请按五个维度打分（1-5 整数，5 最好）：

1. insight 洞察质量：准确性、相关性、深度、具体性
2. third_party 第三方立场：完全独立观察者，使用成对临时功能角色
3. language 语言质感：口语、假设语气、"也许""可能"、无术语
4. evidence 证据锚定：有原文锚点，不脑补内心活动
5. focus 聚焦度：只打一个盲区，不散不乱

再写一段 50 字以内的 comment。

只输出 JSON：{{"insight": 4, "third_party": 5, "language": 4, "evidence": 4, "focus": 4, "comment": "..."}}"""

    for attempt in range(3):
        try:
            resp = litellm.completion(
                model=JUDGE_MODEL,
                api_key=JUDGE_API_KEY,
                api_base=JUDGE_API_BASE,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
                timeout=180,
            )
            raw = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:
            logger.warning("judge 尝试 %d/3 失败: %s", attempt + 1, e)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    # Parse JSON
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"raw": raw, "parse_error": True}
    try:
        soft = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"raw": raw, "parse_error": True}

    defaults = {"insight": 3, "third_party": 3, "language": 3, "evidence": 3, "focus": 3, "comment": ""}
    for k, v in defaults.items():
        if k not in soft:
            soft[k] = v
    for k in ("insight", "third_party", "language", "evidence", "focus"):
        try:
            soft[k] = max(1, min(5, int(soft[k])))
        except (TypeError, ValueError):
            soft[k] = 3

    # Hard check
    violations = []
    n = len(re.sub(r"\s", "", popup_text.strip()))
    if n > 180:
        violations.append(f"字数 {n} > 180")
    elif n < 60:
        violations.append(f"字数 {n} < 60")
    if "你" in popup_text:
        violations.append("包含'你'字")
    if not popup_text.strip().endswith(("。", "？", "！", "”", '"', "」", "』")):
        violations.append("疑似截断")

    dims = ("insight", "third_party", "language", "evidence", "focus")
    soft_mean = sum(soft[k] for k in dims) / len(dims)
    penalty = min(1.5, 0.5 * len(violations))
    aggregate = round(max(1.0, soft_mean - penalty), 2)

    return {
        "aggregate": aggregate,
        "soft": soft,
        "hard": {"pass": not violations, "violations": violations},
        "comment": soft.get("comment", ""),
    }


# ── 主流程 ──

def main():
    if not GEN_API_KEY:
        logger.error("缺少生成模型 API key（设置 DEEPSEEK_API_KEY 或 GEN_API_KEY）")
        sys.exit(1)

    prompt_path = HERE / "system_prompt_v2.1.txt"
    if not prompt_path.exists():
        logger.error("找不到 v2.1 prompt: %s", prompt_path)
        sys.exit(1)
    system_prompt = prompt_path.read_text(encoding="utf-8")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 50)
    logger.info("v2.1 回归验证 — %d 个样本", len(REGRESSION_CASES))
    logger.info("=" * 50)

    results = []
    passed = 0
    failed = 0

    for case in REGRESSION_CASES:
        case_id = case["id"]
        expect = case["expect"]
        logger.info("\n▶ %s", case_id)
        logger.info("  来源: %s", case["source"])
        logger.info("  期望: %s — %s", expect, case["expect_reason"])

        try:
            popup_text = generate_popup(system_prompt, case["dialogue"])
        except Exception as e:
            logger.error("  生成失败: %s", e)
            results.append({"case_id": case_id, "expect": expect, "error": str(e), "pass": False})
            failed += 1
            continue

        if expect == "安静":
            if popup_text is None:
                logger.info("  ✅ PASS — 正确保持安静")
                results.append({
                    "case_id": case_id, "expect": expect, "popup_text": None,
                    "pass": True, "note": "正确保持安静",
                })
                passed += 1
            else:
                logger.warning("  ❌ FAIL — 应该安静但弹了: %s", popup_text[:100])
                results.append({
                    "case_id": case_id, "expect": expect, "popup_text": popup_text,
                    "pass": False, "note": f"应安静但弹窗: {popup_text[:120]}",
                })
                failed += 1

        elif expect == "弹窗":
            if popup_text is None:
                logger.warning("  ❌ FAIL — 应弹未弹")
                results.append({
                    "case_id": case_id, "expect": expect, "popup_text": None,
                    "pass": False, "note": "应弹未弹",
                })
                failed += 1
            else:
                # Judge
                if JUDGE_API_KEY:
                    try:
                        eval_result = _judge_popup(case["dialogue"], popup_text)
                    except Exception as e:
                        logger.error("  评审失败: %s", e)
                        eval_result = {"aggregate": 0, "soft": None, "hard": {"pass": False, "violations": [str(e)]}, "comment": str(e)}
                else:
                    eval_result = {"aggregate": 0, "soft": None, "hard": {"pass": True, "violations": []}, "comment": "无 Judge API key，跳过评审"}

                # 关键词检查
                keywords = case.get("expect_keywords", [])
                kw_hits = [kw for kw in keywords if kw in popup_text]

                logger.info("  弹窗: %s", popup_text[:150])
                if eval_result.get("aggregate", 0) > 0:
                    logger.info("  综合分: %.2f, 软维度: %s",
                                eval_result["aggregate"],
                                {k: eval_result["soft"][k] for k in ("insight", "third_party", "language", "evidence", "focus")} if eval_result.get("soft") else "N/A")
                if keywords:
                    logger.info("  关键词命中: %s/%s — %s", len(kw_hits), len(keywords), kw_hits)

                # 通过条件：综合分 >= 3.0 或有关键词命中
                passed_case = eval_result.get("aggregate", 0) >= 3.0
                if not passed_case and keywords:
                    passed_case = len(kw_hits) >= 1

                if passed_case:
                    logger.info("  ✅ PASS")
                    passed += 1
                else:
                    logger.warning("  ❌ FAIL — 分太低或无关键词命中")
                    failed += 1

                results.append({
                    "case_id": case_id, "expect": expect, "popup_text": popup_text,
                    "pass": passed_case,
                    "aggregate": eval_result.get("aggregate"),
                    "soft": eval_result.get("soft"),
                    "hard": eval_result.get("hard"),
                    "comment": eval_result.get("comment", ""),
                    "keyword_hits": kw_hits,
                    "note": "",
                })

    # ── 保存报告 ──
    report = {
        "meta": {
            "timestamp": timestamp,
            "prompt_version": "v2.1",
            "n_cases": len(REGRESSION_CASES),
            "gen_model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
        },
        "summary": {"passed": passed, "failed": failed, "total": len(REGRESSION_CASES)},
        "results": results,
    }

    out_path = RESULTS_DIR / f"regression_v21_{timestamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("\n报告: %s", out_path)

    # ── 终端输出 ──
    print("\n" + "=" * 60)
    print("  v2.1 回归验证结果")
    print("=" * 60)
    for r in results:
        status = "✅" if r["pass"] else "❌"
        print(f"  {status} {r['case_id']}")
        if not r["pass"] and r.get("note"):
            print(f"     → {r['note'][:120]}")
    print(f"\n  通过: {passed}/{len(REGRESSION_CASES)}, 失败: {failed}/{len(REGRESSION_CASES)}")
    print("=" * 60)

    return passed == len(REGRESSION_CASES)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
