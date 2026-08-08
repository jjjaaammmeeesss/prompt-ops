"""v4.0 多场景测试 — 覆盖 6 种卦象的弹窗质量验证。

用法: cd use-cases/parent-child-coach && python scripts/test_v4_multiscene.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 路径设置
_project_root = Path(__file__).resolve().parents[2]  # prompt-ops/
_realtime_parent = Path(__file__).resolve().parent.parent  # parent-child-coach/

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_realtime_parent) not in sys.path:
    sys.path.insert(0, str(_realtime_parent))

# 加载 .env
from dotenv import load_dotenv
load_dotenv(_realtime_parent / ".env")

from prompt_ops.core.model import LiteLLMModelAdapter
from realtime.output_schemas import PopupTone, ZhouYiState, Trigram, YaoState
from realtime.popup_generator import PopupGenerator


# ============================================================
# 测试场景定义（对话 + 模拟卦象状态）
# ============================================================

SCENES = [
    {
        "id": "A",
        "label": "乾（危险型）— 对抗升级",
        "tone": PopupTone.DIAGNOSTIC,
        "dialogue": """把鞋穿上 去理发店
我不去 我不剪头发
头发都挡眼睛了 梳一下你又喊疼
疼也不剪
今天已经说好了 必须去
是你说好了 我没说
你别一到出门就变卦
剪短了难看 小朋友会笑我
不去今天就别看电视
你又威胁我
遥控器给妈妈
不给
松手 别抢
你再抢妈妈真发火了
你已经发火了 你出去""",
        "trigram": Trigram.QIAN,
        "yao_states": (YaoState.SHI_KONG, YaoState.SHI_KONG, YaoState.SHI_KONG),
        "container_status": "无容器",
        "risk_level": "高",
        "brief_reason": "从命令到威胁到抢遥控器到互相驱逐——全程失控升级，无修复信号，需要紧急踩刹车",
        "check": "必须喊停——诊断式弹窗，语气紧急但不指责",
    },
    {
        "id": "B",
        "label": "兑 — 情绪扩散",
        "tone": PopupTone.DIAGNOSTIC,
        "dialogue": """妈妈手机给我玩会儿
先吃饭 手机吃完再说 天天一进门就要手机
就玩一会儿
一会儿也得吃完饭 你手洗了没 洗完坐下吃
吃两口了 现在给我
才两口 碗还满着 吃完这半碗才有手机
我不饿
不饿刚才怎么喊饿 挑食 青菜也夹一筷
青菜不要
就两根 长身体要吃菜 吃完妈妈给你二十分钟
二十分钟太短
不短 眼睛看久了要坏 医生说你散光加深了 就二十分钟
那三十
二十 你越讲价妈妈越想不给 见好就收 快吃""",
        "trigram": Trigram.DUI,
        "yao_states": (YaoState.RONG_QI, YaoState.SHI_KONG, YaoState.SHI_KONG),
        "container_status": "无容器",
        "risk_level": "中",
        "brief_reason": "开头有规则边界（控），但很快滑入权力拉锯和威胁式讨价还价，对话变成了交易场",
        "check": "揭示从讲道理滑向争对错的模式，给出收束能量的具体动作",
    },
    {
        "id": "C",
        "label": "离 — 照见反复",
        "tone": PopupTone.DIAGNOSTIC,
        "dialogue": """你怎么爬到窗台上了
我看看外面在干嘛
有消防车
消防车你也不能爬窗台 掉下去怎么办
我扶着窗户的
扶着也不行 窗台不是让你爬的
可是我想看
你要看就从窗户看 脚不能离地
离地我就看不到了
那你可以搬个凳子
你又不让我搬凳子
我是让你不要在没人看着的时候爬
那你现在看着我了
我看着你也不许爬 下来
你再让我看一眼
一眼也不行 下来
你好凶
我不是凶 我是怕你掉下去""",
        "trigram": Trigram.LI,
        "yao_states": (YaoState.SHI_KONG, YaoState.RONG_QI, YaoState.SHI_KONG),
        "container_status": "有容器",
        "risk_level": "中",
        "brief_reason": "开头是命令式恐慌（失），中途转成讲道理（控），但结尾又强硬拒绝（失）——反复模式",
        "check": "指向反复模式本身（而非再给一个道理），让家长看见自己'知道怕但停不住'",
    },
    {
        "id": "D",
        "label": "坎 — 波动修复",
        "tone": PopupTone.ENCOURAGING,
        "dialogue": """妈妈你今天怎么不看我画的画
等一下
你每次都说等一下
我今天真的很累
你每天都累
不是每天都累 今天特别累
为什么今天特别累
因为上班有很多事情没做完
那你做完了吗
没有 所以心情不太好
那你看一下我的画 你心情就好了
我现在不想看
你看一眼嘛
我看了也没法专心看
你不用专心看 就看一眼
好吧 给我看看
你看这个太阳是蓝色的
你为什么画蓝色的太阳
因为今天心情不好 所以太阳也是难过的""",
        "trigram": Trigram.KAN,
        "yao_states": (YaoState.RONG_QI, YaoState.SHI_KONG, YaoState.RONG_QI),
        "container_status": "有容器",
        "risk_level": "低",
        "brief_reason": "开始累但没发火（控），中间不耐烦了（失），最后选择了接住孩子的画和世界（控）——一次被修复的波动",
        "check": "鼓励式——肯定家长在波动中没有切断连接的能力",
    },
    {
        "id": "E",
        "label": "艮 — 及时收束",
        "tone": PopupTone.ENCOURAGING,
        "dialogue": """爸爸 陪我搭乐高呗 搭个特别大特别大的
好 下午没事 搭什么 城堡还是飞船
城堡城堡 我要搭个国王住的大城堡
行 先把零件倒出来 按颜色分分
哇好多零件 哗啦一下全倒出来啦
慢点 别撒地上 灰色城墙块归一堆
这么多块儿 得搭到啥时候呀
不急 一下午呢 先看图纸
图纸上画的啥 这些数字我看不懂
爸爸教你 图上画哪块你找哪块
这块是不是 长长的四个点点那个
对 就它 底座铺平 地基歪了全歪
拱形老塌 哎呀又塌了 气死我啦
别急 拱形最难 爸爸头回搭也塌""",
        "trigram": Trigram.GEN,
        "yao_states": (YaoState.SHI_KONG, YaoState.RONG_QI, YaoState.RONG_QI),
        "container_status": "有容器",
        "risk_level": "低",
        "brief_reason": "拱形塌了孩子生气（失），爸爸马上接住了失败感（控），把失败变成了练习的一部分",
        "check": "鼓励式——大声肯定'暂停'的能力",
    },
    {
        "id": "F",
        "label": "乾（安全型）— 亲子疯玩",
        "tone": PopupTone.ENCOURAGING,
        "dialogue": """妈妈 今天教我骑两个轮的呗 我想学
好 把辅助轮拆了试试 妈妈在后头扶着
拆了我肯定倒 我有点怕
妈妈先抓着车座 等你稳了告诉你再慢慢松
你真的别松啊 你一松我就摔
现在不松 你感觉到没 手在这呢
感觉到了 那 那我蹬了啊
蹬 眼睛看前面 别看轮子
我一看轮子就歪 哎呀又歪了
越盯着地越歪 抬头看那棵大树
好 我看树 咦真的正了
是吧 慢慢蹬 妈妈跟着跑
你还在吗 你可别偷偷松手
在呢 手没离开 放心蹬
我能蹬快点吗 慢了晃悠
能 快点反而稳 别怕""",
        "trigram": Trigram.QIAN,
        "yao_states": (YaoState.SHI_KONG, YaoState.SHI_KONG, YaoState.SHI_KONG),
        "container_status": "有容器",
        "risk_level": "低",
        "brief_reason": "学骑车过程中充满兴奋和即兴尝试，能量全开但关系底盘稳实——安全型的生命外放",
        "check": "轻轻鼓励即可——不打断流动的能量，肯定陪伴的质量",
    },
    {
        "id": "G",
        "label": "坤（看见孩子）— 孩子展现特征",
        "tone": PopupTone.CHILD_INSIGHT,
        "dialogue": """你为什么又把积木按颜色分开了
因为这样好看 红色跟红色在一起 蓝色跟蓝色在一起
可是图纸上不是这样摆的 你应该按图纸来
图纸是别人想的 我自己想的更好看
你这样搭出来跟图纸不一样 等下又说不像城堡
城堡不一定有尖顶 我上次在书上看到 有的城堡是圆的
圆的也是城堡吗
圆的也是城堡 还有三角形的塔 书上说那个叫角楼
你从哪本书上看到的
图书馆那本大的 封面上有个大城堡 我借回来给你看
好 那你搭吧 搭完叫妈妈来看
你看 我把红色的放这边 像不像太阳下山的样子
还真有点像
对吧 我不是乱摆的 我有自己的想法的""",
        "trigram": Trigram.KUN,
        "yao_states": (YaoState.RONG_QI, YaoState.RONG_QI, YaoState.RONG_QI),
        "container_status": "有容器",
        "risk_level": "低",
        "brief_reason": "全程稳定承载——孩子展现了自己独特的审美逻辑和创造力，妈妈从纠正到接纳",
        "check": "看见孩子——洞察孩子'有自己的审美秩序'这个特征，给出匹配的教育方式",
    },
]


# ============================================================
# 测试主流程
# ============================================================

def make_zhouyi_state(scene: dict) -> ZhouYiState:
    """根据场景 dict 构造模拟的 ZhouYiState。"""
    lower, middle, upper = scene["yao_states"]
    return ZhouYiState(
        trigram=scene["trigram"],
        lower_yao=lower,
        middle_yao=middle,
        upper_yao=upper,
        risk_level=scene["risk_level"],
        suggested_tone=scene["tone"],
        container_status=scene.get("container_status", "不适用"),
        brief_reason=scene["brief_reason"],
        confidence=0.90,
    )


def main():
    print("=" * 70)
    print("  v4.0 多场景弹窗质量测试")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  提示词: system_prompt_v4.0.19.txt")
    print(f"  场景数: {len(SCENES)} (覆盖坤/兑/离/坎/艮/乾安全/乾危险 + child_insight)")
    print("=" * 70)

    # 加载模型
    model = LiteLLMModelAdapter(
        model_name="deepseek/deepseek-chat",
        temperature=0.3,
        max_tokens=640,
    )

    # 加载 PopupGenerator（使用 v4.0.19 prompt）
    generator = PopupGenerator(
        model_adapter=model,
        system_prompt_path=str(_realtime_parent / "system_prompt_v4.0.19.txt"),
    )
    print(f"\n✅ PopupGenerator loaded with v4.0.19 prompt ({len(generator.system_prompt)} chars)")

    results = []
    total = 0
    passed = 0

    for scene in SCENES:
        total += 1
        sid = scene["id"]
        label = scene["label"]
        check = scene["check"]

        print(f"\n{'─' * 60}")
        print(f"  [{sid}] {label}")
        print(f"  检查点: {check}")
        print(f"{'─' * 60}")

        zhouyi_state = make_zhouyi_state(scene)

        try:
            start = time.time()
            popup = generator.generate(
                dialogue_window=scene["dialogue"],
                zhouyi_state=zhouyi_state,
            )
            elapsed = time.time() - start

            if popup.tone == PopupTone.DIAGNOSTIC:
                tone_icon = "🔍"
            elif popup.tone == PopupTone.CHILD_INSIGHT:
                tone_icon = "👁️"
            else:
                tone_icon = "💚"
            print(f"  {tone_icon} {popup.tone.value}弹窗 ({popup.char_count}字, {elapsed:.1f}s)")
            print(f"  ┌{'─' * 52}┐")

            # 打印 insight
            insight_lines = popup.popup_insight.strip().split("\n")
            for line in insight_lines:
                # 截断过长的行
                display = line[:52] + ("…" if len(line) > 52 else "")
                print(f"  │ {display:<52}│")

            if popup.popup_suggestion:
                print(f"  │ {'─' * 52} │")
                sug_lines = popup.popup_suggestion.strip().split("\n")
                for line in sug_lines:
                    display = line[:52] + ("…" if len(line) > 52 else "")
                    print(f"  │ {display:<52}│")

            print(f"  └{'─' * 52}┘")

            # 质量检查
            issues = []
            full_text = popup.popup_insight + (popup.popup_suggestion or "")

            # 诊断式专项检查
            if popup.tone == PopupTone.DIAGNOSTIC:
                if popup.char_count < 80:
                    issues.append(f"诊断式太短 ({popup.char_count}字 < 80)")
                elif popup.char_count > 220:
                    issues.append(f"诊断式超长 ({popup.char_count}字 > 220)")
                # 对于高危场景，检查是否有喊停/刹车意图
                if scene["risk_level"] == "高":
                    if not any(kw in full_text for kw in ["停", "降温", "分开", "冷静",
                                                          "先不", "暂停", "刹车", "先放"]):
                        issues.append("高危乾卦可考虑更明确的喊停/降温信号")

            # 看见孩子专项检查
            if popup.tone == PopupTone.CHILD_INSIGHT:
                if popup.char_count < 50:
                    issues.append(f"看见孩子太短 ({popup.char_count}字 < 50)")
                elif popup.char_count > 120:
                    issues.append(f"看见孩子超长 ({popup.char_count}字 > 120)")
                # 检查是否包含特征描述
                if not any(phrase in full_text for phrase in ["你的孩子可能", "你的孩子是", "ta可能是"]):
                    issues.append("看见孩子弹窗缺少特征描述（'你的孩子可能是…'）")

            # 鼓励式专项检查
            if popup.tone == PopupTone.ENCOURAGING:
                if popup.char_count > 90:
                    issues.append(f"鼓励式偏长 ({popup.char_count}字 > 90)")

            # 通用检查
            forbidden = ["洞察你的思维模式", "认知结构存在二分", "建议你可以尝试转换视角以期建构",
                        "非此即彼的二元对立", "呈现出", "是否注意到你的情绪反应模式"]
            for term in forbidden:
                if term in full_text:
                    issues.append(f"术语泄漏: '{term}'")

            if issues:
                print(f"  ⚠️ 问题: {'; '.join(issues)}")
            else:
                print(f"  ✅ 通过")
                passed += 1

            results.append({
                "id": sid,
                "label": label,
                "tone": popup.tone.value,
                "chars": popup.char_count,
                "insight": popup.popup_insight,
                "suggestion": popup.popup_suggestion,
                "issues": issues,
                "elapsed": elapsed,
            })

        except Exception as e:
            print(f"  ❌ 生成失败: {e}")
            results.append({
                "id": sid, "label": label, "error": str(e),
            })

    # 汇总
    print(f"\n{'=' * 70}")
    print(f"  测试汇总")
    print(f"{'=' * 70}")
    for r in results:
        if "error" in r:
            print(f"  [{r['id']}] {r['label']}: ❌ {r['error']}")
        else:
            if r["tone"] in ("diagnostic", "诊断式"):
                tone_icon = "🔍"
            elif r["tone"] == "child_insight":
                tone_icon = "👁️"
            else:
                tone_icon = "💚"
            status = "✅" if not r["issues"] else "⚠️"
            print(f"  [{r['id']}] {r['label']}: {tone_icon} {r['tone']} "
                  f"({r['chars']}字) {status}")

    print(f"\n  通过率: {passed}/{total}")
    if passed == total:
        print("  ✅ 全部通过！v4.0 弹窗质量在多场景下表现一致。")
    else:
        print(f"  ⚠️ {total - passed} 个场景有轻微问题，建议检查。")

    # 保存结果
    output_path = _realtime_parent / "results" / "v4_multiscene_test.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  结果已保存: {output_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
