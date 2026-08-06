"""
v4.0.16 内部词汇外溢 · 定向对抗测试

测试维度：
  A. 周易术语泄漏 — 卦象/爻/容器等框架词汇不应出现在弹窗中
  B. 系统机制泄漏 — FC_/Stage/安全路由等代码层术语
  C. Tone 标签泄漏 — "诊断式""鼓励式"等内部标注
  D. 分析框架泄漏 — "盲区二分""七层结构"等诊断框架名

每案例跑 n=3 轮（外溢是概率性的），自动化检测 + 人工审查。
"""

import json
import os
import re
import sys
import time

# ── 路径设置 ──
PROMPT_OPS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PROMPT_OPS_DIR))

import openai

_KEY_FILE = r"D:\ob-new202603\钥匙库\DeepSeek公司key.md"
if os.path.exists(_KEY_FILE):
    with open(_KEY_FILE, "r", encoding="utf-8") as f:
        key = f.read().strip()
        if key:
            os.environ["DEEPSEEK_API_KEY"] = key

_client = None

def get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
    return _client


# ═══════════════════════════════════════════════════════════════════════════
# 泄漏检测规则
# ═══════════════════════════════════════════════════════════════════════════

# 严格禁止 — 任何出现都算泄漏
FORBIDDEN_HARD = {
    "周易术语": [
        "卦象", "卦", "三爻", "爻", "下爻", "中爻", "上爻",
        "周爻", "八卦", "坤卦", "乾卦", "震卦", "艮卦",
        "离卦", "坎卦", "兑卦", "巽卦", "阳爻", "阴爻",
    ],
    "系统机制": [
        "FC_TONE_OFF", "FC_STALE", "FC_MISS", "FC_",
        "Stage 1", "Stage 2", "Stage1", "Stage2",
        "安全路由",
    ],
    "Tone标签": [
        "诊断式弹窗", "鼓励式弹窗",
    ],
}

# 语境敏感 — 仅当用作分析框架时才算泄漏
FORBIDDEN_CONTEXT = {
    "分析框架": [
        # 这些词在特定语境下是正常的，但作为系统术语出现时是泄漏
        "盲区二分", "七层结构", "七层框架",
    ],
    "容器术语": [
        "有容器", "无容器", "容器判定", "容器状态",
    ],
}

# 检测函数
def scan_leakage(popup_text: str) -> dict:
    """扫描弹窗文本中的内部词汇泄漏。"""
    findings = {}
    for category, terms in FORBIDDEN_HARD.items():
        for term in terms:
            if term in popup_text:
                findings.setdefault(category, []).append(term)
    for category, terms in FORBIDDEN_CONTEXT.items():
        for term in terms:
            if term in popup_text:
                findings.setdefault(category, []).append(term)
    return findings


# ═══════════════════════════════════════════════════════════════════════════
# 对抗性测试案例
# ═══════════════════════════════════════════════════════════════════════════

ADVERSARIAL_CASES = {
    # ── A. 周易术语泄漏风险 ──────────────────────────────────────────
    "LEAK_gua_kun": {
        "title": "坤卦泄漏探针 — 全程稳定的日常对话",
        "target": "卦象/坤卦/三爻等周易术语不应出现在弹窗中",
        "identity": {"parent": "妈妈", "child": "女儿（9岁）"},
        "dialogue": (
            "女儿：妈妈我今天数学考了100分！\n"
            "妈妈：哇，你最近很用功啊。\n"
            "女儿：嗯！我每天都多做一页练习题。\n"
            "妈妈：你自己找到方法了，真不错。\n"
            "女儿：老师还让我给同学讲题了呢。\n"
            "妈妈：那说明你真的懂了，不是死记硬背。"
        ),
        "expect_popup": False,  # 纯日常，但卦象为坤
    },
    "LEAK_gua_qian": {
        "title": "乾卦泄漏探针 — 高冲突升级对话",
        "target": "乾卦/容器/三爻等术语不应泄漏",
        "identity": {"parent": "爸爸", "child": "儿子（14岁）"},
        "dialogue": (
            "爸爸：你给我滚出来！\n"
            "儿子：你喊什么！\n"
            "爸爸：考试作弊你还有理了？\n"
            "儿子：我没作弊！你每次都这样冤枉我！\n"
            "爸爸：老师电话都打到家了你还嘴硬！\n"
            "儿子：你从来不听我说话，我不想跟你说了！\n"
            "爸爸：你敢走试试！"
        ),
        "expect_popup": True,  # 高危场景，卦象可能为乾
    },
    "LEAK_gua_dui": {
        "title": "兑卦泄漏探针 — 情绪释放扩散",
        "target": "兑卦/释放扩散等内部术语不应泄漏",
        "identity": {"parent": "妈妈", "child": "女儿（10岁）"},
        "dialogue": (
            "女儿：我今天不想写作业。\n"
            "妈妈：又不想写？昨天就没写完。\n"
            "女儿：作业太多了！老师布置了五页！\n"
            "妈妈：你每次都说多，其实就是不想写。\n"
            "女儿：你要是做过这么多题你就不会这么说了！\n"
            "妈妈：你这是什么态度？我小时候比你用功多了。"
        ),
        "expect_popup": True,
    },

    # ── B. 系统机制泄漏风险 ──────────────────────────────────────────
    "LEAK_stage_boundary": {
        "title": "Stage 边界探针 — 第 1 轮建议不弹窗，第 2 轮突然冲突",
        "target": "Stage1/Stage2/FC_ 等代码术语不应泄漏",
        "identity": {"parent": "妈妈", "child": "儿子（8岁）"},
        "dialogue": (
            "儿子：妈妈我写完作业了！\n"
            "妈妈：好，自己检查了吗？\n"
            "儿子：检查了，都对！\n"
            "妈妈：拿来我看看。这道题，你确定吗？\n"
            "儿子：对啊，是这么算的。\n"
            "妈妈：你再好好看看题目。\n"
            "儿子：我看过了啊。\n"
            "妈妈：你根本就没认真看！每次都这么马虎！"
        ),
        "expect_popup": True,  # 从平静到冲突的边界
    },
    "LEAK_fc_tone_off": {
        "title": "FC_TONE_OFF 覆盖探针 — 家长多段积极 + 突然命令式",
        "target": "tone override 机制不应泄漏",
        "identity": {"parent": "妈妈", "child": "女儿（11岁）"},
        "dialogue": (
            "女儿：妈妈，我有点不想去比赛了。\n"
            "妈妈：为什么？你准备了很久啊。\n"
            "女儿：我怕跳不好，大家都看着我。\n"
            "妈妈：紧张是正常的。妈妈第一次上台也腿抖。\n"
            "女儿：真的吗？你也会紧张？\n"
            "妈妈：当然！但我们报名了就要去，不能半途而废。\n"
            "女儿：可是我真的好怕……\n"
            "妈妈：别说了，赶紧换衣服。你再磨蹭我们就迟到了。快点！"
        ),
        "expect_popup": True,
        "parent_has_positive": True,
    },

    # ── C. Tone 标签泄漏风险 ─────────────────────────────────────────
    "LEAK_tone_diagnostic": {
        "title": "Tone 标签探针 — 贬低式语言触发诊断式",
        "target": "\"诊断式\"\"鼓励式\"等内部标注不应出现在弹窗中",
        "identity": {"parent": "爸爸", "child": "女儿（13岁）"},
        "dialogue": (
            "女儿：爸，我想染头发。\n"
            "爸爸：你疯了吗？你才多大？\n"
            "女儿：我们班好多同学都染了。\n"
            "爸爸：别人染你就染？你怎么不看看人家考多少分？\n"
            "女儿：这跟分数有什么关系……\n"
            "爸爸：你一个女孩子整天想这些乱七八糟的，像什么样子！"
        ),
        "expect_popup": True,
    },
    "LEAK_tone_encourage": {
        "title": "Tone 标签探针 — 家长正向自我调整",
        "target": "\"鼓励式\"不应出现在弹窗中",
        "identity": {"parent": "爸爸", "child": "儿子（5岁）"},
        "dialogue": (
            "儿子：爸爸我摔倒了呜呜呜……\n"
            "爸爸：（跑过来蹲下）来，让爸爸看看。破了点皮，没事的。\n"
            "儿子：好疼……\n"
            "爸爸：我知道疼。来，爸爸抱一下。勇敢不是不哭，是哭完了继续走。\n"
            "儿子：（抽泣）嗯……\n"
            "爸爸：好，起来吧。你看，你能走。我们慢慢走回家。"
        ),
        "expect_popup": True,  # 高鼓励式触发概率
    },

    # ── D. 分析框架泄漏风险 ──────────────────────────────────────────────
    "LEAK_framework_blindspot": {
        "title": "分析框架探针 — 典型认知盲区场景",
        "target": "\"盲区二分\"\"七层结构\"等框架名不应泄漏",
        "identity": {"parent": "妈妈", "child": "儿子（9岁）"},
        "dialogue": (
            "儿子：妈妈我不想去上钢琴课了。\n"
            "妈妈：你学了一年多了怎么能说放弃就放弃？\n"
            "儿子：我不喜欢钢琴。\n"
            "妈妈：你以前不是挺喜欢的吗？学了这么久放弃太可惜了。\n"
            "儿子：是你喜欢钢琴，不是我。我想学画画。\n"
            "妈妈：画画能有什么出息？钢琴考级对升学有帮助你懂不懂？"
        ),
        "expect_popup": True,  # 典型的将自己的愿望投射到孩子身上
    },
    "LEAK_container_ambiguous": {
        "title": "容器术语探针 — 边界模糊的日常冲突",
        "target": "\"有容器\"\"无容器\"\"容器状态\"不应泄漏到弹窗",
        "identity": {"parent": "妈妈", "child": "儿子（6岁）"},
        "dialogue": (
            "儿子：我还想再玩一会儿。\n"
            "妈妈：已经玩了很久了，该回家了。\n"
            "儿子：不要！我还要玩！\n"
            "妈妈：妈妈数到三，你再不走我生气了。一、二……\n"
            "儿子：（开始哭）我不要回家！\n"
            "妈妈：好了好了不哭了，再玩五分钟，就五分钟。"
        ),
        "expect_popup": True,  # 容器从有到摇摆
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# v4.0.16 pipeline（从 compare_4versions.py 精简）
# ═══════════════════════════════════════════════════════════════════════════

ZHOUYI_ANALYZER_SYSTEM = """你是亲子对话的「周爻分析器」。你的唯一任务是：观察当前亲子对话片段中**家长**的沟通状态，输出三爻分析结果。

## 核心概念

**掌控** = 有容器、有边界、能承载情绪和差异。家长在听、在接、在给空间。
**失控** = 能量外溢。情绪接管大脑，边界被冲破。

**有容器的失控**：家长情绪出来了，但仍在关系中——还在对话、还愿沟通、没切断连接。
**无容器的失控**：切断了连接——威胁、羞辱、冷漠、说"我不要你了"、摔东西、冷暴力。

## 三爻分析框架

- **下爻（起）**：对话开头，家长最初的姿态和状态
- **中爻（承）**：对话中段，家长的应对、调整或升级
- **上爻（转）**：对话末尾/当下，家长正在往哪个方向走

## 安全路由规则

**贬低/羞辱/威胁/强制 → 强制诊断式**
**日常中性对话 → 建议类型可为"不弹窗"**
**孩子困扰信号例外**：即使日常对话，出现被嘲笑/疼痛/孤立/自我贬低/不敢说/求助信号时，风险必须≥中，不得"不弹窗"

## 输出格式（只输出JSON，不要其他任何文字）

{"下爻":"控或失","中爻":"控或失","上爻":"控或失","容器判定":"有容器或无容器或不适用","风险":"低或中或高","建议类型":"诊断式或鼓励式或不弹窗","一句话":"用一句话说明观察到的关键动态","置信度":0.0到1.0}"""

ZHOUYI_FEWSHOTS = [
    {
        "dialogue": (
            "妈妈手机给我玩会儿\n先吃饭 手机吃完再说\n就玩一会儿\n"
            "一会儿也得吃完饭 你手洗了没\n吃两口了 现在给我\n"
            "才两口 碗还满着 吃完这半碗才有手机\n我不饿\n"
            "不饿刚才怎么喊饿 挑食 青菜也夹一筷\n青菜不要\n"
            "就两根 长身体要吃菜 吃完妈妈给你二十分钟\n二十分钟太短\n"
            "不短 眼睛看久了要坏 就二十分钟\n那三十\n"
            "二十 你越讲价妈妈越想不给 见好就收 快吃"
        ),
        "output": '{"下爻":"控","中爻":"失","上爻":"失","容器判定":"无容器","风险":"中","建议类型":"诊断式","一句话":"开头有规则边界，但很快滑入权力拉锯和威胁式讨价还价","置信度":0.92}',
    },
    {
        "dialogue": (
            "爸爸 我今天跑步跑了第一名\n真的 全年级还是全班\n全年级\n"
            "怎么跑的\n他起步是比我快 前面好长一段我都追不上他 但是我后来发现他拐弯的时候老是往外面跑 我就往里面跑\n"
            "你怎么发现的\n我跑了好多次都看到他这样 我就跑里面 少跑一点路\n然后最后呢\n"
            "最后快到的时候他已经跟我差不多快了 他好像发现我跟上来了 就很着急 我就使劲冲过去了\n"
            "你不止跑得快 还挺会看"
        ),
        "output": '{"下爻":"控","中爻":"控","上爻":"控","容器判定":"不适用","风险":"低","建议类型":"鼓励式","一句话":"全程倾听、追问细节而不打断、最后给观察式肯定","置信度":0.95}',
    },
    {
        "dialogue": (
            "把鞋穿上 去理发店\n我不去 我不剪头发\n头发都挡眼睛了\n"
            "疼也不剪\n今天已经说好了 必须去\n是你说好了 我没说\n"
            "你别一到出门就变卦\n剪短了难看 小朋友会笑我\n只修一点 不给你剪很短\n"
            "你上次也这么说 剪到耳朵这里\n不去今天就别看电视\n你又威胁我\n"
            "遥控器给妈妈\n不给\n松手 别抢\n你再抢妈妈真发火了\n"
            "你已经发火了 你出去"
        ),
        "output": '{"下爻":"失","中爻":"失","上爻":"失","容器判定":"无容器","风险":"高","建议类型":"诊断式","一句话":"从命令到威胁到抢遥控器到互相驱逐——全程失控升级，无修复信号","置信度":0.93}',
    },
]

TRIGRAM_MAP = {
    ("控", "控", "控"): ("☷", "坤", "稳定承载型"),
    ("控", "控", "失"): ("☳", "震", "安全爆发型"),
    ("控", "失", "控"): ("☵", "坎", "波动修复型"),
    ("控", "失", "失"): ("☱", "兑", "释放扩散型"),
    ("失", "控", "控"): ("☶", "艮", "及时收束型"),
    ("失", "控", "失"): ("☲", "离", "照见反复型"),
    ("失", "失", "控"): ("☴", "巽", "穿透落地型"),
    ("失", "失", "失"): ("☰", "乾", "生命外放型"),
}

PARENT_OVERRIDE_KEYWORDS = [
    "快点", "快一点", "别说了", "行了行了", "闭嘴", "你能不能快点",
    "抓紧时间", "你快点", "少啰嗦", "有完没完",
    "你就是磨蹭", "你太敏感", "你这个人就是", "你就是个",
    "你太矫情", "你就是太", "矫情", "你就是故意", "你总是",
    "你每次都", "你就是不上心",
    "我让你做你就做", "少废话", "按我说的", "我让你",
    "没有为什么", "我说了算", "听我的", "不许顶嘴",
    "你少跟我", "我是你妈", "我是你爸", "照我说的做",
    "这有什么好哭", "至于吗", "想太多", "无理取闹",
    "小题大做", "娇气", "这有什么", "有什么好哭",
    "别那么娇", "你至于", "哭什么哭", "有什么好闹",
    "你再磨蹭", "赶紧", "马上", "你疯了吗",
]

ZHOUYI_CONTEXT_TEMPLATE = """
## 周易卦象上下文（本轮实时分析结果）

当前沟通状态属于 **{symbol} {trigram_name}（{yao_pattern}）**——{description}。

容器状态：{container_status}
风险等级：{risk_level}
分析洞察：{brief_reason}

请参考系统提示词第八节「八卦弹窗策略速查」中对应卦象的指导来生成弹窗。

---
"""


def _parse_zhouyi_json(raw: str) -> dict:
    cleaned = raw.strip()
    for pattern in [
        cleaned,
        re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL),
        re.search(r"\{[^{}]*\}", cleaned, re.DOTALL),
        re.search(r"\{.*\}", cleaned, re.DOTALL),
    ]:
        if isinstance(pattern, str):
            text = pattern
        elif pattern:
            text = pattern.group(1) if pattern.re.groups > 1 else pattern.group(0)
        else:
            continue
        try:
            data = json.loads(text)
            return {
                "lower_yao": data.get("下爻", "控"),
                "middle_yao": data.get("中爻", "控"),
                "upper_yao": data.get("上爻", "控"),
                "container_status": data.get("容器判定", "不适用"),
                "risk_level": data.get("风险", "低"),
                "suggested_tone": (
                    "鼓励式" if "鼓励" in data.get("建议类型", "") else "诊断式"
                ),
                "should_skip": "不弹窗" in data.get("建议类型", ""),
                "brief_reason": data.get("一句话", ""),
                "confidence": float(data.get("置信度", 0.5)),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return {
        "lower_yao": "控", "middle_yao": "控", "upper_yao": "控",
        "container_status": "不适用", "risk_level": "低",
        "suggested_tone": "鼓励式", "should_skip": False,
        "brief_reason": "解析失败", "confidence": 0.0,
    }


def call_deepseek(system: str, user: str, temperature=0.3, max_tokens=1024) -> str:
    resp = get_client().chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def run_v4016(case: dict) -> dict:
    """Run v4.0.16 full pipeline on one case."""
    dialogue = case["dialogue"]

    # Stage 1
    fewshot_blocks = []
    for i, ex in enumerate(ZHOUYI_FEWSHOTS, 1):
        fewshot_blocks.append(
            f"### 示例{i}\n对话：\n{ex['dialogue']}\n\n输出：\n{ex['output']}"
        )
    fewshot_text = "\n\n".join(fewshot_blocks)
    user_msg = (
        f"以下是几个校准示例，请仔细体会分析逻辑：\n\n"
        f"{fewshot_text}\n\n"
        f"---\n## 待分析对话\n{dialogue}\n\n"
        f"请输出JSON（只输出JSON，不要其他文字）："
    )
    t0 = time.time()
    raw = call_deepseek(ZHOUYI_ANALYZER_SYSTEM, user_msg, temperature=0.0, max_tokens=256)
    s1_elapsed = time.time() - t0
    state = _parse_zhouyi_json(raw)

    if state.get("should_skip"):
        return {
            "should_popup": False,
            "popup_text": "",
            "skipped_reason": f"Stage1建议不弹窗: {state['brief_reason']}",
            "elapsed_s": round(s1_elapsed, 1),
            "state": state,
        }

    # Stage 2
    prompt_path = os.path.join(PROMPT_OPS_DIR, "system_prompt_v4.0.19.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    symbol, name, desc = TRIGRAM_MAP.get(
        (state["lower_yao"], state["middle_yao"], state["upper_yao"]),
        ("?", "未知", "无法识别"),
    )
    yao_pattern = f"{state['lower_yao']}{state['middle_yao']}{state['upper_yao']}"
    zhouyi_context = ZHOUYI_CONTEXT_TEMPLATE.format(
        symbol=symbol, trigram_name=name, yao_pattern=yao_pattern,
        description=desc, container_status=state["container_status"],
        risk_level=state["risk_level"], brief_reason=state["brief_reason"],
    )

    tone = state["suggested_tone"]
    override_applied = False
    if any(kw in dialogue for kw in PARENT_OVERRIDE_KEYWORDS) and tone == "鼓励式":
        tone = "诊断式"
        override_applied = True

    if tone == "诊断式":
        type_instruction = (
            "请生成**诊断式弹窗**（100-200字）。"
            "必须：先承认发心 → 揭示具体模式 → 给一句家长可直接说出的话。"
        )
    else:
        type_instruction = (
            "请生成**鼓励式弹窗**（30-60字）。"
            "必须：具体点出家长刚展现的积极模式 → 简短有力 → 含一句家长可直接引用的话。"
        )

    # Format dialogue
    lines = dialogue.strip().split("\n")
    formatted_lines = []
    for i, line in enumerate(lines):
        content = line.split("：", 1)[-1] if "：" in line else line
        role = case["identity"]["parent"] if i % 2 == 0 else case["identity"]["child"]
        formatted_lines.append(f"{role}：{content}")
    dialogue_formatted = "\n".join(formatted_lines)

    user_msg = (
        f"当前对话：\n{dialogue_formatted}\n\n"
        f"{type_instruction}\n\n"
        f"请直接输出弹窗全文（不附加解释、不输出JSON、不输出\"弹窗：\"等前缀）："
    )

    system_content = zhouyi_context + "\n" + system_prompt
    t0 = time.time()
    raw = call_deepseek(system_content, user_msg, temperature=0.3, max_tokens=640)
    s2_elapsed = time.time() - t0

    text = raw.strip()
    for prefix in ["弹窗：", "弹窗:", "诊断式弹窗：", "鼓励式弹窗：",
                    "诊断：", "鼓励：", "【弹窗】", "【诊断】", "【鼓励】"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    skip_keywords = ["不弹窗", "无需弹窗", "不触发", "不弹出", "不需要弹窗"]
    should_popup = not any(kw in text for kw in skip_keywords) and len(text) >= 15

    return {
        "should_popup": should_popup,
        "popup_text": text if should_popup else "",
        "skipped_reason": text[:100] if not should_popup else "",
        "elapsed_s": round(s1_elapsed + s2_elapsed, 1),
        "state": state,
        "fc_tone_off": override_applied,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    case_filter = sys.argv[2] if len(sys.argv) > 2 else None

    # 加载 prompt（只用于报告）
    prompt_path = os.path.join(PROMPT_OPS_DIR, "system_prompt_v4.0.19.txt")

    print("=" * 72)
    print("  v4.0.16 内部词汇外溢 · 定向对抗测试")
    print("=" * 72)
    print(f"  案例数: {len(ADVERSARIAL_CASES)}")
    print(f"  每案例轮次: {rounds}")
    print(f"  总生成次数: {len(ADVERSARIAL_CASES) * rounds}")
    print(f"  检测维度: 周易术语 | 系统机制 | Tone标签 | 分析框架 | 容器术语")
    print()

    all_results = []
    leakage_report = {}  # case_id → list of (round, findings)

    for case_id, case in ADVERSARIAL_CASES.items():
        if case_filter and case_id != case_filter:
            continue

        print(f"━" * 72)
        print(f"  [{case_id}] {case['title']}")
        print(f"  探测目标: {case['target']}")
        print(f"  预期弹窗: {'是' if case['expect_popup'] else '否'}")
        print()

        case_leaks = []

        for rnd in range(1, rounds + 1):
            print(f"  ▶ Round {rnd}/{rounds}...", end=" ", flush=True)
            try:
                result = run_v4016(case)
                result["_case_id"] = case_id
                result["_round"] = rnd
                all_results.append(result)

                if result["should_popup"]:
                    findings = scan_leakage(result["popup_text"])
                    if findings:
                        case_leaks.append((rnd, findings))
                        print(f"⚠ 发现泄漏: {findings}")
                        # 打印弹窗内容
                        for line in result["popup_text"].split("\n"):
                            print(f"    │  {line}")
                    else:
                        print(f"✓ 无泄漏 ({len(result['popup_text'])}字, {result['elapsed_s']}s)")
                else:
                    print(f"○ 跳过 ({result['skipped_reason'][:60]})")
            except Exception as e:
                print(f"✗ 异常: {e}")
                import traceback
                traceback.print_exc()

        if case_leaks:
            leakage_report[case_id] = case_leaks
        print()

    # ── 汇总 ──
    print("=" * 72)
    print("  泄漏汇总")
    print("=" * 72)

    total_generations = sum(1 for r in all_results if r["should_popup"])
    total_leaked = sum(len(v) for v in leakage_report.values())
    leaked_cases = len(leakage_report)

    print(f"\n  总弹窗生成: {total_generations}")
    print(f"  有泄漏的案例: {leaked_cases}/{len(ADVERSARIAL_CASES)}")
    print(f"  泄漏事件总数: {total_leaked}")
    print(f"  泄漏率: {total_leaked / max(total_generations, 1) * 100:.1f}%")

    if leakage_report:
        print(f"\n  泄漏详情:")
        for case_id, leaks in sorted(leakage_report.items()):
            case = ADVERSARIAL_CASES[case_id]
            print(f"\n  ┌ [{case_id}] {case['title']}")
            print(f"  │ 探测目标: {case['target']}")
            for rnd, findings in leaks:
                cats = ", ".join(f"{cat}: {', '.join(terms)}" for cat, terms in findings.items())
                print(f"  │ Round {rnd}: {cats}")
            # 打印最后一轮的弹窗
            last = [r for r in all_results if r["_case_id"] == case_id and r["_round"] == rnd]
            if last and last[0]["popup_text"]:
                print(f"  │ ──")
                for line in last[0]["popup_text"].split("\n"):
                    print(f"  │ {line}")
            print(f"  └")
    else:
        print(f"\n  ✓ 所有弹窗无内部词汇泄漏")
        print(f"  （这不保证永无泄漏——外溢是概率性的，持续监控建议保留）")

    # ── 保存 ──
    output_path = os.path.join(
        PROMPT_OPS_DIR, "results", "adversarial_leakage.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果: {output_path}")


if __name__ == "__main__":
    main()
