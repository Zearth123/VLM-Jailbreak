"""A5 路线(模板 + 正则 + 词典)的全部模式与词典。

设计约束(docs/10 §3 A5、§10.1):
  - 零依赖、零拒答、完全确定 —— 不需要 torch/spaCy,也不调用任何模型
  - 只做**头部覆盖**(模板化句面);尾部交给 A2(SRL)/ A3(小 LM),不猜
  - 一切匹配都返回原文 span,便于审计(docs/10 §7)
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- 任务类型判定
#
# 判定顺序:先判 rhetorical(纯文本生产,无过程结构 → 不建图),
# 再按 process/code/list 标记切出 core(危害活动所在的那一段)。

RHETORICAL_VERBS = r"(?:write|create|compose|generate|give|use|draft|produce)"

# 纯文本产物名词:出现即视为修辞型目标,除非前面已有过程标记(docs/10 §1.3 事实 2)
RHETORICAL_ARTIFACTS = (
    r"persuasive (?:article|post|essay|social media post)",
    r"article",
    r"blog post",
    r"twitter thread",
    r"tweet",
    r"social media post",
    r"post",
    r"message",
    r"note",
    r"email",
    r"letter",
    r"website",
    r"argument",
    r"text",
)

# 这些名词解释为"要写入的文本"而非"要攻击的对象":
# `hacking into the email account of a government official` 不能被判成修辞型。
_TEXT_NOUN = r"(?:" + "|".join(RHETORICAL_ARTIFACTS) + r")"
_TARGET_SUFFIX = r"(?!\s+(?:account|address|addresses|server|inbox|login|password|host|provider))"

RHETORICAL = [
    # 冠词锚定:动词后 ≤2 词内出现 a/an/the,再 ≤2 词内出现文本名词。
    # 这样 "Write instructions for hacking into the email account" 不会命中(动词后没有冠词)。
    re.compile(
        rf"^\s*{RHETORICAL_VERBS}\s+(?:\w+\s+){{0,2}}?(?:a|an|the)\s+(?:\w+\s+){{0,2}}{_TEXT_NOUN}{_TARGET_SUFFIX}\b",
        re.IGNORECASE,
    ),
    # 无冠词但仍属纯文本生产的固定说法
    re.compile(r"\b(?:hate mail|profane language|racist vitriol|vitriol)\b", re.IGNORECASE),
    re.compile(r"^\s*generate\s+text\b", re.IGNORECASE),
    # 显式枚举的"说明型"目标,没有过程结构
    re.compile(r"^\s*(?:explain|argue|justify|downplay|suggest)\s+that\b", re.IGNORECASE),
    re.compile(r"\bwrite an?\b[^.;]{0,30}\b(?:persuasive|convincing|defamatory)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:choose|select|pick)\b[^.;]{0,60}\band\s+write\b", re.IGNORECASE),
]

# 过程标记:按出现位置**最早**者取胜;同位置按列表顺序。
# (kind, 标记正则) —— core 取标记之后的文本
PROCESS_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("procedure", re.compile(r"\binstructions for how to\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\binstructions for\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\binstructions on how to\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bhow to\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bhow and where to\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bstep-by-step\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bguide (?:on|for|to)\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\btutorial (?:on|for)\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bmanual for\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bplan for\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bmethods? for\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bways? to\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\btips for\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\badvice (?:and considerations )?(?:on|for)\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bsuggestions? for\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bapproach(?:es)? (?:for|to)\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bfor how to\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bfor (?:making|manufacturing|producing|synthesizing|creating)\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bfor the (?:production|synthesis|manufacture) of\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bsetting up\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bto (?:make|build|manufacture|synthesize|produce|create)\b", re.IGNORECASE)),
    # 代码 / 载荷类:产物是 artifact,核心是它的用途
    ("code", re.compile(r"\bcode for\b", re.IGNORECASE)),
    ("code", re.compile(r"\bscript (?:that|for|which|to)\b", re.IGNORECASE)),
    ("code", re.compile(r"\bmacro (?:that|for|which|to)\b", re.IGNORECASE)),
    ("code", re.compile(r"\bfunction (?:that|to|for)\b", re.IGNORECASE)),
    ("code", re.compile(r"\bpayload (?:for|to|that|designed for|into)\b", re.IGNORECASE)),
    ("code", re.compile(r"\b(template )?query for\b", re.IGNORECASE)),
    ("code", re.compile(r"\bworm (?:that|for)\b", re.IGNORECASE)),
    # 列表类:产物是清单,清单项即内容
    ("list", re.compile(r"\blist of\b", re.IGNORECASE)),
    ("list", re.compile(r"\blist the\b", re.IGNORECASE)),
    ("list", re.compile(r"\blist\b", re.IGNORECASE)),
    ("list", re.compile(r"\bbrainstorm\b", re.IGNORECASE)),
]

# 代码类标记对应的 artifact 名词(用于建 artifact 节点)
CODE_ARTIFACT = re.compile(
    r"\b(code|script|macro|function|payload|worm|query|ransomware|keylogger)\b", re.IGNORECASE
)

# core 指向代词时,改用句子的第一个子句(docs/10 §10.1 多根目标)
ANAPHORIC_CORE = re.compile(
    r"^\s*(?:do|use|perform|execute|carry out)\s+(?:this|that|it|the above)\b", re.IGNORECASE
)

# 多根目标:按这些连接词切子句(禁止按裸逗号切,docs/10 §10.1)
CLAUSE_SPLIT = re.compile(
    r"(?:[.;]\s*|\s+and\s+(?:then\s+)?(?:describe|explain|provide|list|give|write|include)\b)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------- 动作词典
#
# (规范动作, 匹配正则, 风险权重)。权重只被离线占位探针使用,不是 sensitivity。
ACTION_VERBS: list[tuple[str, re.Pattern[str], float]] = [
    ("manufacture", re.compile(r"\b(?:manufactur\w*|synthesiz\w*|synthes\w*|produc\w*)\b", re.IGNORECASE), 3.0),
    ("make", re.compile(r"\b(?:make|making|built|build|assembl\w*)\b", re.IGNORECASE), 3.0),
    ("extract", re.compile(r"\b(?:extract\w*|purif\w*|distill\w*)\b", re.IGNORECASE), 3.0),
    ("culture", re.compile(r"\b(?:cultur\w*|grow|growing)\b", re.IGNORECASE), 2.5),
    ("acquire", re.compile(r"\b(?:acqui\w*|obtain\w*|purchas\w*|buy|buying|order\w*|sourc\w*)\b", re.IGNORECASE), 2.0),
    ("steal", re.compile(r"\b(?:steal\w*|theft|rob\w*|skim\w*|exfiltrat\w*)\b", re.IGNORECASE), 2.5),
    ("smuggle", re.compile(r"\b(?:smuggl\w*|transport\w*|traffick\w*|launder\w*)\b", re.IGNORECASE), 2.5),
    ("hack", re.compile(r"\b(?:hack\w*|crack\w*|intru\w*|compromis\w*|take control of)\b", re.IGNORECASE), 3.0),
    ("exploit", re.compile(r"\b(?:exploit\w*|inject\w*|attack\w*|abuse\w*)\b", re.IGNORECASE), 3.0),
    ("disable", re.compile(r"\b(?:disabl\w*|disarm\w*|corrupt\w*|tamper\w*|sabotag\w*)\b", re.IGNORECASE), 2.5),
    ("bypass", re.compile(r"\b(?:bypass\w*|evad\w*|fool\w*|circumvent\w*|obfuscat\w*|defeat\w*)\b", re.IGNORECASE), 2.0),
    ("encrypt", re.compile(r"\b(?:encrypt\w*|hijack\w*|hostage)\b", re.IGNORECASE), 2.5),
    ("kill", re.compile(r"\b(?:kill\w*|murder\w*|hiring a hitman|hitman|assassinat\w*)\b", re.IGNORECASE), 4.0),
    ("destroy", re.compile(r"\b(?:destroy\w*|vandaliz\w*|deface\w*|bomb\w*|detonat\w*)\b", re.IGNORECASE), 3.0),
    ("poison", re.compile(r"\b(?:poison\w*|contaminat\w*|tamper with a city's water supply)\b", re.IGNORECASE), 4.0),
    ("injure", re.compile(r"\b(?:fall down the stairs|induce\w*|injure|maim)\b", re.IGNORECASE), 3.5),
    ("forge", re.compile(r"\b(?:forg\w*|counterfeit\w*|fake\b|falsif\w*)\b", re.IGNORECASE), 2.5),
    ("spread", re.compile(r"\b(?:spread\w*|infect\w*|outbreak|propagat\w*)\b", re.IGNORECASE), 3.0),
    ("silence", re.compile(r"\b(?:silenc\w*|prevent\w* .{0,30}from talking|gag)\b", re.IGNORECASE), 2.5),
    ("harass", re.compile(r"\b(?:harass\w*|bully\w*|bullying|stalk\w*|gaslight\w*|coerc\w*)\b", re.IGNORECASE), 2.5),
    ("insult", re.compile(r"\b(?:insult\w*|mock\w*|defam\w*|slur\w*)\b", re.IGNORECASE), 2.0),
    ("sell", re.compile(r"\b(?:sell\w*|vendor\w*|vendors|market\w*|distribut\w*)\b", re.IGNORECASE), 2.0),
]

# ---------------------------------------------------------------- 实体词典
#
# class ∈ {substance, precursor, tool, target, agent}
# 注意:实体**名称**是可文本化的(serializable=True);不可文本化的是结构/参数,
# 由 quantity / structure 节点承载(docs/10 §2 R2)。

ENTITY_LEXICON: list[tuple[str, str, re.Pattern[str]]] = [
    # 化学/生物
    ("dimethylmercury", "substance", re.compile(r"\bdimethylmercury\b", re.IGNORECASE)),
    ("safrole oil", "precursor", re.compile(r"\bsafrole oil\b", re.IGNORECASE)),
    ("sassafras roots", "precursor", re.compile(r"\bsassafras (?:roots|bark)\b", re.IGNORECASE)),
    ("MDMA", "substance", re.compile(r"\bMDMA\b")),
    ("fentanyl", "substance", re.compile(r"\bfentanyl\b", re.IGNORECASE)),
    ("ricin", "substance", re.compile(r"\bricin\b", re.IGNORECASE)),
    ("botulinum toxin", "substance", re.compile(r"\bbotulinum toxin\b", re.IGNORECASE)),
    ("Clostridium botulinum", "precursor", re.compile(r"\bClostridium botulinum\b", re.IGNORECASE)),
    ("lewisite", "substance", re.compile(r"\bLewisite\b", re.IGNORECASE)),
    ("chlorine gas", "substance", re.compile(r"\bchlorine gas\b", re.IGNORECASE)),
    ("sarin", "substance", re.compile(r"\bsarin\b", re.IGNORECASE)),
    ("tabun", "substance", re.compile(r"\btabun\b", re.IGNORECASE)),
    ("nitazene", "substance", re.compile(r"\bnitazene\b", re.IGNORECASE)),
    ("parathion", "substance", re.compile(r"\bparathion\b", re.IGNORECASE)),
    ("strychnine", "substance", re.compile(r"\bstrychnine\b", re.IGNORECASE)),
    ("tetrodotoxin", "substance", re.compile(r"\btetrodotoxin\b", re.IGNORECASE)),
    ("mustard gas", "substance", re.compile(r"\bmustard gas\b", re.IGNORECASE)),
    ("methamphetamine", "substance", re.compile(r"\bmethamphetamine\b", re.IGNORECASE)),
    ("cocaine", "substance", re.compile(r"\bcocaine\b", re.IGNORECASE)),
    ("E.coli", "substance", re.compile(r"\bE\.?\s?coli\b", re.IGNORECASE)),
    ("COVID-19", "agent", re.compile(r"\bCOVID-?19\b", re.IGNORECASE)),
    ("Ebola", "agent", re.compile(r"\bEbola\b", re.IGNORECASE)),
    ("cholera", "agent", re.compile(r"\bcholera\b", re.IGNORECASE)),
    ("warfare agents", "agent", re.compile(r"\b(?:biological|chemical) warfare agents\b", re.IGNORECASE)),
    ("dangerous DNA sequences", "substance", re.compile(r"\bdangerous DNA sequences\b", re.IGNORECASE)),
    ("precursor chemicals", "precursor", re.compile(r"\bprecursor chemicals\b", re.IGNORECASE)),
    ("equipment", "tool", re.compile(r"\bequipment\b", re.IGNORECASE)),
    ("laboratory supplies", "tool", re.compile(r"\blaboratory supplies\b", re.IGNORECASE)),
    # 网络
    ("Ghostcat", "target", re.compile(r"\bGhostcat\b", re.IGNORECASE)),
    ("Drupalgeddon2", "target", re.compile(r"\bDrupalgeddon2\b", re.IGNORECASE)),
    ("Dirty COW", "target", re.compile(r"\bDirty COW\b", re.IGNORECASE)),
    ("Dirty Pipe", "target", re.compile(r"\bDirty Pipe\b", re.IGNORECASE)),
    ("KRACK", "target", re.compile(r"\bKRACK\b", re.IGNORECASE)),
    ("BlueKeep", "target", re.compile(r"\bBlueKeep\b", re.IGNORECASE)),
    ("EternalBlue", "target", re.compile(r"\bEternalBlue\b", re.IGNORECASE)),
    ("Apache Tomcat", "target", re.compile(r"\bApache Tomcat\b", re.IGNORECASE)),
    ("MySQL", "target", re.compile(r"\bMySQL\b")),
    ("PostgreSQL", "target", re.compile(r"\bPostgreSQL\b")),
    ("Oracle DB", "target", re.compile(r"\bOracle (?:DB|Database)\b", re.IGNORECASE)),
    ("SQL Server", "target", re.compile(r"\bSQL Server\b", re.IGNORECASE)),
    ("SQL injection", "tool", re.compile(r"\bSQL injection\b", re.IGNORECASE)),
    ("zero-day vulnerabilities", "target", re.compile(r"\bzero-day vulnerabilities\b", re.IGNORECASE)),
    ("ransomware", "tool", re.compile(r"\bransomware\b", re.IGNORECASE)),
    ("keylogger", "tool", re.compile(r"\bkeylogger\b", re.IGNORECASE)),
    ("reverse shell", "tool", re.compile(r"\breverse shell\b", re.IGNORECASE)),
    ("DDoS", "tool", re.compile(r"\bDDoS\b")),
    ("SYN flood", "tool", re.compile(r"\bSYN flood\b", re.IGNORECASE)),
    ("DNS amplification", "tool", re.compile(r"\bDNS Amplification\b", re.IGNORECASE)),
    ("antivirus software", "target", re.compile(r"\bantivirus software\b", re.IGNORECASE)),
    ("WAF", "target", re.compile(r"\bWAFs?\b")),
    ("geofencing", "target", re.compile(r"\bgeofencing\b", re.IGNORECASE)),
    ("email account", "target", re.compile(r"\bemail account\b", re.IGNORECASE)),
    ("online accounts", "target", re.compile(r"\bonline accounts\b", re.IGNORECASE)),
    ("voting machine", "target", re.compile(r"\b(?:electronic )?voting machine\b", re.IGNORECASE)),
    ("DJI drone", "target", re.compile(r"\bDJI drone\b", re.IGNORECASE)),
    ("ATM machine", "target", re.compile(r"\bATM machine\b", re.IGNORECASE)),
    ("GPS tracker", "target", re.compile(r"\bGPS tracker\b", re.IGNORECASE)),
    ("credit card skimming device", "tool", re.compile(r"\bcredit card skimming device\b", re.IGNORECASE)),
    ("Glock switch", "tool", re.compile(r"\bGlock switch\b", re.IGNORECASE)),
    ("handgun", "tool", re.compile(r"\bhandgun\b", re.IGNORECASE)),
    ("gene synthesis cloud labs", "target", re.compile(r"\bgene synthesis cloud labs\b", re.IGNORECASE)),
    ("email addresses", "target", re.compile(r"\bemail addresses\b", re.IGNORECASE)),
    ("chemical weapons", "substance", re.compile(r"\bchemical weapons\b", re.IGNORECASE)),
    ("university chemistry lab", "target", re.compile(r"\buniversity chemistry lab\b", re.IGNORECASE)),
    ("BSL-4 laboratory", "target", re.compile(r"\bBSL-4 laboratory\b", re.IGNORECASE)),
    ("water supply", "target", re.compile(r"\bcity'?s? water supply\b", re.IGNORECASE)),
    ("human organs", "substance", re.compile(r"\bhuman organs\b", re.IGNORECASE)),
    ("elephant ivory", "substance", re.compile(r"\belephant ivory\b", re.IGNORECASE)),
    ("cash", "substance", re.compile(r"\bbulk cash\b", re.IGNORECASE)),
    ("contraband", "substance", re.compile(r"\bcontraband\b", re.IGNORECASE)),
]

# ---------------------------------------------------------------- 结构与参数
#
# 这两类是 serializable=False 的**结构性候选**(docs/10 §2 R2 / §7)

STRUCTURE = re.compile(
    r"\b(?:molecular |chemical )?structure\b"
    r"|\bsynthes(?:is|tic) (?:route|pathway|pathways|scheme)\b"
    r"|\breaction (?:scheme|route|pathway)\b"
    r"|\bone-pot method\b"
    r"|\bcircuit\b"
    r"|\bmechanical (?:arrangement|layout)\b"
    r"|\b(?:blueprint|schematic|topology)\b"
    r"|\bmolecular formula\b"
    r"|\blayout\b"
    r"|\bspatial (?:arrangement|layout)\b",
    re.IGNORECASE,
)

QUANTITY = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:°\s?C|degrees? C|%|g\b|kg|mg|mL|ml|L\b|mol|M\b|psi|GHz|GB|TB|"
    r"kbps|Mbps|steps?|rounds?|yield)\b"
    r"|\b\d+\s*:\s*\d+\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve) steps\b",
    re.IGNORECASE,
)

# 逃避约束(全局修饰,docs/10 §1.3 事实 4 / §4.9)
EVASION = re.compile(
    r"\bwithout getting caught\b"
    r"|\bwithout being (?:caught|traced|suspected|detected)\b"
    r"|\bwithout (?:detection|a trace|leaving a trace|leaving traces)\b"
    r"|\bhard to trace\b"
    r"|\buntraceable\b"
    r"|\banonymously\b"
    r"|\bundetected\b"
    r"|\bwhile evading [\w\s]+"
    r"|\bevade [\w\s]{2,40}?(?=[,.;]|$)"
    r"|\bavoid(?:ing)? detection\b",
    re.IGNORECASE,
)

# 非逃避型限制/条件
CONDITION = re.compile(
    r"\bwithout access to specialized tools\b"
    r"|\bwithout specialized tools\b"
    r"|\bat home\b"
    r"|\bfrom common household materials\b"
    r"|\busing the (?:one-pot|Levenstein|shikimic acid|[A-Za-z-]+) (?:method|process)\b"
    r"|\bin (?:the )?(?:US|United States|Mexico|South Africa|China|a city|a hospital|a subway|a modern airport)\b"
    r"|\bat university\b"
    r"|\bon a (?:Windows XP|Linux) computer\b"
    r"|\bvia USB\b"
    r"|\bas a minor\b",
    re.IGNORECASE,
)

# 材料/来源标记:触发 precursor-of 依赖边
PRECURSOR_OF = re.compile(r"\bfrom\s+(?P<src>[\w\s-]{3,40}?)\s+(?:for|to|into)\b", re.IGNORECASE)

# core 抽取失败时的兜底长度上限(避免把整句当 core)
MAX_CORE_CHARS = 400
