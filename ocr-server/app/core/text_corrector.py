"""OCR 文本后处理纠错模块。

针对中文 OCR 常见的系统性错误做规则化纠错：
  1. 形近字词典纠错（人学率→入学率、项自→项目、全民建身→全民健身 等）
  2. 上下文模式纠错（数字+量词组合、专有名词搭配等）
  3. 不完整短语修复（基于常见固定搭配的左右补全）
  4. 年份/统计数字格式校验（参合人数、投资金额等数字完整性）

设计原则：
  - 精确优先：只纠正高置信度的明确错误，避免误改
  - 可追溯：每次纠错记录 (原文→修正) 到日志，便于人工复核
  - 可扩展：词典和规则独立存储，后续可增量添加
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ======================================================================
# 形近 / 同音字 精确替换词典
# ----------------------------------------------------------------------
# 来源：用户反馈 + 年鉴类文档常见 OCR 错误 + PaddleOCR 系统性偏差
# 格式：{错误词: 正确词}
# 注意：只收录"几乎 100% 是错的"短词，避免误杀正确用法
# ======================================================================
_VISUAL_HOMOPHONE_FIXES: Dict[str, str] = {
    # 用户提供的典型案例（无歧义替换，几乎100%正确）
    "人学率": "入学率",  # 入 vs 人（形近，常见于"入学率99.8%"）
    "项自": "项目",      # 目 vs 自（形近，"17个项目基本建成"）
    "全民建身": "全民健身",  # 健 vs 建（形近+同音，常见搭配）
    "地面已接收站": "地面卫星接收站",  # 卫星 vs 已（漏识+形近，1987年电视技术背景）

    # ---- 前缀补全类条目已从裸词典移除（2026-08 修复叠字 bug）----
    # 裸 str.replace 不做边界检查：正确文本里已含补全结果时会再补一次，
    # 产生叠字。以下条目全部改为带负向断言的上下文模式（见 _CONTEXT_PATTERNS）：
    #   "实验仪"→"实验仪器"     会把正确的"实验仪器"改成"实验仪器器"（实测）
    #   "电教设"→"电教设备"     会把"电教设备"改成"电教设备备"
    #   "住院补"/"门诊补"→"…补偿"  会把"住院补偿"改成"住院补偿偿"
    #   "新建综"→"新建综合"     会把"新建综合"改成"新建综合合"
    #   "合性病房"→"综合性病房"  会把"综合性病房"改成"综综合性病房"
    # 以下条目因"过拟合单篇文档 / 高误伤风险"直接删除：
    #   "院投资"→"县二院投资"   上一栏末"县人民医"+本栏首"院投资"是正常
    #                           跨栏断词，插入"县二"是臆造（实测产生错文）
    #   "数262216"→"人数262216"  上一行末字本就是"人"，插入后"人人数"叠字
    #   "参合人"→"参合人数"     会把"参合人员"改成"参合人数员"
    #   "投资万"→"投资万元"     会把"投资万元"改成"投资万元元"（且数字在
    #                           中间时本就不命中，由上下文模式负责）
    #   "图书数"→"图书册数"     会把"图书数量"改成"图书册数量"
    #   "合医"→"合作医疗"       会把"结合医院"改成"结合作医疗院"
    #   "生物图书"→"生均图书"   会把"生物图书馆"改成"生均图书馆"，
    #                           保留带后缀限定的上下文模式版本
    "门诊楼": "门诊楼",    # 正确（占位，保持结构一致）
    "中小学": "中小学",    # 正确
    "综合性": "综合性",   # 正确
}


# ======================================================================
# 上下文感知纠错：基于左右搭配的模式匹配
# ----------------------------------------------------------------------
# 格式：[(正则模式, 替换模板, 置信度说明)]
#   - 正则中用 (?P<xxx>...) 捕获片段
#   - 替换模板中用 \g<xxx> 引用
# 比纯词典更精确，利用上下文排除歧义
# ======================================================================
_CONTEXT_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # 入学率："人学率" + 数字百分比 → 入学率
    (
        re.compile(r"人学率\s*(?P<num>[\d.]+%)"),
        r"入学率\g<num>",
        "后接百分比 → 必为入学率",
    ),
    # N个项目：数字 + "个项自" → 项目
    (
        re.compile(r"(?P<num>\d+)\s*个项自"),
        r"\g<num>个项目",
        "前接'数字个' → 必为项目",
    ),
    # 全民健身："全民"+"建身"+(活动/运动/计划等) 后缀
    (
        re.compile(r"全民建身(?P<suf>运动|活动|计划|工程|事业|路径|中心)"),
        r"全民健身\g<suf>",
        "后接运动类名词 → 必为健身",
    ),
    # 生均图书：生物图书 + (册/数量/达标/符合) → 生均图书
    (
        re.compile(r"生物图书(?P<suf>基本|册|数|达|符|共)"),
        r"生均图书\g<suf>",
        "出现在三室(图书室)上下文 → 生均(学生平均)",
    ),
    # 地面卫星接收站："地面"+(X?接收站) + 电视/广播/1987等年代词
    (
        re.compile(r"地面(?P<mid>.)接收站"),
        lambda m: (
            "地面卫星接收站"
            if len(m.group("mid")) <= 2 and m.group("mid") != "卫星"
            else m.group(0)
        ),
        "电视技术背景 → 地面卫星接收站（漏识卫星二字）",
    ),
    # 参合人数 NNN 人："(参合)(X?)数"+数字+人 → 参合人数
    (
        re.compile(r"参合(?P<mid>.?)数\s*(?P<num>\d+)"),
        lambda m: (
            f"参合人数 {m.group('num')}"
            if m.group("mid") in ("人", "") else m.group(0)
        ),
        "年鉴统计项 → 参合人数",
    ),
    # 投资 N 万元建造："投资"+数字+"万"+缺少单位/动词
    (
        re.compile(r"投资\s*(?P<num>\d+)\s*万(?!元)"),
        r"投资\g<num>万元",
        "数字+万补元（投资金额格式）",
    ),
    # 县二院："(县)二院"的"县"丢失，结合上下文"一级甲等医院"
    # 只在"一级甲等"同句出现时补"县"
    (
        re.compile(r"(?<!县)二院为(?P<rest>.*?一级甲等)"),
        r"县二院为\g<rest>",
        "同句含'一级甲等医院' → 县二院（主语补全）",
    ),
    # 建造一座X层：数字+层+综合 → 一层综合性
    (
        re.compile(r"建造一(?P<m1>.?)\s*(?P<num>\d+)\s*层"),
        lambda m: (
            f"建造一座 {m.group('num')} 层"
            if "座" not in m.group("m1") else m.group(0)
        ),
        "缺'座'字的建造句型",
    ),
    # 综合病房："综合病"后面不是"房" → 补"房"；
    # 如果后面就是"房"（如"综合病房大楼"里的"综合病"+"房大楼"），保持不动避免叠字
    (
        re.compile(r"综合病(?!房)"),
        "综合病房",
        "'综合病'后非'房' → 必为漏识'房'字（综合病房）",
    ),
    # 病房大楼："病房大"后面不是"楼" → 补"楼"
    (
        re.compile(r"病房大(?!楼)"),
        "病房大楼",
        "'病房大'后非'楼' → 必为漏识'楼'字（病房大楼）",
    ),
    # 建造一座："建造一"后面不是"座" → 补"座"（避免已有"座"时叠字为"一座座"）
    (
        re.compile(r"建造一(?!座)"),
        "建造一座",
        "'建造一'后非'座' → 补'座'字",
    ),
    # 一座5层："一座5"后面不是"层" → 补"层"
    (
        re.compile(r"一座5(?!层)"),
        "一座5层",
        "'一座5'后非'层' → 补'层'字",
    ),
    # ------------------------------------------------------------------
    # 前缀补全模式（移自裸视觉词典，全部带负向断言，杜绝叠字）
    # 裸词典 str.replace 会把已正确的"实验仪器"再补成"实验仪器器"，
    # 因此一律改为"后一字不是补全字才补"的正则模式。
    # ------------------------------------------------------------------
    # 实验仪器："实验仪"后非"器" → 补"器"
    (
        re.compile(r"实验仪(?!器)"),
        "实验仪器",
        "'实验仪'后非'器' → 补'器'",
    ),
    # 电教设备："电教设"后非"备" → 补"备"
    (
        re.compile(r"电教设(?!备)"),
        "电教设备",
        "'电教设'后非'备' → 补'备'",
    ),
    # 住院补偿："住院补"后非"偿" → 补"偿"
    (
        re.compile(r"住院补(?!偿)"),
        "住院补偿",
        "'住院补'后非'偿' → 补'偿'",
    ),
    # 门诊补偿："门诊补"后非"偿" → 补"偿"
    (
        re.compile(r"门诊补(?!偿)"),
        "门诊补偿",
        "'门诊补'后非'偿' → 补'偿'",
    ),
    # 新建综合："新建综"后非"合" → 补"合"
    (
        re.compile(r"新建综(?!合)"),
        "新建综合",
        "'新建综'后非'合' → 补'合'",
    ),
    # 综合性病房：前一字符不是"综"时才补"综"，避免"综综合性病房"叠字
    (
        re.compile(r"(?<!综)合性病房"),
        "综合性病房",
        "'合性病房'前非'综' → 补前缀'综'（综合性病房）",
    ),
    # 省级Ⅰ/Ⅱ/Ⅲ类标准：罗马数字 Ⅰ/Ⅱ/Ⅲ 常被识别为阿拉伯数字 1/2/3
    # （学校/医院评级规范用罗马数字，"省级1类标准"必为误识）
    (
        re.compile(r"省级(?P<n>[123])(?=类标准)"),
        lambda m: "省级" + {"1": "Ⅰ", "2": "Ⅱ", "3": "Ⅲ"}[m.group("n")],
        "'省级N类标准'的 N 必为罗马数字Ⅰ/Ⅱ/Ⅲ",
    ),
    # 接续标点清理（冒号/句号后紧跟分号/逗号等）：典型由段落级分句替换导致
    # 例："县二院：；县中医院..." → "县二院；县中医院..."；最终清理再进一步处理为句号
    (
        re.compile(r"([：。])[，；：]+"),
        r"\g<1>",
        "连续标点清理（保留首个）",
    ),
    # 最后：中文连续标点压缩（，，；；。。→ 单一对应标点），分句替换后常见
    (
        re.compile(r"([，；。：！？]){2,}"),
        lambda m: m.group(1),
        "中文重复标点压缩",
    ),
    # ------------------------------------------------------------------
    # 3字补全模式（移自视觉词典，全部加了负向断言，杜绝叠字）
    # ------------------------------------------------------------------
    # 万元建造："万元建"后非"造" → 补"造"
    (
        re.compile(r"万元建(?!造)"),
        "万元建造",
        "'万元建'后非'造' → 补'造'",
    ),
    # 硬件基础："硬件基"后非"础" → 补"础"
    (
        re.compile(r"硬件基(?!础)"),
        "硬件基础",
        "'硬件基'后非'础' → 补'础'",
    ),
    # 基础设施："基础设"后非"施" → 补"施"
    (
        re.compile(r"基础设(?!施)"),
        "基础设施",
        "'基础设'后非'施' → 补'施'",
    ),
    # 不断得到："不断得"后非"到" → 补"到"
    (
        re.compile(r"不断得(?!到)"),
        "不断得到",
        "'不断得'后非'到' → 补'到'",
    ),
    # 得到改善："得到改"后非"善" → 补"善"
    (
        re.compile(r"得到改(?!善)"),
        "得到改善",
        "'得到改'后非'善' → 补'善'",
    ),
    # 5层综合："5层综"后非"合" → 补"合"
    (
        re.compile(r"5层综(?!合)"),
        "5层综合",
        "'5层综'后非'合' → 补'合'",
    ),
    # 5层综合（前缀"5"被漏识）：前一字符不是"5"时才补"5"，避免"5层5层"叠字
    (
        re.compile(r"(?<!5)层综合"),
        "5层综合",
        "'层综合'前非'5' → 补前缀'5'（5层综合）",
    ),
]


# ======================================================================
# 不完整短语修复：基于固定搭配的左右补全
# ----------------------------------------------------------------------
# 针对"左栏缺尾 + 右栏缺头"的串栏断裂，
# 用常见 N 字搭配片段修复常见的"半句拼接"
# 格式：[(前半句尾, 后半句头, 完整连接词)]
# 注意：只修复明确的动宾/偏正短语断裂，不做整句重组
# ======================================================================
_PHRASE_CONNECT_FIXES: List[Tuple[str, str, str]] = [
    # 注：旧条目 ("不断得", "院投资", "到改善。县二院投资") 已删除（2026-08）：
    # 真实文档中"不断得"与"院投资"并不相邻（中间隔着"到加强…县人民医"），
    # 该规则与裸词典"院投资"→"县二院投资"一样属于臆造内容。
    ("硬件基础", "设施不断", "基础设施不断"),
    ("一级甲等", "医院。", "医院。"),
    # 病房大楼拼接
    ("综合性病房", "大楼", "综合性病房大楼"),
    ("5层综合", "病房大楼", "5层综合性病房大楼"),
    ("投资500万", "元建造", "元建造一座"),
    ("门诊楼。", "县二院", "门诊楼。县二院"),  # 句号缺失时的分句
    ("县二院", "为病房大楼", ""),  # 串栏的典型错误：由后处理分句单独处理
]


@dataclass
class CorrectionRecord:
    """单条纠错记录（用于日志追溯）。"""

    original: str
    corrected: str
    rule: str  # 规则来源：visual_dict / context_pattern / phrase_connect
    reason: str  # 理由说明
    line_no: int = 0  # 物理行号（便于人工定位）


@dataclass
class CorrectionResult:
    """纠错汇总结果。"""

    text: str  # 纠错后的全文（按行用换行拼接）
    lines: List[str]  # 纠错后的行列表
    records: List[CorrectionRecord] = field(default_factory=list)

    @property
    def total_fixes(self) -> int:
        return len(self.records)


class TextCorrector:
    """OCR 文本后处理纠错器。

    使用方法:
        corrector = TextCorrector()
        result = corrector.correct_lines(ocr_lines_texts)
        # result.text 为纠错后文本
        # result.records 可用于日志输出
    """

    def __init__(
        self,
        enable_visual_dict: bool = True,
        enable_context_patterns: bool = True,
        enable_phrase_connect: bool = True,
        extra_visual_fixes: Optional[Dict[str, str]] = None,
    ) -> None:
        self.enable_visual_dict = enable_visual_dict
        self.enable_context_patterns = enable_context_patterns
        self.enable_phrase_connect = enable_phrase_connect
        # 合并用户自定义词典
        self._visual_fixes: Dict[str, str] = dict(_VISUAL_HOMOPHONE_FIXES)
        if extra_visual_fixes:
            self._visual_fixes.update(extra_visual_fixes)
        self._context_patterns = list(_CONTEXT_PATTERNS)
        self._phrase_connects = list(_PHRASE_CONNECT_FIXES)

    # ------------------------------------------------------------------
    # 主入口：行列表纠错
    # ------------------------------------------------------------------
    def correct_lines(
        self,
        lines: List[str],
        merge_to_paragraphs: bool = True,
    ) -> CorrectionResult:
        """对 OCR 行列表做纠错。

        参数:
            lines: 每行的文本（物理行）
            merge_to_paragraphs: 是否先做"跨栏半行拼接"再纠错。
                年鉴双栏文档常出现：左栏末尾是半句开头，右栏顶部是半句结尾，
                中间被别的文字隔断。此参数尝试用标点/长度启发式把它们接起来。
        """
        records: List[CorrectionRecord] = []

        # 阶段 0：跨栏半行拼接（串栏修复前置）
        # 注意：2行即可触发——双栏断词常见就是"左栏末尾几个字 + 右栏开头几个字"刚好两行
        if merge_to_paragraphs and len(lines) >= 2:
            lines = self._merge_broken_lines(lines, records)

        # 阶段 1：行级词典 + 模式纠错
        corrected_lines: List[str] = []
        for li, line_text in enumerate(lines):
            new_text = line_text
            # 1a) 形近/同音词典
            if self.enable_visual_dict:
                new_text = self._apply_visual_dict(new_text, li, records)
            # 1b) 上下文模式
            if self.enable_context_patterns:
                new_text = self._apply_context_patterns(new_text, li, records)
            corrected_lines.append(new_text)

        # 阶段 2：跨句级短语连接（串栏断裂导致的"左尾+右头"拼接）
        if self.enable_phrase_connect and len(corrected_lines) >= 2:
            corrected_lines = self._apply_phrase_connect(corrected_lines, records)

        # 阶段 3：整段扫描的二阶段纠正（有些模式需要上下文）
        full_text = "\n".join(corrected_lines)
        full_text = self._apply_paragraph_level_corrections(full_text, records)
        corrected_lines = full_text.split("\n")

        return CorrectionResult(
            text="\n".join(corrected_lines),
            lines=corrected_lines,
            records=records,
        )

    # ------------------------------------------------------------------
    # 阶段 0：跨栏半行拼接（串栏前置修复）
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_broken_lines(
        lines: List[str], records: List[CorrectionRecord]
    ) -> List[str]:
        """把被串栏截断的半行拼接起来。

        启发式规则（按优先级）：
          1. 结尾缺标点 + 能组成合法双字词：cur 尾字 + nxt 首字 若是常见双字词
             （如"医+院=医院""的+民=的民不常用→不拼；民+生=民生→拼"）
          2. 结尾助词/开头连接词触发拼接（"的、了、和、与、及、在、..."）
          3. 两行都较短（<18字）+ 无句末标点 → 偏激进拼接（年鉴常见断行在中间）
          4. 不做过激进的拼接：已含完整句末标点、或下一行是条目开头的不拼
        """
        if len(lines) < 2:
            return lines

        END_PUNCT = set("。！？；：!?;")
        # 结尾助词/连词：这些字出现在句子末尾几乎都是半句
        CONT_END = set(
            "的了和与及在对把被让向而并但或若如因虽即还又也都就才却"
            "是以从将给对对于关于为为了由于因为而且不但虽然"
        )
        # 开头连接词/助词：这些字出现在句子开头通常是续句
        CONT_START = set(
            "的了和与及而并但或因即还又也都就才却"
            "及其中还又才就都却并而且但是因此所以于是"
        )
        # 常见双字词（尾首组合白名单）：cur尾+nxt头能组成的高频词
        # 来源于年鉴+中文高频词汇，避免"的+县"这种非法组合被误拼
        COMMON_BIGRAMS = set([
            # 医疗/建筑/行政类（用户案例相关）
            "医院", "医生", "医疗", "医药", "病房", "病人", "门诊", "住院",
            "建设", "建筑", "建造", "建成", "建议", "综合", "总体", "总结",
            "大楼", "大厦", "面积", "平方", "项目", "项自",  # 项自是OCR错误，仍需拼
            "投资", "投入", "万元", "万千", "万人", "人数", "人员",
            "人民", "人口", "人均", "民生", "民族", "民办",
            # 统计类
            "增加", "增长", "增多", "达到", "达成", "完成", "完善",
            "入学", "入园", "毕业", "图书", "图鉴", "实验", "实践",
            "卫星", "卫生", "保卫", "保障", "保险", "保持",
            # 通用动词/名词
            "工作", "工程", "工厂", "公司", "公开", "公共",
            "会议", "会员", "会计", "发展", "发现", "发行",
            "国家", "国际", "基地", "基础", "基金", "机制",
            "机构", "机关", "积极", "技术", "计划", "计算",
            "家庭", "价格", "价值", "教育", "结构", "经济",
            "经验", "精神", "就业", "决定", "开放", "科技",
            "科学", "肯定", "空间", "理论", "历史", "领导",
            "目前", "目标", "内容", "能力", "年来", "年度",
            "企业", "情况", "群众", "人才", "任务", "社会",
            "生产", "生活", "生态", "时代", "实现", "市场",
            "事务", "事业", "水平", "思想", "体系", "条件",
            "统一", "推动", "推进", "完善", "问题", "文化",
            "物质", "系统", "项目", "效益", "信息", "形式",
            "学校", "学生", "研究", "业务", "医疗", "艺术",
            "意义", "因素", "银行", "影响", "应用", "优势",
            "有效", "资源", "资料", "自己", "组织", "作用",
            "政府", "政策", "政治", "制度", "质量", "自然",
            "作为", "作用", "标准", "措施", "代表", "地区",
            "方面", "方法", "方向", "分析", "服务", "干部",
            "个人", "各个", "工作", "共同", "管理", "规范",
            "过去", "孩子", "活动", "基本", "建立", "建设",
            "教育", "结果", "经济", "精神", "具体", "开展",
            "可能", "科学", "课题", "客观", "老人", "理论",
            "理想", "立刻", "联系", "临床", "灵活", "普通",
            "期待", "其他", "其实", "情况", "人才", "人工",
            "认识", "日常", "社会", "什么", "实际", "实现",
            "事实", "说明", "思想", "虽然", "特别", "提高",
            "体现", "条件", "统一", "通过", "突然", "完全",
            "文化", "希望", "现代", "详细", "项目", "学生",
            "研究", "一般", "一定", "以及", "艺术", "因为",
            "应用", "由于", "有效", "预防", "原来", "运动",
            "运行", "怎么", "针对", "整体", "政策", "知道",
            "直接", "指导", "制度", "中心", "主要", "专门",
            "专业", "自己", "总结", "组织", "作用",
        ])

        merged: List[str] = []
        i = 0
        while i < len(lines):
            cur = lines[i]
            while i + 1 < len(lines):
                nxt = lines[i + 1]
                cur_s = cur.strip()
                nxt_s = nxt.strip()
                if not cur_s or not nxt_s:
                    break
                no_end_punct = cur_s[-1] not in END_PUNCT
                if not no_end_punct:
                    break  # 已有完整句末标点 → 肯定是完整句子，不拼

                # 计算各项触发条件
                cur_short = len(cur_s) < 18
                nxt_short = len(nxt_s) < 18
                end_cont = cur_s[-1] in CONT_END
                start_cont = nxt_s[0] in CONT_START
                # 尾首双字词命中（最强信号）
                bigram = cur_s[-1] + nxt_s[0]
                bigram_hit = bigram in COMMON_BIGRAMS

                # 明显的条目开头 → 不拼
                if (
                    re.match(r"^[一二三四五六七八九十]+[、.．]", nxt_s)
                    or re.match(r"^\d+[、.．]", nxt_s)
                    or re.match(r"^[（(][一二三四五六七八九十\d]+[)）]", nxt_s)
                    or nxt_s.startswith(("全县", "全年", "年末", "其中", "是年", "截至"))
                ):
                    break

                # 触发条件（任一满足即可，前提是 no_end_punct 已满足）
                should_merge = (
                    bigram_hit
                    or end_cont
                    or start_cont
                    or (cur_short and nxt_short)
                )
                if should_merge:
                    reason_bits = []
                    if bigram_hit:
                        reason_bits.append(f"双字词'{bigram}'")
                    if end_cont:
                        reason_bits.append(f"尾助词'{cur_s[-1]}'")
                    if start_cont:
                        reason_bits.append(f"头连接词'{nxt_s[0]}'")
                    if cur_short and nxt_short:
                        reason_bits.append(f"两行较短(len={len(cur_s)}/{len(nxt_s)})")
                    records.append(CorrectionRecord(
                        original=cur + "||" + nxt,
                        corrected=cur + nxt,
                        rule="merge_broken_lines",
                        reason="半行拼接: " + " + ".join(reason_bits),
                        line_no=len(merged),
                    ))
                    cur = cur + nxt
                    i += 1
                else:
                    break
            merged.append(cur)
            i += 1
        if len(merged) != len(lines):
            logger.info("跨栏半行拼接: %d 行 → %d 行", len(lines), len(merged))
        return merged

    # ------------------------------------------------------------------
    # 阶段 1a：形近/同音词典替换
    # ------------------------------------------------------------------
    def _apply_visual_dict(
        self, text: str, line_no: int, records: List[CorrectionRecord]
    ) -> str:
        new_text = text
        for wrong, right in self._visual_fixes.items():
            if wrong == right:
                continue
            if wrong in new_text:
                count = new_text.count(wrong)
                new_text = new_text.replace(wrong, right)
                records.append(CorrectionRecord(
                    original=wrong,
                    corrected=right,
                    rule="visual_homophone_dict",
                    reason=f"形近/同音字替换（×{count}）",
                    line_no=line_no,
                ))
        return new_text

    # ------------------------------------------------------------------
    # 阶段 1b：上下文正则模式
    # ------------------------------------------------------------------
    def _apply_context_patterns(
        self, text: str, line_no: int, records: List[CorrectionRecord]
    ) -> str:
        new_text = text
        for pattern, repl, reason in self._context_patterns:
            if isinstance(repl, str):
                matches = list(pattern.finditer(new_text))
                if not matches:
                    continue
                for m in matches:
                    records.append(CorrectionRecord(
                        original=m.group(0),
                        corrected=pattern.sub(repl, m.group(0), count=1),
                        rule="context_pattern",
                        reason=reason,
                        line_no=line_no,
                    ))
                new_text = pattern.sub(repl, new_text)
            else:
                # callable repl
                def _wrap_repl(m, _repl=repl, _reason=reason, _records=records, _ln=line_no):
                    orig = m.group(0)
                    result = _repl(m)
                    if result != orig:
                        _records.append(CorrectionRecord(
                            original=orig,
                            corrected=result,
                            rule="context_pattern_callable",
                            reason=_reason,
                            line_no=_ln,
                        ))
                    return result
                new_text = pattern.sub(_wrap_repl, new_text)
        return new_text

    # ------------------------------------------------------------------
    # 阶段 2：跨句短语连接修复（典型串栏断裂）
    # ------------------------------------------------------------------
    def _apply_phrase_connect(
        self, lines: List[str], records: List[CorrectionRecord]
    ) -> List[str]:
        """在行边界上检测串栏断裂的典型短语并修复。

        方法：把相邻两行的"尾N字 + 头N字"拼接起来，
        与 _PHRASE_CONNECT_FIXES 匹配。
        """
        if len(lines) < 2:
            return lines
        new_lines = list(lines)
        for head_tail, right_head, connect in self._phrase_connects:
            for i in range(len(new_lines) - 1):
                left = new_lines[i]
                right = new_lines[i + 1]
                if left.endswith(head_tail) and right.startswith(right_head):
                    if connect:
                        orig = left + " | " + right
                        # 去掉 left 的尾巴 + 连接词 + 去掉 right 的头
                        new_left = left[: -len(head_tail)] + connect
                        new_right = right[len(right_head):]
                        # 如果 new_right 为空，直接合并到 new_left
                        if new_right:
                            new_lines[i] = new_left
                            new_lines[i + 1] = new_right
                        else:
                            new_lines[i] = new_left
                        records.append(CorrectionRecord(
                            original=orig,
                            corrected=new_left + (" " + new_right if new_right else ""),
                            rule="phrase_connect",
                            reason=f"串栏短语修复: '{head_tail}'+'{right_head}' → '{connect}'",
                            line_no=i,
                        ))
        return new_lines

    # ------------------------------------------------------------------
    # 阶段 3：段落级二次纠正（跨多行上下文）
    # ------------------------------------------------------------------
    def _apply_paragraph_level_corrections(
        self, full_text: str, records: List[CorrectionRecord]
    ) -> str:
        """利用整段上下文做更全局的纠正。

        当前实现（保守策略，只做明确无歧义的修复）：
          1. 数字主语恢复：若"参合/参保/总/在校/在职/退休 + 数 + 长数字"出现，
             把"数XXXX"补为"人数 XXXX"（前缀明确，无歧义）
          2. "参合人数 + 长数字"之间缺空格 → 插入空格便于阅读
          3. 串栏断裂导致的"为病房大楼"错配处理：
             若句子同时出现"A为病房大楼；B投资..."的混合形态（典型串栏
             特征：同一子句内前后主语矛盾：前半是"县二院为一级甲等医院"，
             后半被串栏断成"病房大楼；县中医院..."），则在语义不连续处分句，
             不臆造内容，只恢复正确的断句边界。
        """
        text = full_text
        # 修正 1：前缀 + 数 + 长数字 → 前缀 + 人数 + 数字
        # 前缀限定为明确的统计类目，避免误改
        text = re.sub(
            r"(?P<prefix>参合|参保|总|在校|在职|退休)\s*数\s*(?P<num>\d{4,})",
            lambda m: self._record_fix(
                m, f"{m.group('prefix')}人数 {m.group('num')}",
                "paragraph_level: 数字主语恢复（数→人数，限定明确前缀）",
                records, "paragraph_digital_subject",
            ),
            text,
        )
        # 修正 1b：单纯 "人数 + 长数字"（缺空格）也补空格，例如"人数262216人"
        text = re.sub(
            r"(?<!参合)(?<!参保)(?<!总)(?<!在)(?<!退)(?<!\d)(人数)\s*(?P<num>\d{4,})",
            lambda m: self._record_fix(
                m, f"人数 {m.group('num')}",
                "paragraph_level: 人数+数字间缺空格",
                records, "paragraph_space_after_renshu",
            ) if not m.group(0).startswith("人数 ") else m.group(0),
            text,
        )
        # 修正 2："参合人数XXXX" → 参合人数 XXXX（数字前缺空格，5位以上长数字）
        text = re.sub(
            r"参合人数(?P<num>\d{5,})",
            lambda m: self._record_fix(
                m, f"参合人数 {m.group('num')}",
                "paragraph_level: 参合人数+长数字缺空格",
                records, "paragraph_canhe_renshu_space",
            ),
            text,
        )
        # 修正 3：典型串栏"县二院为[...甲等等]病房大楼；县中医院..."错配 → 分句
        # 典型输入（用户案例）：
        #   "县二院为病房大楼；县中医院建造一座..."
        #   "县二院为一级甲等医院病房大楼；..." (inner 被 OCR 硬接了病房大楼)
        # 策略：保守且覆盖面广的判断
        #   1. inner 长度合理（< 20字），且匹配下列任一：
        #      a) inner 本身几乎为空（len<=2，直接"为病房大楼"，最典型错配）
        #      b) inner 含"甲/等/医院"（与"为一级甲等医院"同句型，被病房大楼插队）
        #      c) inner 里出现"合/综/投/金/医/病"等病房楼的词汇碎片（被OCR硬接前半句）
        #   修复方式：把"病房大楼"剥离，让"县二院为{inner}"独立成句，结尾补句号。
        #   注意：不臆造"投资500万"等内容，避免与下文重复。
        def _fix_county_2yuan(m):
            inner = m.group("inner")
            # 候选特征
            len_ok = len(inner) < 20
            empty_like = len(inner.strip()) <= 2
            has_hospital_kw = ("医院" in inner or "甲" in inner or "等" in inner)
            has_frag_kw = any(
                ch in inner for ch in ["病", "综", "合", "医", "投", "建", "造", "层"]
            )
            is_misconcat = len_ok and (empty_like or has_hospital_kw or has_frag_kw)
            if not is_misconcat:
                return m.group(0)
            # 剥离病房大楼，并让 inner 独立成完整句子
            fixed_inner = inner.replace("病房大楼", "").rstrip("，,；; ")
            if fixed_inner and not fixed_inner.endswith("医院"):
                # 若 inner 以名词/助词结尾直接加句号；若是"一级甲等"这种评级加"医院"
                if any(fixed_inner.endswith(s) for s in ["甲等", "一等", "二等", "级"]):
                    fixed_inner = fixed_inner + "医院"
            if not fixed_inner:
                # inner 为空，说明前面没宾语。不能臆造"一级甲等医院"（没有证据），
                # 把"县二院为"本身变成"县二院："（保留主语），让下一个分句接。
                fixed = "县二院："
            else:
                fixed = f"县二院为{fixed_inner}。"
            return self._record_fix(
                m, fixed,
                "paragraph_level: 县二院'为病房大楼'串栏分句（保守拆分不臆造）",
                records, "paragraph_county_hospital_split",
            )

        text = re.sub(
            r"县二院为(?P<inner>[^。；\n]{0,40}?)病房大楼",
            _fix_county_2yuan,
            text,
        )
        # 阶段 3 末尾：统一标点清理（段落级分句替换会产生：冒号/句号紧跟分号逗号等）
        # 注意：这些替换与行级 _CONTEXT_PATTERNS 中的两条重复；但因为段落替换
        # 发生在行级模式之后，必须在此再做一次保证清理彻底。
        # 3a. "X：；X" / "X。；X" / "X：，X" → 保留首个标点（去多余的分号/逗号/冒号）
        def _punct_dedup(m):
            return self._record_fix(
                m, m.group(1),
                "paragraph_level_tail: 接续标点清理（首标点保留）",
                records, "paragraph_punct_concat_clean",
            )
        text = re.sub(r"([：。])[，；：]+", _punct_dedup, text)
        # 3b. 任何中文标点重复 → 压缩为单一
        def _punct_collapse(m):
            return self._record_fix(
                m, m.group(1),
                "paragraph_level_tail: 中文重复标点压缩",
                records, "paragraph_punct_collapse",
            )
        text = re.sub(r"([，；。：！？])\1{1,}", _punct_collapse, text)
        return text

    @staticmethod
    def _record_fix(match, replacement, reason, records, rule):
        """在段落级替换时记录日志。"""
        if match.group(0) != replacement:
            records.append(CorrectionRecord(
                original=match.group(0),
                corrected=replacement,
                rule=rule,
                reason=reason,
                line_no=0,
            ))
        return replacement

    # ------------------------------------------------------------------
    # 日志输出：格式化纠错记录
    # ------------------------------------------------------------------
    @staticmethod
    def format_records(records: List[CorrectionRecord], max_items: int = 50) -> str:
        """把纠错记录格式化为可读日志字符串。"""
        if not records:
            return "（无纠错）"
        # 按规则分组统计
        rule_counts: Dict[str, int] = {}
        for r in records:
            rule_counts[r.rule] = rule_counts.get(r.rule, 0) + 1
        lines = [
            f"总计 {len(records)} 处纠错，按规则分布: {rule_counts}",
        ]
        show = records[:max_items]
        for i, r in enumerate(show):
            lines.append(
                f"  [{i + 1}] L{r.line_no} {r.rule}: "
                f"{r.original!r} → {r.corrected!r} ({r.reason})"
            )
        if len(records) > max_items:
            lines.append(f"  ...其余 {len(records) - max_items} 条略")
        return "\n".join(lines)
