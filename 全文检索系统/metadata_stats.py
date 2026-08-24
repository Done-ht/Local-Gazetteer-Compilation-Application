"""元数据统计模块：基于分词技术从文本中提取统计特征。

统计维度：
1. 朝代/时代标记：朝代词频率 + 年号识别，用于 title_filter 跨朝代过滤
2. 主题领域：经济/政治/军事/文化/地理 关键词分布，用于查询路由
3. 实体密度：人名/地名/机构名密度（jieba 词性标注），辅助实体消歧

统计策略：
- 大 chunk 粒度（~10000字）统计，避免小 chunk 词频波动
- 窗口滑动平均：同文档多个 chunk 的统计值做邻域平均，得到文档级稳定特征

jieba 为可选依赖，未安装时统计功能降级（跳过实体密度，用正则匹配朝代/主题词）。
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Any

# ---------------- 朝代/年号词表 ----------------

# 朝代名 → 关联关键词（用于检测文档所属朝代）
# 关键词中包含朝代名本身（如"南宋"二字也参与识别），并避免易混淆地名
_DYNASTY_KEYWORDS: Dict[str, List[str]] = {
    "先秦": ["春秋", "战国", "左传", "公羊", "谷梁", "周天子", "诸侯", "卿大夫", "先秦"],
    "秦": ["秦始皇", "秦二世", "嬴政", "蒙恬", "李斯", "焚书坑儒", "秦朝"],
    "西汉": ["汉高祖", "汉武帝", "汉昭帝", "汉宣帝", "刘邦", "卫青", "霍去病", "董仲舒", "司马迁", "张骞", "苏武", "西汉"],
    "东汉": ["光武帝", "汉明帝", "汉章帝", "刘秀", "班固", "张衡", "蔡伦", "窦宪", "东汉"],
    "三国": ["曹操", "曹丕", "曹叡", "刘备", "诸葛亮", "孙权", "周瑜", "司马懿", "关羽", "张飞", "赵云", "三国"],
    "西晋": ["晋武帝", "司马炎", "羊祜", "杜预", "王濬", "贾南风", "八王之乱", "西晋"],
    "东晋": ["晋元帝", "司马睿", "王导", "谢安", "桓温", "刘裕", "淝水之战", "东晋"],
    "南朝宋": ["宋武帝", "刘裕", "刘义隆", "元嘉", "南朝宋"],
    "南齐": ["齐高帝", "萧道成", "萧赜", "南齐"],
    "南梁": ["梁武帝", "萧衍", "萧统", "昭明太子", "南梁"],
    "南陈": ["陈武帝", "陈霸先", "陈后主", "陈叔宝", "南陈"],
    "北魏": ["魏道武帝", "拓跋珪", "拓跋宏", "孝文帝", "冯太后", "均田制", "北魏"],
    "北齐": ["齐文宣帝", "高欢", "高洋", "高澄", "北齐"],
    "北周": ["周文帝", "宇文泰", "宇文觉", "宇文邕", "北周武帝", "北周"],
    "隋": ["隋文帝", "隋炀帝", "杨坚", "杨广", "开皇", "大业", "科举", "大运河", "隋朝"],
    "唐": ["唐高祖", "唐太宗", "唐高宗", "武则天", "唐玄宗", "李渊", "李世民", "贞观", "开元", "天宝", "安史之乱", "安禄山", "史思明", "李白", "杜甫", "韩愈", "柳宗元", "唐朝"],
    "五代": ["朱温", "李克用", "李存勖", "石敬瑭", "刘知远", "郭威", "柴荣", "五代十国"],
    "北宋": ["宋太祖", "宋太宗", "宋仁宗", "赵匡胤", "赵光义", "范仲淹", "王安石", "苏轼", "欧阳修", "司马光", "靖康", "北宋"],
    "南宋": ["宋高宗", "赵构", "岳飞", "韩世忠", "秦桧", "辛弃疾", "陆游", "临安府", "南宋"],
    "辽": ["辽太祖", "耶律阿保机", "耶律德光", "契丹", "萧太后", "辽朝"],
    "金": ["金太祖", "完颜阿骨打", "完颜亮", "女真", "猛安谋克", "金朝"],
    "元": ["元世祖", "忽必烈", "成吉思汗", "铁木真", "窝阔台", "蒙哥", "马可波罗", "行省", "元朝"],
    "明": ["明太祖", "明成祖", "朱元璋", "朱棣", "洪武", "永乐", "嘉靖", "万历", "崇祯", "郑和", "张居正", "戚继光", "王阳明", "明朝"],
    "清": ["清太祖", "努尔哈赤", "皇太极", "顺治", "康熙", "雍正", "乾隆", "嘉庆", "道光", "咸丰", "同治", "光绪", "宣统", "慈禧", "曾国藩", "李鸿章", "清朝"],
}

# 弱信号朝代词：这些词在现代表述中高频出现（季节、成语、历史引用、三国故事），
# 单独命中不足以说明文档属于该朝代，计 0.5 权
_DYNASTY_WEAK_KEYWORDS = {"春秋", "战国", "三国", "诸侯"}

# 年号表（常用年号 → 所属朝代）
_ERA_NAMES: Dict[str, str] = {
    # 西汉
    "建元": "西汉", "元光": "西汉", "元朔": "西汉", "元狩": "西汉", "元鼎": "西汉",
    "元封": "西汉", "太初": "西汉", "天汉": "西汉", "太始": "西汉", "征和": "西汉",
    "始元": "西汉", "元凤": "西汉", "元平": "西汉", "本始": "西汉", "地节": "西汉",
    "元康": "西汉", "神爵": "西汉", "五凤": "西汉", "甘露": "西汉", "黄龙": "西汉",
    # 东汉
    "建武": "东汉", "中元": "东汉", "永平": "东汉", "建初": "东汉", "元和": "东汉",
    "章和": "东汉", "永元": "东汉", "元兴": "东汉", "延平": "东汉", "永初": "东汉",
    # 三国
    "黄初": "三国", "太和": "三国", "青龙": "三国", "景初": "三国",
    "章武": "三国", "建兴": "三国", "延熙": "三国", "景耀": "三国",
    "黄武": "三国", "黄龙": "三国", "嘉禾": "三国", "赤乌": "三国",
    # 晋
    "泰始": "西晋", "太康": "西晋", "太熙": "西晋",
    "建武": "东晋", "大兴": "东晋", "永昌": "东晋", "太宁": "东晋",
    # 南朝
    "永初": "南朝宋", "元嘉": "南朝宋", "大明": "南朝宋", "景和": "南朝宋",
    "建元": "南齐", "永明": "南齐",
    "天监": "南梁", "普通": "南梁", "大通": "南梁", "大同": "南梁",
    "永定": "南陈", "天嘉": "南陈", "太建": "南陈",
    # 北朝
    "皇始": "北魏", "太和": "北魏", "景明": "北魏", "正光": "北魏",
    "天保": "北齐", "河清": "北齐",
    "大统": "北周", "保定": "北周", "建德": "北周",
    # 隋
    "开皇": "隋", "仁寿": "隋", "大业": "隋",
    # 唐
    "武德": "唐", "贞观": "唐", "永徽": "唐", "显庆": "唐", "龙朔": "唐",
    "麟德": "唐", "乾封": "唐", "总章": "唐", "咸亨": "唐", "上元": "唐",
    "仪凤": "唐", "调露": "唐", "永隆": "唐", "开耀": "唐", "永淳": "唐",
    "弘道": "唐", "嗣圣": "唐", "神龙": "唐", "景龙": "唐", "景云": "唐",
    "先天": "唐", "开元": "唐", "天宝": "唐", "至德": "唐", "乾元": "唐",
    "宝应": "唐", "广德": "唐", "永泰": "唐", "大历": "唐", "建中": "唐",
    "兴元": "唐", "贞元": "唐", "永贞": "唐", "元和": "唐", "长庆": "唐",
    "宝历": "唐", "大和": "唐", "开成": "唐", "会昌": "唐", "大中": "唐",
    "咸通": "唐", "乾符": "唐", "广明": "唐", "中和": "唐", "光启": "唐",
    "文德": "唐", "龙纪": "唐", "大顺": "唐", "景福": "唐", "乾宁": "唐",
    "光化": "唐", "天复": "唐", "天祐": "唐",
    # 宋
    "建隆": "北宋", "乾德": "北宋", "开宝": "北宋", "太平兴国": "北宋", "雍熙": "北宋",
    "端拱": "北宋", "淳化": "北宋", "至道": "北宋", "咸平": "北宋", "景德": "北宋",
    "大中祥符": "北宋", "天禧": "北宋", "乾兴": "北宋", "天圣": "北宋", "明道": "北宋",
    "景祐": "北宋", "宝元": "北宋", "康定": "北宋", "庆历": "北宋", "皇祐": "北宋",
    "至和": "北宋", "嘉祐": "北宋", "治平": "北宋", "熙宁": "北宋", "元丰": "北宋",
    "元祐": "北宋", "绍圣": "北宋", "元符": "北宋", "建中靖国": "北宋", "崇宁": "北宋",
    "大观": "北宋", "政和": "北宋", "重和": "北宋", "宣和": "北宋", "靖康": "北宋",
    "建炎": "南宋", "绍兴": "南宋", "隆兴": "南宋", "乾道": "南宋", "淳熙": "南宋",
    "绍熙": "南宋", "庆元": "南宋", "嘉泰": "南宋", "开禧": "南宋", "嘉定": "南宋",
    "宝庆": "南宋", "绍定": "南宋", "端平": "南宋", "嘉熙": "南宋", "淳祐": "南宋",
    "宝祐": "南宋", "开庆": "南宋", "景定": "南宋", "咸淳": "南宋", "德祐": "南宋",
    # 辽
    "神册": "辽", "天赞": "辽", "天显": "辽", "会同": "辽", "大同": "辽",
    "天禄": "辽", "应历": "辽", "保宁": "辽", "乾亨": "辽", "统和": "辽",
    "开泰": "辽", "太平": "辽", "景福": "辽", "重熙": "辽", "清宁": "辽",
    # 金
    "收国": "金", "天辅": "金", "天会": "金", "天眷": "金", "皇统": "金",
    "天德": "金", "贞元": "金", "正隆": "金", "大定": "金", "明昌": "金",
    # 元
    "中统": "元", "至元": "元", "元贞": "元", "大德": "元", "至大": "元",
    "皇庆": "元", "延祐": "元", "至治": "元", "泰定": "元", "致和": "元",
    "天历": "元", "至顺": "元", "元统": "元", "至元": "元", "至正": "元",
    # 明
    "洪武": "明", "建文": "明", "永乐": "明", "洪熙": "明", "宣德": "明",
    "正统": "明", "景泰": "明", "天顺": "明", "成化": "明", "弘治": "明",
    "正德": "明", "嘉靖": "明", "隆庆": "明", "万历": "明", "泰昌": "明",
    "天启": "明", "崇祯": "明",
    # 清
    "天命": "清", "天聪": "清", "崇德": "清", "顺治": "清", "康熙": "清",
    "雍正": "清", "乾隆": "清", "嘉庆": "清", "道光": "清", "咸丰": "清",
    "同治": "清", "光绪": "清", "宣统": "清",
}

# ---------------- 主题领域词表 ----------------

# 主题词表（带权重）——三层设计：
# 第一层 核心主题词（权重 4-5）：真正决定文档主题的词，如"财政""科举""战争"
# 第二层 一般主题词（权重 2-3）：有主题倾向但区分度较低，如"税""商""兵"
# 第三层 低权词（权重 1）：单字通名（如"山""河"），会被 jieba ns 实体排除进一步降权
# 地点实体词（北京/上海/郎溪县）：权重 0，由 jieba ns 标注自动排除，不参与打分
_TOPIC_WEIGHTS: Dict[str, Dict[str, float]] = {
    "政治": {
        # 核心词
        "朝廷": 5, "皇帝": 5, "宰相": 5, "丞相": 5, "奏议": 5, "诏书": 5,
        "敕": 4, "诏": 4, "策": 4, "奏": 4, "疏": 4, "表": 4,
        "封": 3, "拜": 3, "除": 3, "迁": 3, "贬": 4, "黜": 4, "罢": 3, "诛": 3, "赦": 3,
        # 职官
        "丞相": 5, "尚书": 4, "仆射": 4, "侍中": 4, "刺史": 4, "太守": 4,
        "都督": 4, "节度使": 4, "总兵": 4, "参将": 4,
        # 一般词
        "帝": 3, "皇": 3, "后": 2, "太子": 3, "诸侯": 4,
        "令": 2, "丞": 2, "尉": 2, "卿": 2, "郎": 2,
        "朝": 2, "廷": 2, "宫": 2, "省": 2, "部": 2, "寺": 2, "监": 2, "院": 2, "司": 2, "局": 2,
    },
    "经济": {
        # 核心词
        "田赋": 5, "钱粮": 5, "厘金": 5, "两税法": 5, "租庸调": 5,
        "盐铁": 5, "酒榷": 5, "均输": 5, "平准": 5, "常平": 5, "和籴": 5, "和买": 5,
        "财政": 5, "税收": 5, "预算": 5, "审计": 5,
        # 一般词
        "税": 3, "租": 3, "课": 3, "役": 3, "货币": 4, "铜钱": 4, "宝钞": 4,
        "屯田": 4, "均田": 4, "井田": 4,
        "财政": 5, "财务": 4, "会计": 4, "经营": 3, "资产": 3, "负债": 3,
        "利润": 4, "成本": 3, "产值": 4, "企业": 3,
        # 低权词
        "粮": 2, "钱": 2, "帛": 2, "绢": 2, "布": 2,
        "商": 2, "市": 2, "贸": 2, "贾": 2,
        "银": 2, "金": 2, "亩": 1, "石": 1, "斛": 1, "贯": 1, "两": 1,
        "户口": 3, "垦田": 3,
    },
    "金融": {
        # 独立"金融"主题，避免被"经济"吞没
        "银行": 5, "贷款": 5, "存款": 5, "储蓄": 5, "信贷": 5, "支行": 5,
        "信用社": 5, "联社": 5, "营业部": 5, "分理处": 5,
        "利率": 4, "汇率": 4, "股票": 4, "债券": 4, "国债": 4, "利息": 4,
        "准备金": 4, "再贷款": 4, "抵押": 4, "贴现": 4, "汇票": 4, "结算": 4,
        "通存通兑": 5, "汇兑": 4, "联行": 4, "拆借": 4,
        "货币政策": 5, "信贷资金": 5, "存贷差": 4, "头寸": 4, "国库": 4, "现钞": 4,
        "发行库": 5, "现金调拨": 5, "票据": 4, "承兑": 4, "外汇": 4, "外币": 4,
        "金融": 5,
    },
    "军事": {
        # 核心词
        "战争": 5, "战役": 5, "驻防": 5, "兵制": 5, "都督": 4, "节度使": 4,
        "将军": 4, "总兵": 4, "参将": 4, "游击": 4,
        "粮草": 4, "辎重": 4,
        # 近现代战争（抗战/内战叙事高频词）
        "轰炸": 5, "空袭": 5, "抗日": 5, "抗战": 5,
        "游击队": 5, "八路军": 5, "新四军": 5,
        "扫荡": 4, "沦陷": 4, "起义": 4, "自卫队": 4,
        "击毙": 3, "缴获": 3, "部队": 3,
        # 一般词
        "兵": 2, "军": 2, "将": 2, "帅": 2, "校": 2, "尉": 2, "司马": 3,
        "战": 3, "伐": 3, "攻": 3, "守": 3, "围": 2, "降": 2, "败": 2, "克": 2,
        "阵": 2, "营": 2, "垒": 2, "寨": 2,
        "骑": 2, "步": 2, "弓": 2, "弩": 2, "矛": 2, "戟": 2, "甲": 2, "盾": 2, "铠": 2,
        "援": 2, "溃": 2, "遁": 2, "追": 2, "击": 2, "斩": 2, "俘": 2,
    },
    "文化": {
        # 核心词
        "科举": 5, "书院": 5, "太学": 5, "国子监": 5,
        "诗": 3, "文": 3, "赋": 3, "词": 3, "曲": 3, "经": 3, "史": 3,
        "礼": 3, "乐": 3, "射": 3, "御": 3, "数": 3, "艺": 3,
        "儒": 4, "墨": 4, "道": 4, "法": 4,
        "佛": 4, "禅": 4, "僧": 4, "寺": 3, "庙": 3, "观": 3, "庵": 3,
        "学": 3, "庠": 3, "序": 3,
        "祭": 3, "祀": 3, "祠": 3, "坛": 3,
        "画": 3, "琴": 3, "棋": 3, "书": 3, "印": 3,
    },
    "社会": {
        # 人口、民族、饥荒、风俗、福利
        "人口": 5, "民族": 5, "饥荒": 5, "风俗": 5, "福利": 5,
        "户口": 4, "赈灾": 5, "救济": 4, "灾荒": 5, "旱灾": 4, "水灾": 4,
        # 灾异叙事高频词（大事记/灾异志）
        "赈恤": 5, "灾民": 4, "难民": 4,
        "大旱": 4, "大水": 3, "蝗虫": 4, "瘟疫": 4, "地震": 4,
        # 民生/聚落（乡镇概况章节高频：人均收入、集镇、医疗、学校）
        "收入": 5, "卫生院": 5, "敬老院": 5, "中学": 4, "小学": 4,
        "集镇": 3, "医院": 3,
        "婚俗": 4, "丧葬": 4, "节庆": 4, "民俗": 5,
        "养老": 4, "抚恤": 4, "孤儿": 4, "寡妇": 4,
    },
    "法律": {
        # 核心词
        "法律": 5, "立法": 5, "司法": 5, "法院": 5, "检察": 5,
        "律例": 5, "律令": 5, "判牍": 5, "刑部": 5, "诉讼": 5, "判决": 5,
        "监狱": 5, "逮捕": 4, "审判": 4, "上诉": 4, "抗诉": 4,
        "公安": 4, "治安": 4, "案件": 4,
        "契约": 4, "纠纷": 4, "调解": 4,
        # 一般词（单字词降权：宗教文本中"罪业""地狱""戒律"高频，防误判）
        "刑": 3, "罚": 3,
        "罪": 2, "囚": 2, "狱": 2,
        # 注：单字"律"歧义过大（律宗/戒律/音律/律诗），已移除
    },
    "宗教": {
        # 佛教
        "佛教": 5, "寺院": 5, "僧侣": 5, "住持": 5, "方丈": 5, "法会": 5,
        "菩萨": 5, "观音": 5, "罗汉": 4, "佛陀": 5, "释迦牟尼": 5,
        "佛像": 5, "佛经": 5, "经书": 4, "香火": 4, "开光": 5, "功德": 4,
        "居士": 4, "出家": 4, "修行": 4, "禅宗": 5, "净土": 4,
        "戒律": 5, "律宗": 5, "忏悔": 4, "地狱": 4, "轮回": 4, "因果": 4,
        # 道教
        "道教": 5, "道观": 5, "道士": 5, "道长": 5, "宫观": 5, "神仙": 4, "符箓": 5,
        # 基督教/天主教/伊斯兰教
        "基督教": 5, "天主教": 5, "教堂": 5, "教会": 5, "牧师": 5, "神父": 5,
        "伊斯兰": 5, "清真寺": 5, "阿訇": 5, "穆斯林": 5,
        # 通用
        "宗教": 5, "信仰": 4, "信徒": 5, "教务": 5, "朝觐": 5, "庙会": 4,
        "佛": 3, "禅": 3, "僧": 3, "寺": 3, "庙": 3, "观": 2, "庵": 3, "尼": 3,
    },
    "地理": {
        # 只限自然地理：山脉、河流、气候、植被、灾害
        "山脉": 5, "河流": 5, "气候": 5, "植被": 5, "灾害": 5,
        "地形": 4, "地貌": 4, "水文": 5, "土壤": 4, "水系": 5,
        # 水文观测/水资源：防止水文类文档被"水利"带进农工建设
        "水位": 5, "流量": 5, "径流": 5, "流域": 5, "水资源": 5,
        "降水量": 4, "水质": 4, "河道": 4,
        "洪水": 4, "汛期": 4, "洪峰": 4, "枯水": 4,
        # 低权通名（会被实体排除进一步降权）
        "山": 1, "河": 1, "江": 1, "溪": 1, "湖": 1, "海": 1, "泽": 1, "池": 1, "泉": 1,
        # 注："水库"是人工水利设施，归农工建设，不在此列
    },
    "农工建设": {
        # 漕运、水利、屯垦、桑蚕
        "漕运": 5, "水利": 5, "屯垦": 5, "桑蚕": 5, "农业": 5, "工业": 5,
        "灌溉": 4, "农具": 4, "耕作": 4, "施肥": 4,
        # 种植业（农业章节高频词；权重适中，避免乡镇概况中附带的产量数据压过民生词）
        "农作物": 5, "水稻": 4, "单产": 4, "亩产": 4,
        "耕地": 3, "水田": 3, "旱地": 3, "播种": 3, "种植": 3,
        "茶园": 3, "棉花": 3, "油菜": 3, "小麦": 3, "粮食": 2, "总产": 2,
        # 水利工程设施（水库/塘坝为人工设施，与自然地理区分）
        "水库": 5, "塘坝": 5, "灌区": 5, "库容": 5, "溢洪道": 5,
        "大坝": 4, "蓄水": 4, "排灌": 4, "涵闸": 4,
        "手工业": 5, "纺织": 4, "陶瓷": 4, "冶铁": 4, "采矿": 4,
        "建筑": 4, "工程": 4, "施工": 4,
        "公路": 4, "桥梁": 4, "铁路": 5, "电力": 5, "电信": 5,
    },
}

# 主题得分上限（防止长文中高频通名压倒真实主题）
# 设为 60：对数压缩后增长缓慢，达到此分数说明主题词确实密集；
# 过低会导致多个主题同时触顶同分（如水利工程章节地理/农工建设同分 tie）
_TOPIC_SCORE_CAP = 100.0

# ---------------- 现代时期词表 ----------------
# 用于无朝代概念的近现代文本（学术论文、方志现代部分、年鉴等）
# 每个时期附带对应年代范围，前端展示如"民国(1910s-1940s)"
_MODERN_ERA_KEYWORDS: Dict[str, Dict[str, Any]] = {
    "清末": {
        "decades": ["1900s", "1910s"],
        "decade_range": "1900s-1910s",
        "keywords": ["清末", "晚清", "光绪末", "宣统", "辛丑条约", "日俄战争"],
    },
    "民国": {
        "decades": ["1910s", "1920s", "1930s", "1940s"],
        "decade_range": "1910s-1940s",
        "keywords": ["民国", "北洋", "国民政府", "抗日战争", "解放前", "辛亥革命",
                     "袁世凯", "段祺瑞", "蒋介石", "汪精卫", "重庆谈判"],
    },
    "新中国初期": {
        "decades": ["1950s", "1960s", "1970s"],
        "decade_range": "1950s-1970s",
        "keywords": ["新中国成立", "解放后", "建国初期", "土改", "三大改造",
                     "人民公社", "大跃进", "文化大革命", "知青", "上山下乡",
                     "抗美援朝", "中苏交恶",
                     # 现代政权/组织词：1949年后文本的强信号（县志/年鉴几乎必见）
                     "人民政府", "党委", "党支部", "自治区", "人民代表"],
    },
    "改革开放": {
        "decades": ["1980s", "1990s", "2000s", "2010s", "2020s"],
        "decade_range": "1980s-",
        "keywords": ["改革开放", "十一届三中全会", "现代化建设", "市场经济",
                     "邓小平", "南方谈话", "特区", "招商引资", "入世", "WTO",
                     "互联网", "城镇化", "科学发展观", "新时代",
                     # 现代经济/金融词：让现代方志的金融章节被识别为现代文本
                     "金融机构", "商业银行", "支行", "营业部", "分理处",
                     "金融监管", "金融改革", "金融生态", "货币政策", "信贷政策",
                     "存贷款", "再贷款", "支农再贷款", "专项贷款", "住房贷款",
                     "票据承兑", "通存通兑", "电子汇兑", "支付清算",
                     "办公自动化", "电子化", "电算化"],
    },
}

# 公历年份正则（4位数字，1900-2099）
_YEAR_RE = re.compile(r'(?:19|20)\d{2}')

# "朝代名+年号"复合模式（如"清康熙年间""明洪武初"）
# 古籍记本朝事不加朝代前缀（只写"康熙年间"），加前缀是后代文本
# （近现代方志/史书）叙述历史的标志，可作为"本文并非古籍"的强信号
_HIST_REF_PREFIXES = ("清", "明", "元", "宋", "唐", "隋", "汉", "秦", "魏", "晋", "辽", "金", "周", "夏", "商")
# 年号按长度降序，保证"太平兴国"先于"太平"匹配
_ERA_NAMES_SORTED = sorted(_ERA_NAMES.keys(), key=len, reverse=True)
_HIST_REF_RE = re.compile(
    r'(?:' + '|'.join(_HIST_REF_PREFIXES) + r')(?:' + '|'.join(map(re.escape, _ERA_NAMES_SORTED)) + r')'
)


# ---------------- 主导标签判定 ----------------
# "不确定就不贴标签"：分数过低或前两名难分时返回 None，避免强行归类
#
# min_score      : 最低分阈值（每千词分数低于此值视为偶发提及，不判定）
# ambiguity_ratio: 歧义比例。第二名分数 ≥ 第一名 × ratio 时视为难分
# min_clear_score: 只有第一名分数达到此值时，才允许在难分情况下仍判定
def pick_dominant(scores: Dict[str, float],
                  min_score: float = 1.0,
                  ambiguity_ratio: float = 0.6,
                  min_clear_score: float = 5.0) -> Optional[str]:
    """从分数表中取主导标签；信号弱或歧义时返回 None。"""
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_score = ranked[0]
    if top_score < min_score:
        return None
    # 前两名接近且第一名未形成压倒性优势 → 信号模糊，不强行贴标签
    if len(ranked) >= 2:
        second_score = ranked[1][1]
        if second_score >= top_score * ambiguity_ratio and top_score < min_clear_score:
            return None
    return top_name


def _classify_year_to_decade(year: int) -> Optional[str]:
    """将公历年份归类到年代，如 2009 → '2000s'。"""
    if 1900 <= year <= 1999:
        return f"{year // 10 * 10}s"
    if 2000 <= year <= 2099:
        return f"{year // 10 * 10}s"
    return None


# ---------------- 实体密度词性映射 ----------------
# jieba 词性标注 → 实体类型
_POS_ENTITY_MAP = {
    "nr": "person",   # 人名
    "ns": "place",    # 地名
    "nt": "org",      # 机构名
    "nz": "other",    # 其他专名
}

# ---------------- 人名识别黑名单 ----------------
# jieba 词性标注会把一些地名/机构名误判为人名（nr），如"郎溪县""南京分行"
# 这里做后置过滤：以下后缀结尾的实体不可能是人名
_PERSON_NAME_SUFFIX_BLACKLIST = (
    "县", "市", "镇", "乡", "村", "区", "州", "省", "街", "路", "巷",
    "所", "室", "科", "股", "组", "办", "局", "行", "社", "队", "部",
    "会", "中心", "公司", "厂", "站", "院", "校", "团", "军", "师",
    "联", "盟", "派", "帮", "教", "党", "府", "署",
)
# 已知高频误判词（精确匹配）
_PERSON_NAME_EXACT_BLACKLIST = {
    "郎溪", "郎溪县", "县委", "县政府", "县人民政府", "南京", "北京", "上海",
    "天津", "重庆", "纪检监察", "宣传周", "金元宝", "凌管", "凌曾",
    # 抽象概念/动作词/术语词（被 jieba 误判为 nr）
    "文明", "党风廉政", "廉政", "青少年", "盖章", "申请人", "许可证",
    "黑名单", "籍贯", "荣获", "陈述", "纪律检查", "金穗卡",
    "改革开放", "现代化", "科学发展观", "新时代",
    # 更多 jieba 误判词
    "郎政", "建立健全", "阳光", "道德", "康复", "宣传教育",
    "幼儿园", "南漪湖", "陈列", "纪律检查",
}

# ---------------- jieba 可选加载 ----------------

_jieba = None
_jieba_posseg = None
_jieba_loaded = False

def _ensure_jieba():
    """延迟加载 jieba，未安装时返回 None。"""
    global _jieba, _jieba_posseg, _jieba_loaded
    if _jieba_loaded:
        return
    _jieba_loaded = True
    try:
        import jieba
        import jieba.posseg as posseg
        _jieba = jieba
        _jieba_posseg = posseg
        # 静默初始化
        _jieba.setLogLevel(60)  # WARNING+ 不输出
    except ImportError:
        _jieba = None
        _jieba_posseg = None


# ---------------- 核心统计函数 ----------------

def compute_chunk_stats(text: str) -> Dict[str, Any]:
    """对单个大 chunk 文本做分词统计，返回统计特征。

    返回结构：
    {
        "dynasty_scores": {朝代名: 分数},   # 朝代词命中频率（归一化到每千词）
        "era_names": [识别到的年号],
        "era_dynasty": 年号归属朝代（若有）,
        "topic_scores": {主题: 分数},       # 主题领域词命中频率
        "entity_density": {实体类型: 密度},  # 实体词数/总词数
        "word_count": 总词数,
    }
    """
    if not text or len(text) < 50:
        return _empty_stats()

    _ensure_jieba()

    # 分词
    if _jieba is not None:
        words = list(_jieba.cut(text))
    else:
        # 无 jieba 时用字符级 bigram 近似（精度低但能跑）
        words = [text[i:i+2] for i in range(0, len(text) - 1, 2)]

    word_count = len(words)
    if word_count == 0:
        return _empty_stats()

    # 构建分词结果的词频 Counter（用于精确匹配关键词，避免子串误命中）
    word_counter = Counter(words)
    # "农业"词频改用文本级计数并排除"非农业"（否定用法）：
    # bigram 模式下"非农业人口"按对齐位置约半数切出"农业"，词频计数不可靠
    _freq_override = {
        "农业": max(0, text.count("农业") - text.count("非农业")),
    }

    # 1. 朝代关键词频率（基于分词结果匹配，避免子串误命中）
    text_lower = text  # 中文不区分大小写
    dynasty_scores: Dict[str, float] = {}
    for dynasty, keywords in _DYNASTY_KEYWORDS.items():
        hits = 0.0
        for kw in keywords:
            # 优先用分词结果精确匹配；多字词用子串匹配作为补充
            if len(kw) >= 2:
                cnt = word_counter.get(kw, 0)
            else:
                cnt = text_lower.count(kw)
            # 弱信号词降权（"春秋""战国"等在现代表述中高频出现）
            if kw in _DYNASTY_WEAK_KEYWORDS:
                cnt *= 0.5
            hits += cnt
        if hits > 0:
            # 归一化到每千词
            dynasty_scores[dynasty] = round(hits * 1000.0 / word_count, 2)

    # 2. 年号识别
    # 要求年号后跟"年"或"间"字才算年号用法（如"广德二年""绍兴年间"）
    # 避免将地名/人名误识别为年号（如"广德县""大兴村""太平军"）
    era_names_found = []
    era_dynasty = None
    for era, dynasty in _ERA_NAMES.items():
        if (era + '年') in text_lower or (era + '间') in text_lower:
            era_names_found.append(era)
            if era_dynasty is None:
                era_dynasty = dynasty
    # 年号去重，最多保留5个
    era_names_found = list(dict.fromkeys(era_names_found))[:5]

    # 3. 实体密度（需 jieba 词性标注）+ 人名频率统计 + 实体集合收集
    # 实体识别放在主题打分之前，以便在主题打分时排除地名/机构名实体
    entity_density: Dict[str, float] = {}
    person_counter: Counter = Counter()
    place_entity_set: set = set()  # 地名/机构名实体集合，用于主题打分时排除
    if _jieba_posseg is not None:
        pos_counts: Dict[str, int] = {}
        for w, flag in _jieba_posseg.cut(text):
            if len(w) < 2:
                continue
            entity_type = _POS_ENTITY_MAP.get(flag)
            if entity_type:
                pos_counts[entity_type] = pos_counts.get(entity_type, 0) + 1
                # 收集地名/机构名实体，用于主题打分时排除
                if entity_type in ("place", "org"):
                    place_entity_set.add(w)
                # 人名单独计数，用于 top10 展示
                # 应用黑名单过滤：jieba 会把地名/机构名误判为人名（如"郎溪县"）
                if entity_type == "person":
                    if w in _PERSON_NAME_EXACT_BLACKLIST:
                        continue
                    if w.endswith(_PERSON_NAME_SUFFIX_BLACKLIST):
                        continue
                    person_counter[w] += 1
        for etype, cnt in pos_counts.items():
            entity_density[etype] = round(cnt / word_count, 4)
    # top10 人名（按出现次数降序）
    top_persons = [{"name": n, "count": c} for n, c in person_counter.most_common(10)]

    # 4. 主题领域打分（权重 + 去实体 + 对数压缩 + 饱和上限）
    # 设计要点：
    # a) 三层词库：核心主题词(权重4-5) > 一般词(2-3) > 低权通名(1)
    # b) 去实体：地名(ns)/机构名(nt)实体不参与主题打分，避免"郎溪县"被当成"地理"
    # c) 对数压缩：log2(1+频次) × 权重，防止长文中高频通名压倒真实主题
    # d) 饱和上限：每主题得分上限 _TOPIC_SCORE_CAP，防止"河流"出现50次覆盖真实主题
    # e) 多标签：输出所有非零主题，由 summarize 阶段取 top-3
    topic_scores: Dict[str, float] = {}
    for topic, kw_weights in _TOPIC_WEIGHTS.items():
        score = 0.0
        for kw, weight in kw_weights.items():
            freq = _freq_override.get(kw) if kw in _freq_override else word_counter.get(kw, 0)
            if freq <= 0:
                continue
            # 去实体：如果该词被 jieba 识别为地名/机构名实体，跳过
            if kw in place_entity_set:
                continue
            # 对数压缩：log2(1+频次) × 权重
            score += math.log2(1 + freq) * weight
        if score > 0:
            # 饱和上限
            score = min(score, _TOPIC_SCORE_CAP)
            # 归一化到每千词分数（保留2位小数）
            topic_scores[topic] = round(score * 1000.0 / word_count, 2)

    # 5. 现代时期关键词频率 + 年代提取
    # 用于无朝代概念的近现代文本（学术论文、年鉴等）
    modern_era_scores: Dict[str, float] = {}
    modern_era_decade_range: Dict[str, str] = {}  # 时期 → 年代范围
    for era_name, era_info in _MODERN_ERA_KEYWORDS.items():
        hits = 0
        for kw in era_info["keywords"]:
            if len(kw) >= 2:
                hits += word_counter.get(kw, 0)
            else:
                hits += text_lower.count(kw)
        if hits > 0:
            modern_era_scores[era_name] = round(hits * 1000.0 / word_count, 2)
            modern_era_decade_range[era_name] = era_info["decade_range"]

    # 公历年份提取 → 年代分布
    year_matches = _YEAR_RE.findall(text_lower)
    decade_counter: Counter = Counter()
    for ystr in year_matches:
        try:
            y = int(ystr)
            decade = _classify_year_to_decade(y)
            if decade:
                decade_counter[decade] += 1
        except ValueError:
            continue
    # 年代分数（归一化到每千词）
    decade_scores: Dict[str, float] = {
        d: round(c * 1000.0 / word_count, 2)
        for d, c in decade_counter.items() if c > 0
    }
    # 主年代（出现次数最多的）
    dominant_decade = decade_counter.most_common(1)[0][0] if decade_counter else None

    # 6. "朝代名+年号"后设叙述命中次数（如"清康熙""明洪武"），用于现代文本仲裁
    hist_ref_hits = len(_HIST_REF_RE.findall(text))

    return {
        "dynasty_scores": dynasty_scores,
        "era_names": era_names_found,
        "era_dynasty": era_dynasty,
        "topic_scores": topic_scores,
        "entity_density": entity_density,
        "top_persons": top_persons,
        "modern_era_scores": modern_era_scores,
        "modern_era_decade_range": modern_era_decade_range,
        "decade_scores": decade_scores,
        "dominant_decade": dominant_decade,
        "hist_ref_hits": hist_ref_hits,
        "word_count": word_count,
    }


def _empty_stats() -> Dict[str, Any]:
    return {
        "dynasty_scores": {},
        "era_names": [],
        "era_dynasty": None,
        "topic_scores": {},
        "entity_density": {},
        "top_persons": [],
        "modern_era_scores": {},
        "modern_era_decade_range": {},
        "decade_scores": {},
        "dominant_decade": None,
        "hist_ref_hits": 0,
        "word_count": 0,
    }


# ---------------- 窗口滑动平均 ----------------

def sliding_window_average(stats_list: List[Dict[str, Any]], window: int = 1) -> List[Dict[str, Any]]:
    """对同文档的多个 chunk 统计值做窗口滑动平均。

    stats_list: 同一文档按顺序排列的 chunk 统计值列表
    window: 滑动窗口半径（k=1 表示取前后各1个 chunk 做平均，即3个chunk的窗口）
    返回平滑后的统计值列表（长度与输入相同）

    平滑策略：
    - dynasty_scores / topic_scores / entity_density：取窗口内非零值的平均
    - era_names / era_dynasty：取窗口内出现频率最高的（多数投票）
    - word_count：取窗口内平均
    """
    if not stats_list:
        return []
    n = len(stats_list)
    if n == 1 or window <= 0:
        return list(stats_list)

    result = []
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        window_stats = stats_list[lo:hi]

        # 数值型字段平滑
        smoothed = _average_numeric_fields(window_stats)
        # 年号多数投票
        era_names, era_dynasty = _vote_era(window_stats)
        smoothed["era_names"] = era_names
        smoothed["era_dynasty"] = era_dynasty
        # word_count 取平均
        smoothed["word_count"] = int(sum(s.get("word_count", 0) for s in window_stats) / len(window_stats))
        # 保留 top_persons（窗口内合并计数后取 top10）
        person_counter: Counter = Counter()
        for s in window_stats:
            for p in s.get("top_persons", []):
                person_counter[p.get("name", "")] += p.get("count", 0)
        smoothed["top_persons"] = [{"name": n, "count": c} for n, c in person_counter.most_common(10)]
        # 保留 modern_era_decade_range（取窗口内第一个非空值）
        for s in window_stats:
            ranges = s.get("modern_era_decade_range")
            if ranges:
                smoothed["modern_era_decade_range"] = ranges
                break
        else:
            smoothed["modern_era_decade_range"] = {}
        # 重新计算 dominant_decade
        decade_scores = smoothed.get("decade_scores", {})
        smoothed["dominant_decade"] = max(decade_scores, key=decade_scores.get) if decade_scores else None
        result.append(smoothed)
    return result


def _average_numeric_fields(window_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对 dynasty_scores / topic_scores / entity_density / modern_era_scores / decade_scores 做窗口平均。"""
    result: Dict[str, Any] = {
        "dynasty_scores": {},
        "topic_scores": {},
        "entity_density": {},
        "modern_era_scores": {},
        "decade_scores": {},
    }
    # dynasty_scores / topic_scores / modern_era_scores / decade_scores 用每千词分数（2位小数）
    # entity_density 是密度（4位小数）
    for field, ndigits in (("dynasty_scores", 2), ("topic_scores", 2),
                           ("entity_density", 4),
                           ("modern_era_scores", 2), ("decade_scores", 2)):
        all_keys = set()
        for s in window_stats:
            all_keys.update(s.get(field, {}).keys())
        for key in all_keys:
            vals = [s.get(field, {}).get(key, 0) for s in window_stats]
            nonzero = [v for v in vals if v > 0]
            if nonzero:
                result[field][key] = round(sum(nonzero) / len(nonzero), ndigits)
    return result


def _vote_era(window_stats: List[Dict[str, Any]]) -> tuple:
    """对窗口内的年号做多数投票，返回 (era_names, era_dynasty)。"""
    era_counter: Counter = Counter()
    dynasty_counter: Counter = Counter()
    for s in window_stats:
        for era in s.get("era_names", []):
            era_counter[era] += 1
        ed = s.get("era_dynasty")
        if ed:
            dynasty_counter[ed] += 1
    # 取频率最高的年号（窗口内至少2票才采纳，避免单chunk噪声）
    era_names = [era for era, cnt in era_counter.most_common(5) if cnt >= 2]
    era_dynasty = dynasty_counter.most_common(1)[0][0] if dynasty_counter else None
    return era_names, era_dynasty


# ---------------- 文档级汇总 ----------------

def summarize_document_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总同一文档所有 chunk 的统计值，生成文档级元数据。

    返回文档级特征（用于展示和路由）：
    {
        "dominant_dynasty": 主朝代（分数最高的）,
        "dynasty_scores": {朝代: 平均分数},
        "era_names": 文档中出现的年号列表,
        "era_dynasty": 年号归属朝代,
        "dominant_topic": 主主题,
        "topic_scores": {主题: 平均分数},
        "entity_density": {类型: 平均密度},
        "top_persons": [{name, count}],  # 文档级 top10 人名
        "time_span": "跨时代" 或 None,    # 显著时代 ≥ 3 时标记，单值不引入数组
        "total_words": 总词数,
        "chunk_count": chunk数,
    }
    """
    if not stats_list:
        return {
            "dominant_dynasty": None,
            "dynasty_scores": {},
            "era_names": [],
            "era_dynasty": None,
            "dominant_topic": None,
            "topic_scores": {},
            "entity_density": {},
            "top_persons": [],
            "modern_era_scores": {},
            "dominant_modern_era": None,
            "modern_era_decade_range": None,
            "decade_scores": {},
            "dominant_decade": None,
            "time_span": None,
            "total_words": 0,
            "chunk_count": 0,
        }

    # 先做滑动平均
    smoothed = sliding_window_average(stats_list, window=1)

    # 汇总
    dynasty_scores: Dict[str, float] = {}
    topic_scores: Dict[str, float] = {}
    entity_density: Dict[str, float] = {}
    era_names_counter: Counter = Counter()
    era_dynasty_votes: Counter = Counter()
    person_counter: Counter = Counter()
    modern_era_scores: Dict[str, float] = {}
    decade_counter: Counter = Counter()
    total_words = 0

    for s in smoothed:
        for k, v in s.get("dynasty_scores", {}).items():
            dynasty_scores[k] = dynasty_scores.get(k, 0) + v
        for k, v in s.get("topic_scores", {}).items():
            topic_scores[k] = topic_scores.get(k, 0) + v
        for k, v in s.get("entity_density", {}).items():
            entity_density[k] = entity_density.get(k, 0) + v
        era_names_counter.update(s.get("era_names", []))
        ed = s.get("era_dynasty")
        if ed:
            era_dynasty_votes[ed] += 1
        # 合并 chunk 级 top_persons 计数到文档级
        for p in s.get("top_persons", []):
            person_counter[p.get("name", "")] += p.get("count", 0)
        # 合并现代时期分数
        for k, v in s.get("modern_era_scores", {}).items():
            modern_era_scores[k] = modern_era_scores.get(k, 0) + v
        # 合并年代计数
        for k, v in s.get("decade_scores", {}).items():
            # v 是每千词分数，反推出现次数近似累加
            decade_counter[k] += v
        total_words += s.get("word_count", 0)

    # 取平均
    n = len(smoothed)
    dynasty_scores = {k: round(v / n, 2) for k, v in dynasty_scores.items() if v > 0}
    topic_scores = {k: round(v / n, 2) for k, v in topic_scores.items() if v > 0}
    entity_density = {k: round(v / n, 4) for k, v in entity_density.items() if v > 0}
    modern_era_scores = {k: round(v / n, 2) for k, v in modern_era_scores.items() if v > 0}
    decade_scores = {k: round(v / n, 2) for k, v in decade_counter.items() if v > 0}

    # 主朝代/主主题：信号弱（分数低于阈值）或前两名难分时不强行判定
    dominant_dynasty = pick_dominant(dynasty_scores)
    dominant_topic = pick_dominant(topic_scores)
    era_dynasty = era_dynasty_votes.most_common(1)[0][0] if era_dynasty_votes else None
    # 年号按出现 chunk 数降序排序（出现越多越具代表性）
    era_names = [e for e, _ in era_names_counter.most_common(5)]

    # ---- 跨时代判定 ----
    # 县志/方志/年鉴多为通史性文档，往往纵贯多个时代（先秦→清→民国→新中国），
    # 归入其中任何一个朝代都是误判。显著时代（朝代 + 现代）≥ 3 个时，
    # 单独标记为"跨时代"档位（单值字段，不引入数组），并清空朝代判定。
    modern_decade_total = sum(
        decade_scores.get(d, 0)
        for d in ("1900s", "1910s", "1920s", "1930s", "1940s", "1950s",
                  "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s")
    )
    significant_dynasties: List[str] = []
    if dynasty_scores:
        max_ds = max(dynasty_scores.values())
        # 显著朝代：分数 ≥ 1.0（每千词1次）且不低于最高分的 20%（防长尾噪声）
        significant_dynasties = [
            d for d, s in dynasty_scores.items() if s >= 1.0 and s >= max_ds * 0.2
        ]
    era_count = len(significant_dynasties)
    if modern_era_scores or modern_decade_total > 0:
        era_count += 1  # 现代信号整体计为 1 个时代
    time_span = "跨时代" if era_count >= 3 else None
    if time_span:
        dominant_dynasty = None
        era_names = []
        era_dynasty = None

    # ---- 时代冲突仲裁 ----
    # 语料以现代县志/方志/年鉴为主，此类文档引用"清康熙年间"等历史叙述是常态，
    # 不能因出现朝代词就判为古代。真正的古籍（二十四史等）有三个判别特征：
    #   1) 不会出现 1900 年以后的公历年份；
    #   2) 不会出现"民国""人民政府""改革开放"等现代特有词汇；
    #   3) 不会用"朝代名+年号"（如"清康熙"）这种后设叙述格式——古籍记本朝事
    #      只写"康熙年间"，加朝代前缀是后代文本叙述历史的标志。
    # 此外，现代方志常在年号后括注公元年份（如"康熙九年(1670)"），会触发年份信号。
    dominant_modern_era = pick_dominant(modern_era_scores)
    if dominant_dynasty:
        dynasty_score = dynasty_scores.get(dominant_dynasty, 0)
        modern_score = modern_era_scores.get(dominant_modern_era, 0) if dominant_modern_era else 0
        hist_ref_total = sum(s.get("hist_ref_hits", 0) for s in stats_list)
        is_modern = False
        # 1. 现代时期关键词命中即为强信号（这些词在古籍中不可能出现）
        if modern_score >= max(2.0, dynasty_score * 0.5):
            is_modern = True
        # 2. 出现现代公历年份，且朝代词密度未形成压倒性优势
        elif modern_decade_total > 0 and dynasty_score < max(3.0, modern_decade_total * 2):
            is_modern = True
        # 3. "朝代名+年号"后设叙述（"清康熙""明洪武"）出现2次以上
        elif hist_ref_total >= 2:
            is_modern = True
        # 4. 无任何现代信号时，朝代词密度须达最低阈值才认定朝代；
        #    低密度视为现代文本中的历史引用（县志建置沿革章节常见）
        elif dynasty_score < 2.0:
            is_modern = True
        if is_modern:
            dominant_dynasty = None
            era_names = []
            era_dynasty = None

    # 年号必须依附于朝代：没有朝代就没有年号
    # 学术论文/技术文档中"中和""普通"等年号多为常用词误识别
    if not dominant_dynasty:
        era_names = []
        era_dynasty = None
    # 文档级 top10 人名（按总出现次数降序）
    top_persons = [{"name": name, "count": cnt} for name, cnt in person_counter.most_common(10)]
    # 主现代时期 + 对应年代范围（dominant_modern_era 已在冲突仲裁中计算）
    modern_era_decade_range = None
    if dominant_modern_era:
        # 从原始 stats_list 中查找该时期的年代范围
        for s in stats_list:
            ranges = s.get("modern_era_decade_range") or {}
            if dominant_modern_era in ranges:
                modern_era_decade_range = ranges[dominant_modern_era]
                break
    # 主年代（分数最高的）
    dominant_decade = max(decade_scores, key=decade_scores.get) if decade_scores else None

    return {
        "dominant_dynasty": dominant_dynasty,
        "dynasty_scores": dynasty_scores,
        "era_names": era_names,
        "era_dynasty": era_dynasty,
        "dominant_topic": dominant_topic,
        "topic_scores": topic_scores,
        "entity_density": entity_density,
        "top_persons": top_persons,
        "modern_era_scores": modern_era_scores,
        "dominant_modern_era": dominant_modern_era,
        "modern_era_decade_range": modern_era_decade_range,
        "decade_scores": decade_scores,
        "dominant_decade": dominant_decade,
        "time_span": time_span,
        "total_words": total_words,
        "chunk_count": n,
    }
