"""共享的数据模型和工具函数。"""

import json
import logging
import re

from pydantic import BaseModel

logger = logging.getLogger("planning_server.helpers")

# 自定义 VERBOSE 级别 (低于 DEBUG)
VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")


# ── Request / Response Models ───────────────────────────────────

class RetrievalSetting(BaseModel):
    top_k: int = 1
    score_threshold: float = 0
    search_mode: str = "hybrid"
    search_strategy: str = "precise"


class SubQueryItem(BaseModel):
    sub_query: str
    tool_use: str
    topk: int
    domain: list[str] | None = None


class SubDomainItem(BaseModel):
    scene: str
    desc: str = ""
    info: list[dict] = []


class DomainItem(BaseModel):
    domain: str
    desc: str = ""
    rdf_list: list[SubDomainItem] = []


class PlanningRequest(BaseModel):
    query: str
    retrieval_setting: RetrievalSetting | None = None
    turn: int
    max_turn: int = 3
    max_top_k: int = 3
    max_context_size: int = 3
    tool_hub: str = "es,graph,web"
    tool_hub_optional: str = ""
    history: dict | None = None
    thinking: str = "simple"  # "simple", "dynamic", "deep"
    tool_select_enable: bool = False
    tool_selection_mode: str = "keyword"  # "rule", "model", or "keyword"
    tool_selection_threshold: float | None = 0.4
    domains: list[DomainItem] | None = None


class PlanningResponse(BaseModel):
    is_off_topic: bool = False
    status: str = "running"
    turn: int = 1
    current: list[SubQueryItem] | None = None
    history: dict | None = None
    final: dict | None = None
    plan: list[dict] | None = None
    step_summary: str | None = None
    answer: str | None = None
    reference_tree: list | None = None


# ── JSON 解析 ───────────────────────────────────────────────────

def _parse_llm_json(content: str) -> dict | None:
    """从 LLM 输出中解析 JSON，支持 markdown 包裹和 think 块。"""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.rstrip()).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ── RDF 实体表提取与关键词匹配 ──────────────────────────────────

import jieba

_OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
_OWL_DATATYPE_PROP = "http://www.w3.org/2002/07/owl#DatatypeProperty"
_OWL_OBJECT_PROP = "http://www.w3.org/2002/07/owl#ObjectProperty"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_RDFS_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"
_RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
_RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"

# 中文停用词表
STOPWORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "他", "她", "它", "们", "那", "些", "什么", "怎么", "如何", "为什么",
    "哪", "哪个", "哪些", "多少", "几", "吗", "呢", "吧", "啊", "呀", "嗯",
    "可以", "能", "能够", "应该", "需要", "想", "想要", "请", "请问",
    "这个", "那个", "这些", "那些", "还是", "或者", "以及", "并且", "但是", "然而",
    "因为", "所以", "如果", "虽然", "不过", "而且", "或", "与", "及",
    "从", "对", "把", "被", "让", "给", "向", "跟", "比", "按", "用",
    "之", "其", "该", "此", "每", "各", "某", "另", "其他",
    "已经", "正在", "将", "将要", "曾经", "一直", "还", "又", "再",
    "非常", "特别", "比较", "更", "最", "太", "真", "挺",
    "分别", "一下", "一些", "关于", "通过", "进行", "属于",
}


def _get_rdf_value(node: dict, predicate: str) -> str:
    """从 RDF JSON-LD 节点中提取指定谓词的第一个值。"""
    values = node.get(predicate, [])
    if not values:
        return ""
    v = values[0]
    return v.get("@value", v.get("@id", ""))


def _get_rdf_domains(node: dict) -> list[str]:
    """从 RDF JSON-LD 节点中提取 domain 列表（class URI）。"""
    values = node.get(_RDFS_DOMAIN, [])
    return [v.get("@id", "") for v in values if v.get("@id")]


def _short_name(uri: str) -> str:
    """从 URI 中提取最后一段作为短名。"""
    if "/" in uri:
        return uri.rsplit("/", 1)[-1]
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri


def _range_to_type(range_uri: str) -> str:
    """将 XSD/OWL range URI 转为可读类型名。"""
    if not range_uri:
        return "unknown"
    short = _short_name(range_uri)
    xsd_map = {"string": "string", "decimal": "decimal", "integer": "integer",
               "float": "float", "double": "double", "boolean": "boolean",
               "dateTime": "dateTime", "date": "date"}
    if short in xsd_map:
        return xsd_map[short]
    return f"-> {short}"


def extract_entity_tables(info: list[dict]) -> dict[str, list[dict]]:
    """从 RDF JSON-LD info 列表中提取实体表。

    返回: {实体中文名: [{"property_name": ..., "label": ..., "comment": ..., "type": ...}, ...]}
    """
    classes: dict[str, str] = {}
    for node in info:
        types = node.get("@type", [])
        if _OWL_CLASS in types:
            uri = node.get("@id", "")
            label = _get_rdf_value(node, _RDFS_LABEL)
            if uri:
                classes[uri] = label or _short_name(uri)

    entity_tables: dict[str, list[dict]] = {label: [] for label in classes.values()}

    for node in info:
        types = node.get("@type", [])
        is_prop = _OWL_DATATYPE_PROP in types or _OWL_OBJECT_PROP in types
        if not is_prop:
            continue

        prop_uri = node.get("@id", "")
        prop_name = _short_name(prop_uri)
        label = _get_rdf_value(node, _RDFS_LABEL)
        comment = _get_rdf_value(node, _RDFS_COMMENT)
        range_uri = _get_rdf_value(node, _RDFS_RANGE)
        prop_type = _range_to_type(range_uri)
        domains = _get_rdf_domains(node)

        attr_entry = {
            "property_name": prop_name,
            "label": label or prop_name,
            "comment": comment,
            "type": prop_type,
        }

        for domain_uri in domains:
            class_label = classes.get(domain_uri)
            if class_label:
                entity_tables[class_label].append(attr_entry)

    return {k: v for k, v in entity_tables.items() if v}


def build_keywords_from_entity_tables(entity_tables: dict[str, list[dict]]) -> set[str]:
    """从实体表中提取关键词集合。"""
    keywords: set[str] = set()
    for entity_name, attrs in entity_tables.items():
        keywords.add(entity_name)
        if "（" in entity_name:
            parts = entity_name.replace("）", "").split("（")
            for p in parts:
                p = p.strip()
                if p:
                    keywords.add(p)

        for attr in attrs:
            attr_label = attr["label"]
            if attr_label:
                keywords.add(attr_label)
                cleaned = re.sub(r'[\(（].*?[\)）]', '', attr_label).strip()
                cleaned = re.sub(r'(mm|m2|kg|g/㎡|元|/h)$', '', cleaned).strip()
                if cleaned and cleaned != attr_label:
                    keywords.add(cleaned)

    return keywords


def score_query_against_keywords(query: str, keywords: set[str]) -> tuple[float, int, int, list[str]]:
    """对 query 用 jieba 分词，去停用词后检查每个词是否被某个关键词包含（分词 ⊂ 关键词）。

    每个 token 最多命中一个关键词（避免同一个 token 重复计数）。

    返回: (命中率, 命中数, 有效分词数, 命中的关键词列表)
    命中率 = 命中数 / 有效分词数（归一化到 query 长度）
    """
    tokens = jieba.lcut(query)
    filtered_tokens = {t for t in tokens if len(t) >= 2 and t not in STOPWORDS}
    token_count = len(filtered_tokens)

    hits: list[str] = []
    matched_keywords: set[str] = set()
    for token in filtered_tokens:
        for kw in keywords:
            if len(kw) < 2 or kw in matched_keywords:
                continue
            if token in kw:
                hits.append(kw)
                matched_keywords.add(kw)
                break  # 一个 token 只算一次命中

    hit_count = len(hits)
    hit_ratio = hit_count / token_count if token_count > 0 else 0.0
    return hit_ratio, hit_count, token_count, hits
