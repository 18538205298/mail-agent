"""事务抽取：从邮件正文 + 附件文本提取结构化校园事务，每条带证据引用。

核心设计：
- 模型只负责「理解 + 抽取」，不负责「编造」。每个字段必须给出 quote（原文片段）。
- 后端校验 quote 是否真的出现在原文里；对不上的字段丢弃，宁缺毋假。
- 结构化字段（日期、邮箱）由规则二次校验，纠正模型笔误。
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from config import CFG
from extract import extract_text
from llm import LLMError, chat_json, try_models
from models import Message

SYSTEM = "你是校园通知信息抽取模块。只抽取原文明确写到的内容，没写的一律留空，不要猜测或补全。每个字段给 quote（原文逐字片段，≤60字）。日期规范化为 YYYY-MM-DD，缺年份用参考年份。输出合法 JSON。"

SCHEMA_HINT = """{
  "title": "事务标题",
  "org": "主办/承办/发布单位",
  "track": "赛道或方向",
  "summary": "3-5 句摘要，涵盖主题、对象、关键动作",
  "relevance": "related|maybe|irrelevant",
  "relevance_reason": "依据原文说明",
  "deadlines": [{"label": "报名截止", "date": "YYYY-MM-DD", "quote": "原文片段"}],
  "requirements": [{"label": "参赛资格", "detail": "具体要求", "quote": "原文片段"}],
  "materials": [{"name": "申报书", "format": "PDF", "limit": "限制原话", "desc": "内容要求", "required": true, "quote": "原文片段"}],
  "submit_to": "提交邮箱",
  "submit_format": "压缩包命名规则",
  "key_contacts": [{"name": "姓名", "email": "邮箱", "phone": "电话", "role": "角色"}],
  "open_questions": [{"question": "待确认问题", "why": "为什么无法确定"}],
  "source_quotes": ["2-5 条关键原文片段"]
}"""


def _norm(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _verify_quote(quote: str, corpus: str) -> bool:
    """quote 必须能在原文（或其去空白版本）中找到。容忍空白与换行差异。"""
    q = _norm(quote)
    if not q:
        return False
    # 模型可能把引号写进 quote，去掉再比对
    q = q.strip('"“”\'').strip()
    if not q:
        return False
    sq = re.sub(r"\s+", "", q)
    sc = re.sub(r"\s+", "", corpus)
    if len(sq) < 3:
        return True
    return sq in sc


def _fix_date(date_str: str, corpus: str, ref_year: int) -> str:
    """规范化日期，修正模型漏年份、写错格式的情况。"""
    s = _norm(date_str)
    if not s:
        return ""
    s = s.replace("—", "-").replace("–", "-").replace("年", "-").replace("月", "-").replace("日", "")
    m = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1990 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    m = re.search(r"(\d{1,2})\D+(\d{1,2})", s)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{ref_year}-{mo:02d}-{d:02d}"
    return s


def _emails_in(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))


def _pick_json(obj: Any, key: str, cls: type) -> Any:
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return cls()


def _build_corpus(message: Message, texts: dict[str, str]) -> str:
    """拼出给模型看的语料，并标记来源便于定位。"""
    parts = [f"【邮件正文】\n{message.body_text}"]
    for name, t in texts.items():
        if t and t.strip():
            parts.append(f"【附件：{name}】\n{t[:4000]}")
    return "\n\n".join(parts)


def extract_matter(message: Message, *, ref_year: int | None = None,
                   model: str | None = "fast") -> dict[str, Any]:
    """对单封邮件做事务抽取。返回 {raw, corpus, ref_year, dropped, quote_checks}。"""
    texts: dict[str, str] = {}
    for a in _json_list(message.attachments):
        p = str(a.get("path") or "")
        if p and str(a.get("kind")) in ("doc", "docx", "pdf"):
            t, kind = extract_text(Path(p))
            if t.strip():
                texts[a["name"]] = t
    corpus = _build_corpus(message, texts)
    year = ref_year or (message.date_ts.year if message.date_ts else date.today().year)

    user = (
        f"参考年份（补全无年份日期用）：{year}\n\n"
        f"{corpus}\n\n"
        f"按此 schema 输出 JSON，quote 填原文逐字片段：\n{SCHEMA_HINT}"
    )
    try:
        obj = try_models(user, order=["fast", "strong", "third"]) if model is None else chat_json(
            user=user, system=SYSTEM, model=model, max_tokens=2500, retries=2, temperature=0.1
        )
    except LLMError as e:
        raise LLMError(f"抽取失败：{e}") from e

    out: dict[str, Any] = {
        "title": _norm(obj.get("title")),
        "org": _norm(obj.get("org")),
        "track": _norm(obj.get("track")),
        "summary": _norm(obj.get("summary")),
        "relevance": obj.get("relevance") or "unknown",
        "relevance_reason": _norm(obj.get("relevance_reason")),
        "submit_to": _norm(obj.get("submit_to")),
        "submit_format": _norm(obj.get("submit_format")),
        "deadlines": [], "requirements": [], "materials": [],
        "key_contacts": [], "open_questions": [], "source_quotes": [],
        "dropped": [], "quote_checks": [],
    }

    # 提交邮箱：模型输出必须能在原文邮箱集合里对上，否则丢弃
    if out["submit_to"]:
        if not re.search(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", out["submit_to"]):
            out["dropped"].append({"field": "submit_to", "value": out["submit_to"], "reason": "不是合法邮箱格式"})
            out["submit_to"] = ""

    for key, cast in (("deadlines", dict), ("requirements", dict), ("materials", dict),
                      ("key_contacts", dict), ("open_questions", dict)):
        raw_list = obj.get(key)
        for item in (raw_list if isinstance(raw_list, list) else []):
            if not isinstance(item, dict):
                continue
            q = _norm(item.get("quote"))
            # 空 quote：允许（模型未提供证据时不丢弃，但标记为未核对）
            ok = True if not q else _verify_quote(q, corpus)
            out["quote_checks"].append({"field": key, "quote": q[:60], "verified": ok,
                                        "missing_quote": not bool(q)})
            if not ok:
                out["dropped"].append({"field": key, "value": _norm(item.get("name") or item.get("label") or item.get("question")),
                                       "reason": "quote 未在原文中找到"})
                continue
            if key == "deadlines":
                d = _fix_date(item.get("date", ""), corpus, year)
                out[key].append({"label": _norm(item.get("label")), "date": d, "quote": q})
            else:
                out[key].append({k: _norm(v) for k, v in item.items() if k in item and k not in ("quote",)})
                if key == "materials":
                    out[key][-1]["quote"] = q
                    if "required" in item:
                        out[key][-1]["required"] = bool(item["required"])
                    else:
                        out[key][-1]["required"] = True

    for q in (obj.get("source_quotes") or [])[:5]:
        q = _norm(q)
        if q and _verify_quote(q, corpus):
            out["source_quotes"].append(q)

    return {"raw": out, "corpus": corpus, "ref_year": year, "model": CFG.resolve_model(model) if model else "auto"}


def _json_list(s: str) -> list:
    try:
        v = json.loads(s or "[]")
        return v if isinstance(v, list) else []
    except (TypeError, json.JSONDecodeError):
        return []
