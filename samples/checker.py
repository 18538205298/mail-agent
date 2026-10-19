"""提交材料校验器 + 回复草稿。

校验器是**规则引擎，不调模型**：文件存在性、格式、时长、大小、命名规则都是确定的，
用代码判比用模型判可靠得多（模型会说「符合要求」但其实没检查）。
模型只用来写回复草稿和解释问题。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from llm import chat_text

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ARCHIVE_EXTS = (".zip", ".rar", ".7z", ".tar", ".gz")


@dataclass
class CheckResult:
    ok: bool
    items: list[dict] = field(default_factory=list)
    problems: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _file_duration_sec(path: Path) -> float | None:
    """用 ffprobe 读视频时长。没有 ffprobe 返回 None（不假设）。"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)], capture_output=True, text=True, timeout=60,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(json.loads(out.stdout)["format"]["duration"])
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, json.JSONDecodeError):
        pass
    return None


def _check_video(path: Path, raw_limit: str) -> list[dict]:
    """按「时长≤X分钟，文件≤YMB」原话解析并逐项校验。"""
    problems: list[dict] = []
    max_min = None
    m = re.search(r"时长[≤<]\s*(\d+(?:\.\d+)?)\s*分", raw_limit or "")
    if m:
        max_min = float(m.group(1))
    max_mb = None
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*(?:MB|mb|M)\b", raw_limit or "")
    if m2:
        max_mb = float(m2.group(1))
    exists = path.is_file()
    problems.append({"item": "存在性", "ok": exists, "detail": str(path)})
    if not exists:
        return problems
    size_mb = path.stat().st_size / 1024 / 1024
    problems.append({"item": "大小", "ok": max_mb is None or size_mb <= max_mb,
                     "detail": f"{size_mb:.1f} MB" + (f"（要求 ≤{max_mb:.0f}MB）" if max_mb else "（未给上限）")})
    dur = _file_duration_sec(path)
    if max_min is not None:
        if dur is None:
            problems.append({"item": "时长", "ok": False, "detail": "无法读取时长（缺少 ffprobe）"})
        else:
            problems.append({"item": "时长", "ok": dur <= max_min * 60,
                             "detail": f"{dur/60:.2f} 分钟（要求 ≤{max_min:.0f}分钟）"})
    return problems


def _check_archive(path: Path, expect: list[str]) -> list[dict]:
    """校验压缩包内是否包含所需文件名（不依赖 rarfile，zip 直接读）。"""
    problems: list[dict] = []
    if not path.is_file():
        return [{"item": "存在性", "ok": False, "detail": str(path)}]
    names: set[str] = set()
    if path.suffix.lower() == ".zip":
        try:
            import zipfile

            with zipfile.ZipFile(path) as z:
                names = {os.path.basename(n) for n in z.namelist() if not n.endswith("/")}
        except OSError as e:
            return [{"item": "存在性", "ok": True, "detail": str(path)},
                    {"item": "可读性", "ok": False, "detail": f"zip 打开失败: {str(e)[:80]}"}]
    else:
        problems.append({"item": "格式", "ok": True,
                         "detail": f"{path.suffix} 非 zip，无法读取内部内容，请人工核对"})
        return problems
    missing = []
    for e in expect:
        en = e.get("name", "")
        if not en:
            continue
        if not any(en in n for n in names):
            missing.append(en)
    problems.append({"item": "内含材料", "ok": not missing,
                     "detail": "全部包含" if not missing else "缺少：" + "、".join(missing)})
    return problems


def check_submission(matter: dict, provided: dict[str, str] | None = None,
                     submit_to: str = "") -> dict:
    """校验提交包。matter 是抽取出的事务结构，provided 是用户给出的实际文件路径。"""
    provided = provided or {}
    materials = matter.get("materials") or []
    expected = [m for m in materials if m.get("name")]
    result = CheckResult(ok=True)
    result.notes.append("材料要求由通知原文提取，格式与限制均按原文，未做换算。")

    for m in expected:
        name = m["name"]
        path_s = provided.get(name) or provided.get(m.get("format", ""))
        item = {"name": name, "format": m.get("format", ""), "required": m.get("required", True),
                "limit": m.get("limit", ""), "path": path_s or "", "checks": []}
        if not path_s:
            item["checks"].append({"item": "已提供", "ok": False, "detail": "未提供文件路径"})
            if m.get("required", True):
                result.problems.append({"item": name, "reason": "未提供（必需材料）"})
                result.ok = False
        else:
            p = Path(path_s)
            if not p.is_file():
                item["checks"].append({"item": "存在性", "ok": False, "detail": f"文件不存在：{p}"})
                result.problems.append({"item": name, "reason": f"文件不存在：{p}"})
                result.ok = False
                result.items.append(item)
                continue
            # 格式校验：原文要求 PDF 就必须是 PDF
            want = (m.get("format") or "").upper().replace("扫描件", "")
            got = p.suffix.upper().lstrip(".")
            fmt_ok = (not want) or (got == want) or (want in got)
            item["checks"].append({"item": "格式", "ok": fmt_ok,
                                   "detail": f"要求 {want or '未指定'}，实际 {got}"})
            if not fmt_ok:
                result.problems.append({"item": name, "reason": f"格式不符：要求 {want}，实际 {got}"})
                result.ok = False
            if "视频" in name or got == "MP4" or "MP4" in (m.get("format") or ""):
                item["checks"].extend(_check_video(p, m.get("limit", "")))
                for c in item["checks"]:
                    if not c["ok"]:
                        result.problems.append({"item": name, "reason": f"{c['item']}不达标：{c['detail']}"})
                        result.ok = False
        result.items.append(item)

    # 压缩包命名规则校验
    rule = (matter.get("submit_format") or "").strip()
    if rule:
        if provided.get("__archive__"):
            ap = Path(provided["__archive__"])
            nm = ap.name if ap.is_file() else ap.stem
            if "+" in rule and "+" in nm:
                result.items.append({"name": "__archive__", "checks": [
                    {"item": "命名", "ok": True, "detail": f"「{nm}」符合「{rule}」"}]})
            else:
                result.items.append({"name": "__archive__", "checks": [
                    {"item": "命名", "ok": False, "detail": f"「{nm}」不匹配规则「{rule}」"}]})
                result.problems.append({"item": "压缩包命名", "reason": f"应为「{rule}」"})
                result.ok = False
        else:
            result.notes.append(f"压缩包命名规则：{rule}")

    # 提交邮箱核对
    exp_to = matter.get("submit_to") or ""
    if exp_to and submit_to:
        ok = submit_to.strip().lower() == exp_to.strip().lower()
        result.items.append({"name": "提交邮箱", "checks": [
            {"item": "收件人", "ok": ok, "detail": f"通知要求 {exp_to}，实际填写 {submit_to}"}]})
        if not ok:
            result.problems.append({"item": "收件人", "reason": f"应为 {exp_to}，实际 {submit_to}"})
            result.ok = False
    return result.to_dict()


def draft_reply(matter: dict, intent: str, from_name: str, context: str = "") -> str:
    """基于事务事实起草回复邮件（模型只写，不写事实）。"""
    sysp = (
        "你是学生的邮件助手。根据下方「事务事实」为用户起草一封邮件。"
        "规则：1) 只使用给定事实，不要虚构日期、姓名、编号、链接；"
        "2) 事实里缺失的信息用【待填写】占位，不要瞎猜；"
        "3) 语气礼貌正式，中文，不超 400 字；"
        "4) 直接输出邮件正文（含称呼与落款），不要写额外解释。"
    )
    facts = {
        "标题": matter.get("title"), "主办/发布": matter.get("org"),
        "联系邮箱": matter.get("submit_to"), "联系人": matter.get("key_contacts"),
        "截止": matter.get("deadlines"), "材料要求": matter.get("materials"),
        "待确认问题": matter.get("open_questions"),
    }
    user = (
        f"邮件意图：{intent}\n发送者：{from_name}\n补充背景：{context or '（无）'}\n"
        f"事务事实：\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n请起草邮件正文。"
    )
    return chat_text(user=user, system=sysp, model="strong", max_tokens=2000).strip()
