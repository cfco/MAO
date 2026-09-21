"""Skill 机制：skills/ 目录扫描 + 按需加载 + 脚本执行。

一个技能 = 一个文件夹：
  skills/<技能名>/SKILL.md    # 开头 YAML frontmatter: name + description；正文为操作指引
  skills/<技能名>/scripts/    # 可选，Python 脚本（从 argv[1] 接收 JSON 参数，stdout 输出结果）

省上下文的关键：启动时只把每个技能的 name+description 注入系统提示词，
Agent 判断相关后再用 load_skill 工具读取全文。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from .tools.base import FunctionTool, ToolRegistry

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
SCRIPT_TIMEOUT = 120
# 脚本名白名单：只允许 scripts/ 下的裸文件名（字母/数字/._-，必须以 .py 结尾）
_SCRIPT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+\.py$")


def _safe_script_name(script: str) -> str | None:
    """校验脚本入参是否为 scripts/ 下的裸文件名。合法返回文件名，非法返回 None。

    `script` 来自工具入参（bridge 外部主可直接传），不做校验就等于把任意路径
    交给 subprocess 执行：`../../../../x.py`、绝对路径、`C:/x.py` 都能跑。
    显式拒绝路径分隔符、盘符、`..`、隐藏文件，并要求 .py 后缀。
    """
    s = (script or "").strip()
    if not s or s.startswith(".") or ".." in s:
        return None
    if "/" in s or "\\" in s or ":" in s:
        return None
    if not _SCRIPT_NAME_PATTERN.match(s):
        return None
    return s


class SkillManager:
    def __init__(self, skills_dir: Path | None = None):
        self.skills_dir = skills_dir or SKILLS_DIR
        self.skills: dict[str, dict] = {}

    def scan(self) -> None:
        """扫描所有 */SKILL.md，解析 frontmatter。"""
        self.skills.clear()
        if not self.skills_dir.exists():
            return
        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            try:
                meta = self._parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            except OSError:
                continue
            name = str(meta.get("name") or skill_md.parent.name)
            self.skills[name] = {
                "dir": skill_md.parent,
                "description": str(meta.get("description", "")),
            }

    @staticmethod
    def _parse_frontmatter(text: str) -> dict:
        if not text.startswith("---"):
            return {}
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}
        try:
            data = yaml.safe_load(parts[1]) or {}
            return data if isinstance(data, dict) else {}
        except yaml.YAMLError:
            return {}

    def overview(self) -> str:
        """技能清单文本（注入系统提示词用）。"""
        if not self.skills:
            return "（暂无技能。可在 skills/ 目录按规范添加：每个技能一个文件夹，内含 SKILL.md）"
        return "\n".join(f"- {name}: {info['description']}" for name, info in self.skills.items())

    def load_full(self, name: str) -> str:
        info = self.skills.get(name)
        if not info:
            return f"错误：技能 '{name}' 不存在。可用技能：{', '.join(self.skills) or '无'}"
        return (info["dir"] / "SKILL.md").read_text(encoding="utf-8")

    def run_script(self, skill: str, script: str, args: dict | None = None) -> str:
        info = self.skills.get(skill)
        if not info:
            return f"错误：技能 '{skill}' 不存在。可用技能：{', '.join(self.skills) or '无'}"
        name = _safe_script_name(script)
        if name is None:
            return f"错误：脚本名非法（只能是 scripts/ 下的文件名）：{script!r}"
        scripts_dir = info["dir"] / "scripts"
        path = scripts_dir / name
        # 双保险：即便上面的白名单被绕过，落点也必须仍在技能自己的 scripts/ 内，
        # 否则 execute_skill_script 就能拿 `../../x.py` 跑项目外任意 Python 文件。
        try:
            path.resolve().relative_to(scripts_dir.resolve())
        except (ValueError, OSError):
            return f"错误：脚本路径超出技能目录，已拒绝：{script!r}"
        if not path.exists():
            return f"错误：脚本不存在：{path}"
        payload = json.dumps(args or {}, ensure_ascii=False)
        try:
            proc = subprocess.run(
                [sys.executable, str(path), payload],
                capture_output=True,
                cwd=str(info["dir"]),
                timeout=SCRIPT_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return f"错误：脚本超过 {SCRIPT_TIMEOUT} 秒未完成，已终止"
        from .tools.builtin import _decode
        parts = []
        if proc.stdout:
            parts.append(_decode(proc.stdout)[-8000:])
        if proc.stderr:
            parts.append("[stderr] " + _decode(proc.stderr)[-2000:])
        parts.append(f"[exit code] {proc.returncode}")
        return "\n".join(parts)


def register_skill_tools(registry: ToolRegistry, manager: SkillManager) -> None:
    registry.register(FunctionTool(
        name="load_skill",
        description="读取指定技能的完整说明（SKILL.md）。系统提示词中的技能列表若与当前任务相关，先调用本工具再行动。",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "技能名"}},
            "required": ["name"],
        },
        func=lambda a: manager.load_full(str(a.get("name", ""))),
    ))
    registry.register(FunctionTool(
        name="execute_skill_script",
        description="执行技能自带的 Python 脚本。脚本通过命令行参数接收一个 JSON 字符串，结果用 stdout 输出。",
        input_schema={
            "type": "object",
            "properties": {
                "skill": {"type": "string", "description": "技能名"},
                "script": {"type": "string", "description": "scripts/ 目录下的脚本文件名，如 hello.py"},
                "args": {"type": "object", "description": "传给脚本的参数对象", "default": {}},
            },
            "required": ["skill", "script"],
        },
        func=lambda a: manager.run_script(
            str(a.get("skill", "")), str(a.get("script", "")), a.get("args") or {}
        ),
    ))
