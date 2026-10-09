"""模型清单文件（model_registry.txt）的读取与"下线标记"改写。

为什么单独一个模块：模型清单是**随仓库维护、AI 可改**的文件（git pull 即更新），
它和 .env（本地密钥区）是两码事。模块职责只有两件：

1. load() —— 把文件按 KEY=VALUE 行式解析成 {变量名: 模型列表字符串}，
   config.py 的加载流程用它把 NODE_A_MODELS 等内容注入 os.environ，
   供 config.yaml 的 ${NODE_A_MODELS} 插值使用（.env 不再承载模型清单）。
2. retire_model() —— 给指定变量名下一行的某个模型加 `#`（自动下线标记），
   返回三态（本次下线 / 已带 # / 未找到或写入失败），幂等：已带 # 不重复加。
   删掉 # 即恢复模型（人工 / 手动编辑），符合"标记删除但可手动恢复"的需求。

安全约定：本模块不碰 .env，不打印任何值；写盘用临时文件 + os.replace 原子替换，
避免中途断电/并发写坏文件。模型名匹配按"逗号分隔的裸条目"比较（不含 @ 尺寸后缀
前的模型名部分：`laguna-s-2.1@256k` 匹配模型 `laguna-s-2.1`）。
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from .core.ledger import atomic_write_text

# retire_model 的三态返回值（对外稳定：调用方按值拼提示文本）
RETIRE_OFFLINED = "offlined"
RETIRE_ALREADY = "already"
RETIRE_MISSING = "missing"
# 什么算"窗口标注"后缀：纯数字、可带 k/K（与 config._split_models 同一条规则）
_SIZE_SUFFIX = re.compile(r"\d+[kK]?")


def load_model_registry(path: Path) -> dict[str, str]:
    """读取模型清单文件，返回 {变量名: 值}（忽略注释行与空行）。

    与 _load_dotenv 的解析规则对齐：KEY=VALUE、注释行独占、值去首尾空白。
    文件不存在返回空 dict（不抛异常，配置分层里它是可缺省的基础层）。
    """
    out: dict[str, str] = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, _, val = s.partition("=")
            key = key.strip()
            if not key:
                continue
            out[key] = val.strip()
    except OSError:
        pass  # 文件缺失/读取失败：当作空清单，不拖垮启动
    return out


def _split_items(raw: str) -> list[str]:
    """把一个模型的逗号清单拆成条目（保留 # 前缀，供幂等判断）。"""
    return [p.strip() for p in str(raw or "").split(",") if p.strip()]


def _bare_model(item: str) -> str:
    """剥掉条目的窗口标注得到模型名：`#laguna-s-2.1@256k` → `#laguna-s-2.1`。

    行首 `#`（下线标记）保留。`@` 之后**不像尺寸标注**时整串都是模型名：Cloudflare
    的模型名本身带 @（`@cf/meta/llama-3.1-8b-instruct`），只按"最后一个 @"硬切会把
    不带 @ 的普通条目切成**空串**（`"m-one".rpartition("@")[0]` 是 `''`），于是
    retire_model 对清单里绝大多数模型永远匹配不上、自动下线静默失效。
    口径与 config._split_models 必须一致（同一份"什么算窗口标注"的规则）。
    """
    body = item.lstrip("#")
    name, sep, suffix = body.rpartition("@")
    bare = name.strip() if sep and _SIZE_SUFFIX.fullmatch(suffix.strip()) else body.strip()
    return ("#" if item.startswith("#") else "") + bare


def retire_model(path: Path, env_var: str, model: str) -> str:
    """给 model_registry.txt 中 `env_var=` 这一行的 `model` 加 `#` 下线标记。

    返回三态（调用方据此拼提示，不必自己再解析一遍文件）：
    - RETIRE_OFFLINED：本次加上了 #，已原子写回；
    - RETIRE_ALREADY：该条目已带 #（曾自动下线或人工屏蔽），不动文件；
    - RETIRE_MISSING：没有该变量行 / 行里没有该模型 / 读写失败（含文件缺失）。
    删掉 # 即恢复模型（人工 / 手动编辑），符合"标记删除但可手动恢复"的需求。
    并发安全：进程内按文件串行化（线程锁），写盘临时文件 + os.replace。
    """
    file_path = Path(path)
    lock = _locks.setdefault(str(file_path), threading.Lock())
    with lock:
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return RETIRE_MISSING
        found = False  # 该变量行里是否出现过这个模型的条目（带不带 # 都算出现）
        changed = False
        for i, line in enumerate(lines):
            if not line.strip().startswith(f"{env_var}="):
                continue
            value = line.partition("=")[2].strip()
            items = _split_items(value)
            target = model.strip()
            for j, item in enumerate(items):
                if _bare_model(item).lstrip("#") != target:
                    continue
                found = True
                if not item.startswith("#"):
                    items[j] = "#" + item  # 加 # = 下线标记
                    changed = True
            if changed:
                lines[i] = f"{env_var}={','.join(items)}"
            break  # 变量名唯一，处理第一处匹配即可
        if not changed:
            return RETIRE_ALREADY if found else RETIRE_MISSING
        try:
            atomic_write_text(file_path, "\n".join(lines) + ("\n" if lines else ""))
        except OSError:
            return RETIRE_MISSING  # 写不进去：当作没下线，绝不让 IO 异常炸到派工路径
        return RETIRE_OFFLINED


# 每个文件一把线程锁（同一模型清单文件并发改写串行化）
_locks: dict[str, threading.Lock] = {}
