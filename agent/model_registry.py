"""模型清单文件（model_registry.txt）的读取与"下线标记"改写。

为什么单独一个模块：模型清单是**随仓库维护、AI 可改**的文件（git pull 即更新），
它和 .env（本地密钥区）是两码事。模块职责只有两件：

1. load() —— 把文件按 KEY=VALUE 行式解析成 {变量名: 模型列表字符串}，
   config.py 的加载流程用它把 NODE_A_MODELS 等内容注入 os.environ，
   供 config.yaml 的 ${NODE_A_MODELS} 插值使用（.env 不再承载模型清单）。
2. retire_model() —— 给指定变量名下一行的某个模型加 `#`（自动下线标记），
   幂等：已带 # 不重复加；文件里没有该行/该模型时静默返回 False。
   删掉 # 即恢复模型（人工 / 手动编辑），符合"标记删除但可手动恢复"的需求。

安全约定：本模块不碰 .env，不打印任何值；写盘用临时文件 + os.replace 原子替换，
避免中途断电/并发写坏文件。模型名匹配按"逗号分隔的裸条目"比较（不含 @ 尺寸后缀
前的模型名部分：`laguna-s-2.1@256k` 匹配模型 `laguna-s-2.1`）。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path


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
    """去掉条目里的依赖信息：`#laguna-s-2.1@256k` → `#laguna-s-2.1`。
    返回时保留行首的 `#`（屏蔽标记），模型名本身与 @ 尺寸后缀剥离开。
    """
    name = item.lstrip("#").partition("@")[0].strip()
    return ("#" if item.startswith("#") else "") + name


def retire_model(path: Path, env_var: str, model: str) -> bool:
    """给 model_registry.txt 中 `env_var=` 这一行的 `model` 加 `#` 下线标记。

    - 幂等：模型已带 # 或已不存在 → 不动文件，返回 False；
    - 命中：给该模型加 # 后原子写回，返回 True；
    - 文件/该变量行缺失：静默返回 False（不报错）。
    并发安全：进程内串行化（线程锁），写盘临时文件 + os.replace。
    """
    file_path = Path(path)
    lock = _locks.setdefault(str(file_path), threading.Lock())
    with lock:
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return False
        changed = False
        for i, line in enumerate(lines):
            if not line.strip().startswith(f"{env_var}="):
                continue
            value = line.partition("=")[2].strip()
            items = _split_items(value)
            target = model.strip()
            for j, item in enumerate(items):
                if _bare_model(item).lstrip("#") == target and not item.startswith("#"):
                    items[j] = "#" + item  # 加 # = 下线标记
                    changed = True
            if changed:
                lines[i] = f"{env_var}={','.join(items)}"
            break  # 变量名唯一，处理第一处匹配即可
        if not changed:
            return False
        _atomic_write(file_path, "\n".join(lines) + ("\n" if lines else ""))
        return True


def _atomic_write(path: Path, text: str) -> None:
    """临时文件 + os.replace 原子写盘（失败静默，留旧文件不动）。"""
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# 每个文件一把线程锁（同一模型清单文件并发改写串行化）
_locks: dict[str, threading.Lock] = {}
