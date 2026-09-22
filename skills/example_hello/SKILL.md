---
name: example_hello
description: 示例技能，用于验证 Skill 机制。当用户想测试技能加载、执行技能脚本，或想了解怎么写新技能时使用。
---

# 示例技能 Hello

最小可用的技能示例，演示 Skill 机制三要素：说明文档、按需加载、可执行脚本。

## 什么时候用
- 用户让你演示或测试技能系统
- 用户想参考着写一个新技能

## 怎么用
调用 run_skill_script 工具：
- skill: `example_hello`
- script: `hello.py`
- args: `{"name": "对方的称呼"}`（可省略，默认叫"朋友"）

## 怎么写新技能
1. 在 `skills/` 下新建文件夹（建议英文名 + 短横线）
2. 写 `SKILL.md`：frontmatter 必须包含 `name` 和 `description`——description 决定主控 Agent 什么时候会想到用它，要写清楚适用场景
3. 正文写清使用步骤；需要代码就把 Python 脚本放进 `scripts/` 目录
