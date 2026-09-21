"""示例技能脚本：从 argv[1] 读取 JSON 参数，向对方问好。"""
import json
import sys


def main() -> None:
    raw = sys.argv[1] if len(sys.argv) > 1 else "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        args = {}
    name = args.get("name", "朋友")
    print(f"你好，{name}！这条输出来自技能脚本 example_hello/scripts/hello.py")
    print(f"收到的参数：{json.dumps(args, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
