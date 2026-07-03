"""
Lan-gpt-image-2 配置文件管理模块

首次运行时自动生成 config.json，用户可编辑该文件来修改节点的默认参数。
配置文件包含 API 密钥等敏感信息，已加入 .gitignore，请勿提交到版本控制系统。
"""

import os
import json

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(_PLUGIN_DIR, "config.json")

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "_说明": "Lan-gpt-image-2 配置文件。修改此文件中的值将改变节点的默认参数。重启 ComfyUI 后生效。",
    "_注意": "此文件可能包含 API 密钥等敏感信息，请勿提交到版本控制系统（已加入 .gitignore）。",

    "api_key": "",
    "_api_key_说明": "API 密钥。留空则按顺序读取环境变量 OPENAI_API_KEY。",

    "base_url": "https://api.openai.com/v1",
    "_base_url_说明": "API 基础地址。例如官方 https://api.openai.com/v1，本地代理 http://localhost:8317/v1。",

    "model": "gpt-image-1",
    "_model_说明": "模型名称。代理可能使用不同的模型名。",

    "quality": "auto",
    "_quality_说明": "图像质量：auto / low / medium / high。",

    "size": "auto",
    "_size_说明": "输出尺寸：auto / 1024x1024 / 1024x1536 / 1536x1024 / 1024x1792 / 1792x1024。",

    "n": 1,
    "_n_说明": "每次请求生成的图像数量（1-10）。",

    "background": "opaque",
    "_background_说明": "背景：opaque（不透明）/ transparent（透明）。",

    "moderation": "auto",
    "_moderation_说明": "内容审核：auto / low / none。none 需代理支持，不支持时自动回退到 low。",

    "auto_fallback_moderation": True,
    "_auto_fallback_moderation_说明": "API 拒绝当前审核级别时是否自动回退到 low。",

    "output_format": "png",
    "_output_format_说明": "输出格式：png / jpeg / webp。",

    "output_compression": 85,
    "_output_compression_说明": "压缩率 0-100，仅对 jpeg/webp 生效。",

    "seed": 0,
    "_seed_说明": "随机种子，0 表示随机。",

    "negative_prompt": "",
    "_negative_prompt_说明": "负面提示词，描述不希望出现的内容（需 API 支持）。",

    "timeout": 120,
    "_timeout_说明": "HTTP 请求超时秒数（10-600）。",

    "max_retries": 3,
    "_max_retries_说明": "瞬时失败的最大重试次数（0-10）。",

    "retry_delay": 2.0,
    "_retry_delay_说明": "重试基础延迟秒数，实际延迟按指数退避递增。",

    "extra_headers": "",
    "_extra_headers_说明": "额外 HTTP 头，JSON 格式，用于代理认证。例如 {\"X-Key\": \"value\"}。",

    "save_to_disk": False,
    "_save_to_disk_说明": "是否将生成的图像自动保存到磁盘。",

    "output_dir": "lan_gpt_image_output",
    "_output_dir_说明": "保存目录名。相对路径时位于 ComfyUI 输出目录下。",
}

# 实际参与逻辑的键（排除 _ 开头的说明字段）
_CONFIG_KEYS = [k for k in DEFAULT_CONFIG if not k.startswith("_")]


def load_config() -> dict:
    """
    加载配置文件。如果文件不存在（首次运行），则用默认值生成它。
    返回的字典只包含实际配置项（不含 _ 开头的说明字段）。
    """
    if not os.path.exists(CONFIG_FILE):
        print("[Lan-gpt-image-2] 首次运行，正在生成默认配置文件 config.json ...")
        save_config(DEFAULT_CONFIG)
        print(f"[Lan-gpt-image-2] 配置文件已生成：{CONFIG_FILE}")
        print("[Lan-gpt-image-2] 你可以编辑此文件来修改节点的默认参数。")
    else:
        _migrate_config()

    # 读取配置
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[Lan-gpt-image-2] 配置文件读取失败 ({e})，使用默认值。")
        return {k: DEFAULT_CONFIG[k] for k in _CONFIG_KEYS}

    # 只提取实际配置项，与默认值合并
    cfg = {}
    for k in _CONFIG_KEYS:
        if k in raw:
            cfg[k] = raw[k]
        else:
            cfg[k] = DEFAULT_CONFIG[k]
    return cfg


def save_config(cfg: dict) -> None:
    """将配置写入文件。"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[Lan-gpt-image-2] 配置文件写入失败：{e}")


def _migrate_config() -> None:
    """如果配置文件缺少新字段，补充默认值（向后兼容）。"""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return

    changed = False
    for k in _CONFIG_KEYS:
        if k not in raw:
            raw[k] = DEFAULT_CONFIG[k]
            raw[f"_{k}_说明"] = DEFAULT_CONFIG.get(f"_{k}_说明", "")
            changed = True

    if changed:
        save_config(raw)
        print("[Lan-gpt-image-2] 配置文件已更新，补充了新增字段。")


# 模块导入时加载配置，生成文件（首次运行）
_config = load_config()


def get_config() -> dict:
    """获取当前配置（只读副本）。"""
    return dict(_config)
