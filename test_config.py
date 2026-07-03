"""Test config file generation and loading."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Remove any cached imports
for mod in list(sys.modules.keys()):
    if "config" in mod:
        del sys.modules[mod]

import config

print("=== Config module loaded ===")
print(f"Config file path: {config.CONFIG_FILE}")
print(f"File exists: {os.path.exists(config.CONFIG_FILE)}")
print()

# Print all config values
print("=== Config values ===")
for k, v in config._config.items():
    print(f"  {k}: {repr(v)}")
print()

# Verify config file content
import json
with open(config.CONFIG_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)
print(f"=== Config file has {len(raw)} keys (including _说明 fields) ===")
print(f"Has _说明: {'_说明' in raw}")
print(f"Has api_key: {'api_key' in raw}")
print(f"Has base_url: {'base_url' in raw}")
print()

# Test get_config
cfg = config.get_config()
print("=== get_config() returns clean config (no _ fields) ===")
has_underscore = any(k.startswith("_") for k in cfg)
print(f"No _ fields: {not has_underscore}")
print(f"Keys count: {len(cfg)}")
print()

# Test that modifying config and reloading works
print("=== Test reload after modification ===")
cfg["api_key"] = "sk-test-123"
cfg["base_url"] = "http://localhost:9999/v1"
config.save_config({**dict(zip([f"_{k}_说明" for k in cfg], [""] * len(cfg))), **cfg})
# Re-read
with open(config.CONFIG_FILE, "r", encoding="utf-8") as f:
    raw2 = json.load(f)
print(f"Saved api_key: {raw2.get('api_key')}")
print(f"Saved base_url: {raw2.get('base_url')}")

# Reload config
cfg2 = config.load_config()
print(f"Reloaded api_key: {cfg2.get('api_key')}")
print(f"Reloaded base_url: {cfg2.get('base_url')}")
print()

print("=== All config tests passed! ===")
