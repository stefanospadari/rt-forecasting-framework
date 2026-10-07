"""Prints rooms, metrics and data path of a config, for the bash scripts.
Usage: python3 scripts/config_info.py [configs/base.yaml]   (requires PyYAML)
Output: three lines -> data_path / rooms / metrics
"""
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml"))
data = cfg.get("data", {})
print(data.get("path", "data/archive/KETI"))
print(" ".join(str(r) for r in data.get("rooms", [])))
print(" ".join(data.get("metrics") or ["temperature", "humidity", "co2"]))
