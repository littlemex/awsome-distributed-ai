#!/usr/bin/env python3
"""Write one cell's config: a filled-in base config plus JSON overrides (later arguments win)."""
import json
import sys

if len(sys.argv) < 3:
    sys.exit("usage: make_config.py BASE_CONFIG OUT [JSON_OVERRIDES ...]")
config = json.load(open(sys.argv[1]))
for override in sys.argv[3:]:
    config.update(json.loads(override))
json.dump(config, open(sys.argv[2], "w"), indent=2)
print(sys.argv[2])
