# create_classes.py
import json
from pathlib import Path

OUT = Path("classes.json")
letters = [chr(ord("A")+i).lower() for i in range(26)]
mapping = {}
for i,name in enumerate(letters):
    mapping[name] = i
# extras
mapping["delete"] = 26
mapping["space"]  = 27

OUT.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
print("Wrote classes.json with mapping (first entries):")
for k in list(mapping.keys())[:6]:
    print(k, "->", mapping[k])
print("Total classes:", len(mapping))
