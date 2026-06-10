import json
import sys
from collections import Counter

path = sys.argv[1]

total = 0
correct = 0
counter = Counter()

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        total += 1
        correct += int(x.get("correct", False))
        counter[x.get("prediction", "unknown")] += 1

print("file:", path)
print("total:", total)
print("correct:", correct)
print("accuracy:", correct / total if total else 0)
print("prediction counts:", dict(counter))
