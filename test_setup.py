# test_setup.py
import json
import yaml

with open("data/train.json") as f:
    train = json.load(f)

sample = train[0]

# Verify all three fields exist
assert "instruction" in sample, "Missing instruction"
assert "input" in sample, "Missing input"
assert "output" in sample, "Missing output"

# Verify triage label exists in output
labels = ["DISPATCH", "ESCALATE", "MONITOR", "IGNORE"]
label_found = any(label in sample["output"] for label in labels)
assert label_found, "No triage label found in output"

# Test prompt formatting
PROMPT_TEMPLATE = """Below is an instruction that describes 
a security surveillance task. Analyze the detection log and 
provide a triage decision.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

formatted = PROMPT_TEMPLATE.format(
    sample["instruction"],
    sample["input"],
    sample["output"]
)

print("Sample formatted correctly:")
print(formatted[:400])
print("\nAll checks passed ✓")