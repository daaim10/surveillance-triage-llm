import json
import random
import os

# ── File Paths ─────────────────────────────────────────
DATA_DIR = r"C:\Users\karachigamerz.com\OneDrive\Desktop\lora_finetune\data"

INPUT_FILE = os.path.join(DATA_DIR, "data_seed_claude.json")
TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
VAL_FILE = os.path.join(DATA_DIR, "val.json")
TEST_FILE = os.path.join(DATA_DIR, "test.json")

# ── Load your data ─────────────────────────────────────
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total examples loaded: {len(data)}")

# ── Shuffle with fixed seed ────────────────────────────
random.seed(42)
random.shuffle(data)

# ── Split (80/10/10) ───────────────────────────────────
n = len(data)

train_end = int(0.8 * n)
val_end = train_end + int(0.1 * n)

train = data[:train_end]
val = data[train_end:val_end]
test = data[val_end:]

print(f"Train: {len(train)}")
print(f"Val:   {len(val)}")
print(f"Test:  {len(test)}")

# ── Validate no overlap ────────────────────────────────
train_inputs = set(ex["input"] for ex in train)
val_inputs = set(ex["input"] for ex in val)
test_inputs = set(ex["input"] for ex in test)

assert len(train_inputs & val_inputs) == 0, "Train/Val overlap!"
assert len(train_inputs & test_inputs) == 0, "Train/Test overlap!"
assert len(val_inputs & test_inputs) == 0, "Val/Test overlap!"

print("✓ No overlaps found between splits.")

# ── Check output quality ───────────────────────────────
def check_outputs(split, name):
    empty = [
        i for i, ex in enumerate(split)
        if not ex.get("output", "").strip()
    ]

    short = [
        i for i, ex in enumerate(split)
        if len(ex.get("output", "")) < 50
    ]

    print(f"\n{name}")
    print("-" * 30)
    print(f"Total examples: {len(split)}")
    print(f"Empty outputs : {len(empty)}")
    print(f"Short outputs : {len(short)}")

    if empty:
        print("WARNING - Empty outputs at:", empty)

    if short:
        print("WARNING - Short outputs at:", short)

check_outputs(train, "Train")
check_outputs(val, "Validation")
check_outputs(test, "Test")

# ── Save splits ────────────────────────────────────────
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    json.dump(train, f, indent=2, ensure_ascii=False)

with open(VAL_FILE, "w", encoding="utf-8") as f:
    json.dump(val, f, indent=2, ensure_ascii=False)

with open(TEST_FILE, "w", encoding="utf-8") as f:
    json.dump(test, f, indent=2, ensure_ascii=False)

print("\nFiles saved successfully:")
print(TRAIN_FILE)
print(VAL_FILE)
print(TEST_FILE)

print("\n✓ Test set is now locked.")
print("Use it only once after all training and hyperparameter tuning is complete.")