import json

with open("outputs/ragas_results/5_Full_Architecture/raw_results.json", encoding="utf-8") as f:
    data = json.load(f)

low_ids = ["Q_ACT_10", "Q_ACT_07", "Q_ACT_01"]
for d in data:
    if d["id"] in low_ids:
        print(f"=== {d['id']} ===")
        print("Q:", d["question"])
        print("A:", d["answer"][:600])
        print()
