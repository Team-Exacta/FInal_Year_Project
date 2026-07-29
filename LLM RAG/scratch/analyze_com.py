import json

# Load test questions for ground truth info
with open('data/evaluation/test_questions.json', encoding='utf-8') as f:
    all_q = json.load(f)

com_questions = [q for q in all_q if q['id'].startswith('Q_COM_')]

# Load raw results (current)
with open('outputs/ragas_results/5_Full_Architecture/raw_results.json', encoding='utf-8') as f:
    raw = json.load(f)
raw_by_id = {r['id']: r for r in raw}

lines = []
lines.append("=" * 80)
lines.append("COMPARISON QUERY ANALYSIS")
lines.append("=" * 80)

for q in com_questions:
    qid = q['id']
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"ID: {qid}")
    lines.append(f"Question: {q['question']}")
    lines.append(f"Expected Intent: {q['intent']}")
    lines.append(f"Ground Truth: {q['ground_truth']}")
    
    if qid in raw_by_id:
        r = raw_by_id[qid]
        lines.append(f"Detected Intent: {r['detected_intent']}")
        lines.append(f"Answer (first 400): {r['answer'][:400]}")
        lines.append(f"Contexts ({len(r['contexts'])} chunks):")
        for i, ctx in enumerate(r['contexts']):
            preview = ctx[:300].replace('\n', ' ')
            lines.append(f"  [{i+1}] {preview}...")
    else:
        lines.append("  [NOT YET RUN - needs full pipeline execution]")
    lines.append("")

with open('scratch/com_analysis_full.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Done. Written {len(com_questions)} queries.")
