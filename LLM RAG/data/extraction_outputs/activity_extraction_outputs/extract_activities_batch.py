import sys, os, csv, json, shutil
sys.path.insert(0, '/mnt/data')
from extraction_core import INPUT, extract, clean_text
from collections import defaultdict, Counter

OUT_DIR='/mnt/data'
TMP_DIR='/tmp/activity_extract'
os.makedirs(TMP_DIR, exist_ok=True)
paths={
 'jsonl': os.path.join(TMP_DIR,'review_activity_triples.jsonl'),
 'json': os.path.join(TMP_DIR,'review_activity_triples.json'),
 'csv': os.path.join(TMP_DIR,'review_activity_triples.csv'),
 'summary': os.path.join(TMP_DIR,'activity_summary_by_place.csv'),
 'empty': os.path.join(TMP_DIR,'reviews_with_no_activity.csv'),
 'stats': os.path.join(TMP_DIR,'activity_extraction_stats.json')
}
final={k: os.path.join(OUT_DIR, os.path.basename(v)) for k,v in paths.items()}

triples=[]
empty=[]
place_counter=Counter(); activity_counter=Counter(); sent_counter=Counter(); summary_counts=defaultdict(Counter); review_has=set()
rows_read=0
with open(INPUT, newline='', encoding='utf-8-sig') as f:
    reader=csv.DictReader(f)
    for idx,row in enumerate(reader, start=1):
        rows_read+=1
        place=clean_text(row.get('place_name'))
        rid=row.get('review_id') or f'R{idx:06d}'
        items=extract(place, rid, row.get('title'), row.get('body'))
        if not items:
            empty.append({'review_id':rid,'place_name':place,'title':clean_text(row.get('title')),'body':clean_text(row.get('body'))})
            continue
        review_has.add(rid)
        for it in items:
            triples.append(it)
            place_counter[it['subject']]+=1
            activity_counter[it['object']]+=1
            sent_counter[it['sentiment']]+=1
            summary_counts[(it['subject'], it['object'])][it['sentiment']]+=1

# write files quickly
with open(paths['jsonl'],'w',encoding='utf-8') as f:
    f.write('\n'.join(json.dumps(x, ensure_ascii=False) for x in triples))
    f.write('\n')
with open(paths['json'],'w',encoding='utf-8') as f:
    f.write('[\n')
    f.write(',\n'.join(json.dumps(x, ensure_ascii=False) for x in triples))
    f.write('\n]\n')
fieldnames=["subject","subject_type","relation","object","object_type","sentiment","evidence_id","evidence","confidence"]
with open(paths['csv'],'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(triples)
with open(paths['empty'],'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=['review_id','place_name','title','body']); w.writeheader(); w.writerows(empty)
with open(paths['summary'],'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=['place_name','activity','positive_count','negative_count','neutral_count','total_mentions','dominant_sentiment'])
    w.writeheader()
    for (place,act),counts in sorted(summary_counts.items()):
        total=counts['positive']+counts['negative']+counts['neutral']
        dominant=max(['positive','negative','neutral'], key=lambda s:(counts[s], s=='positive'))
        w.writerow({'place_name':place,'activity':act,'positive_count':counts['positive'],'negative_count':counts['negative'],'neutral_count':counts['neutral'],'total_mentions':total,'dominant_sentiment':dominant})

stats={
 'input_rows': rows_read,
 'reviews_with_activity': len(review_has),
 'reviews_without_activity': len(empty),
 'activity_triples': len(triples),
 'unique_places_with_activities': len(place_counter),
 'unique_activity_types': len(activity_counter),
 'sentiment_counts': dict(sent_counter),
 'top_activities': activity_counter.most_common(20),
 'top_places': place_counter.most_common(20),
 'outputs': {k: final[k if k!='summary' else 'summary'] for k in []}
}
stats['outputs']={'jsonl': final['jsonl'], 'json': final['json'], 'csv': final['csv'], 'summary_csv': final['summary'], 'no_activity_csv': final['empty'], 'stats_json': final['stats']}
with open(paths['stats'],'w',encoding='utf-8') as f: json.dump(stats,f,ensure_ascii=False,indent=2)

# move to /mnt/data
for k in paths:
    shutil.copyfile(paths[k], final[k])
print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
os._exit(0)
