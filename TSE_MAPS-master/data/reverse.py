import jsonlines

data = []
with jsonlines.open('all_bsl.jsonl') as f:
    for i in f:
        data.append(i)

dedup={}
for i in data:
    if i['focal_method_with_context'] not in dedup:
        dedup[i['focal_method_with_context']] = [{'project':i['project'], 'id':i['id']}]
    else:
        dedup[i['focal_method_with_context']].append({'project':i['project'], 'id':i['id']})

processed_data = []
for key in dedup:
    processed_data.append({'method':key, 'projects':dedup[key]})

with jsonlines.open('all_bsl.jsonl','w') as f:
    f.write_all(processed_data)



data = []
with jsonlines.open('sample_bsl.jsonl') as f:
    for i in f:
        data.append(i)

dedup={}
for i in data:
    if i['focal_method_with_context'] not in dedup:
        dedup[i['focal_method_with_context']] = [{'project':i['project'], 'id':i['id']}]
    else:
        dedup[i['focal_method_with_context']].append({'project':i['project'], 'id':i['id']})

processed_data = []
for key in dedup:
    processed_data.append({'method':key, 'projects':dedup[key]})

with jsonlines.open('sample_bsl.jsonl','w') as f:
    f.write_all(processed_data)


    

data = []
with jsonlines.open('all_ours.jsonl') as f:
    for i in f:
        data.append(i)

dedup={}
for i in data:
    if i['focal_method_with_context'] not in dedup:
        dedup[i['focal_method_with_context']] = [{'project':i['project'], 'id':i['id']}]
    else:
        dedup[i['focal_method_with_context']].append({'project':i['project'], 'id':i['id']})

processed_data = []
for key in dedup:
    processed_data.append({'method':key, 'projects':dedup[key]})

with jsonlines.open('all_ours.jsonl','w') as f:
    f.write_all(processed_data)



    

data = []
with jsonlines.open('sample_ours.jsonl') as f:
    for i in f:
        data.append(i)

dedup={}
for i in data:
    if i['focal_method_with_context'] not in dedup:
        dedup[i['focal_method_with_context']] = [{'project':i['project'], 'id':i['id']}]
    else:
        dedup[i['focal_method_with_context']].append({'project':i['project'], 'id':i['id']})

processed_data = []
for key in dedup:
    processed_data.append({'method':key, 'projects':dedup[key]})

with jsonlines.open('sample_ours.jsonl','w') as f:
    f.write_all(processed_data)