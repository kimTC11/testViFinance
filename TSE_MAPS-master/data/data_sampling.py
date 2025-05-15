import jsonlines
import random

D4J_BUG_IDS = {
    "Chart": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    "Cli": [1, 3, 4, 7, 8, 9, 10, 11, 13, 14, 15, 16, 21, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40],
    "Csv": [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "Gson": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "Lang": [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64]
}

# Sampled results in this paper
# D4J_BUG_IDS = {"Chart":[6, 21],
#                "Cli":[30, 37],
#                "Csv":[4, 16],
#                "Gson":[12],
#                "Lang":[21, 34, 35]}

def sample_bug_ids(bug_ids, total_sample_size):
    all_ids = [(project, bug_id) for project, ids in bug_ids.items() for bug_id in ids]
    sampled_pairs = random.sample(all_ids, total_sample_size)
    
    sampled_bug_ids = {}
    for project, bug_id in sampled_pairs:
        if project not in sampled_bug_ids:
            sampled_bug_ids[project] = []
        sampled_bug_ids[project].append(bug_id)

    return sampled_bug_ids

sampled_ids = sample_bug_ids(D4J_BUG_IDS, 10)
print(sampled_ids)

data = []
with jsonlines.open('all_bsl.jsonl') as f:
    for obj in f:
        if obj['id'] in sampled_ids[obj['project']]:
            data.append(obj)


with jsonlines.open('sample_bsl', 'w') as f:
    f.write_all(data)

