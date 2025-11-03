import json


with open("/workspace/dev/arcwise_minidev.json") as f:
    data = json.load(f)

with open("outputs/20251023_063131/sampling_think_sql_generate_pred_major_voting_sqls.json") as f:
    results = json.load(f)

final_results = {d["question_id"]: result for d, result in zip(data, results)}


with open("extracted_results.json", "w") as f:
    json.dump(final_results, f, indent=2)