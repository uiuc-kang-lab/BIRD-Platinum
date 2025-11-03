import json

with open("/workspace/dev/arcwise_minidev.json") as f:
    data = json.load(f)

with open("/workspace/text_to_sql_benchmarks/text_to_sql_agents/OmniSQL/train_and_evaluate/results/OmniSQL-32B_dev_bird/greedy_search_.json") as f:
    results = json.load(f)

final_results = {d["question_id"]: result["pred_sqls"] for d, result in zip(data, results)}
with open("extracted_results.json", "w") as f:
    json.dump(final_results, f, indent=2)