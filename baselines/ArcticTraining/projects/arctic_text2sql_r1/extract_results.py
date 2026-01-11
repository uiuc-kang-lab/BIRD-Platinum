import json

with open("/workspace/dev/arcwise_minidev.json") as f:
    data = json.load(f)

with open("results/infly_inf-rl-qwen-coder-32b-2746_dev_bird/greedy_search_base.json") as f:
    results = json.load(f)

final_results = {d["question_id"]: result["pred_sqls"] for d, result in zip(data, results)}

with open("extracted_results.json", "w") as f:
    json.dump(final_results, f, indent=2)