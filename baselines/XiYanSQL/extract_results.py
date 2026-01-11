import json

with open("results/dev500_original.json") as f:
    results = json.load(f)

extracted_results = {x["question_id"]: x["prediction"] for x in results}

with open("results/dev500_extracted.json", "w") as f:
    json.dump(extracted_results, f, indent=4)