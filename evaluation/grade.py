import sqlite3
from tqdm import tqdm

def grade_basic(result1, result2):
    # should have same number of columns:
    if len(result1) == 0:
        if len(result2) == 0:
            return True
        else:
            return False
    elif len(result2) == 0:
        return False
    if len(result1[0]) == len(result2[0]):
        return True

    return False

def grade_multiset(result1, result2):
    # check if length matches
    if len(result1) != len(result2):
        return False, {"message": "Number of rows do not match"}
    # sort the tuple in each row
    sorted_result1 = [tuple(sorted([str(r) for r in row])) for row in result1]
    sorted_result2 = [tuple(sorted([str(r) for r in row])) for row in result2]
    # check if the sorted results match as multisets
    if sorted(sorted_result1) != sorted(sorted_result2):
        return False, {"message": "Results do not match as multisets"}
    return True, {"message": "Results match as multisets"}

def grade_subset(all_results, matching_results, strict_row_count: int = None, minimum_row_count: int = None):
    assert strict_row_count is None or minimum_row_count is None, "Only one of strict_row_count or minimum_row_count should be provided"
    if strict_row_count is not None:
        if len(matching_results) != strict_row_count:
            return False, {"message": f"Number of matching rows {len(matching_results)} does not equal required {strict_row_count}"}
    if minimum_row_count is not None:
        if len(matching_results) < minimum_row_count:
            return False, {"message": f"Number of matching rows {len(matching_results)} is less than required minimum {minimum_row_count}"}
    # check if all matching_results are in all_results
    set_all = set([tuple(sorted(row)) for row in all_results])
    set_matching = set([tuple(sorted(row)) for row in matching_results])
    if not set_matching.issubset(set_all):
        return False, {"message": "Not all matching rows are in the full result set"}
    return True, {"message": "All matching rows are in the full result set"}

def grade(db_path, ground_truth_query, generated_query, grading_method: str = "multiset"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(ground_truth_query)
        ground_truth_result = cursor.fetchall()
    except Exception as e:
        cursor.close()
        conn.close()
        print(f"Error executing ground truth query: {e}")
        return False, False

    try:
        cursor.execute(generated_query)
        generated_result = cursor.fetchall()
    except Exception as e:
        cursor.close()
        conn.close()
        return False, False
    
    cursor.close()
    conn.close()

    bird_grading = False

    if set(ground_truth_result) == set(generated_result):
        bird_grading = True

    if not grade_basic(ground_truth_result, generated_result):
        # return False, {"message": "Basic grading failed: number of columns does not match"}
        return bird_grading, False

    if "multiset" in grading_method:
        return bird_grading, grade_multiset(ground_truth_result, generated_result)[0]
    elif "subset" in grading_method:
        _, method, row_count = grading_method.split(",")
        if method == "=":
            return bird_grading, grade_subset(ground_truth_result, generated_result, strict_row_count=int(row_count))[0]
        elif method == ">=":
            return bird_grading, grade_subset(ground_truth_result, generated_result, minimum_row_count=int(row_count))[0]
        else:
            return bird_grading, False #, {"message": f"Unknown subset method: {method}"}
    else:
        return bird_grading, False #, {"message": f"Unknown grading method: {grading_method}"}

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_base_path", type=str, required=True, help="Path to the SQLite database file")
    parser.add_argument("--infer_results", type=str, required=True, help="Path to the JSON file containing inference results")
    parser.add_argument("--data_path", type=str, default="data/bird_minidev.json", help="Path to the input data file.")
    args = parser.parse_args()

    with open(args.infer_results, 'r') as f:
        infer_results = json.load(f)

    print(f"Number of inference results: {len(infer_results)}")
    question_id2db_id = {}
    question_id2gt = {}
    with open(args.data_path, 'r') as f:
        raw_data = json.load(f)
        for item in raw_data:
            question_id2db_id[str(item['question_id'])] = item['db_id']
            question_id2gt[str(item['question_id'])] = item['SQL']
    
    total = len(raw_data)
    bird_correct = 0
    strict_correct = 0
    detailed_results = []
    for qid in tqdm(infer_results):
        gt = question_id2gt[qid]
        pred = infer_results[qid][0] if isinstance(infer_results[qid], list) else infer_results[qid]
        db_id = question_id2db_id[qid]
        db_path = f"{args.db_base_path}/{db_id}/{db_id}.sqlite"
        bird_grading, strict_grading = grade(db_path, gt, pred, grading_method="multiset")
        if bird_grading:
            bird_correct += 1
        if strict_grading:
            strict_correct += 1
        detailed_results.append({
            "question_id": qid,
            "db_id": db_id,
            "ground_truth": gt,
            "prediction": pred,
            "bird_grading": bird_grading,
            "strict_grading": strict_grading
        })
    bird_accuracy = bird_correct / total
    strict_accuracy = strict_correct / total

    method_name = args.infer_results.split('/')[1].replace('.json', '')

    print(f"Evaluation results of {method_name}:")
    print(f"Bird Accuracy: {bird_accuracy:.4f} ({bird_correct}/{total})")
    print(f"Strict Accuracy: {strict_accuracy:.4f} ({strict_correct}/{total})")

    with open(f"reports/{method_name}_evaluation.json", 'w') as f:
        json.dump({
            "bird_accuracy": bird_accuracy,
            "strict_accuracy": strict_accuracy,
            "detailed_results": detailed_results
        }, f, indent=4)
