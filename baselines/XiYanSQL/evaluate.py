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
        return False, {"message": f"Error executing ground truth query: {e}"}

    try:
        cursor.execute(generated_query)
        generated_result = cursor.fetchall()
    except Exception as e:
        cursor.close()
        conn.close()
        return False, {"message": f"Error executing generated query: {e}"}
    
    cursor.close()
    conn.close()

    if not grade_basic(ground_truth_result, generated_result):
        return False, {"message": "Basic grading failed: number of columns does not match"}

    if "multiset" in grading_method:
        return grade_multiset(ground_truth_result, generated_result)
    elif "subset" in grading_method:
        _, method, row_count = grading_method.split(",")
        if method == "=":
            return grade_subset(ground_truth_result, generated_result, strict_row_count=int(row_count))
        elif method == ">=":
            return grade_subset(ground_truth_result, generated_result, minimum_row_count=int(row_count))
        else:
            return False, {"message": f"Unknown subset method: {method}"}
    else:
        return False, {"message": f"Unknown grading method: {grading_method}"}

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
    with open(args.data_path, 'r') as f:
        raw_data = json.load(f)
        for item in raw_data:
            question_id2db_id[item['question_id']] = item['db_id']
    
    total = len(infer_results)
    correct = 0
    detailed_results = []
    for res in tqdm(infer_results):
        qid = res['question_id']
        gt = res['ground_truth']
        pred = res['prediction']
        db_id = question_id2db_id[qid]
        db_path = f"{args.db_base_path}/{db_id}/{db_id}.sqlite"
        is_correct, info = grade(db_path, gt, pred, grading_method="multiset")
        if is_correct:
            correct += 1
        detailed_results.append({
            "question_id": qid,
            "db_id": db_id,
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": is_correct,
            "info": info
        })
    accuracy = correct / total if total > 0 else 0.0
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.4f}")