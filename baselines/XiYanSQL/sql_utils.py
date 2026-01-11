"""
SynSQL reward calculation.

- Format reward: <think>...</think> <solution>...</solution>
- Outcome reward: check ground truth and predicted patch similarity
"""

import re
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
import sys, os
import random
from copy import deepcopy

THINK_START, THINK_END = "<think>", "</think>"
SQL_START, SQL_END = "<sql>", "</sql>"
SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"
OBS_START, OBS_END = "<observation>", "</observation>"


# NOTE: bring back reward
def verify_format_and_extract(output: str):
    # if output.count(SOLUTION_START) != 1:
    #     return False, None, None, None
    tail = output.split(SOLUTION_START, 1)[-1]

    if tail.count(SOLUTION_END) != 1:
        return False, None, None, None

    solution_text, _ = tail.split(SOLUTION_END, 1)

    if re.search(r"</?(think|sql|observation)\b", solution_text, re.I):
        return False, None, None, None

    # thoughts = re.findall(r"<think>(.*?)</think>", output, re.S)
    # if not thoughts:
    #     return False, None, None, None

    # for m in re.finditer(r"</observation>", pre_solution, re.I):
    #     rest = pre_solution[m.end() :].lstrip()
    #     if not rest.lower().startswith(THINK_START):
    #         return False, None, None, None

    return True, None, solution_text.strip(), None

def execute_sql_single(db_file, sql):
    try:
        conn = sqlite3.connect(db_file)
        # db_id = db_file.split("/")[-1].replace(".sqlite", "")
        # sql_commands = None
        # if os.path.exists(f"/data/yuxuan_zhu/noisy-rl/BIRD-Platinum/db_modification_eval/modify_{db_id}.sql"):
        #     with open(f"/data/yuxuan_zhu/noisy-rl/BIRD-Platinum/db_modification_eval/modify_{db_id}.sql", "r") as f:
        #         sql_commands = f.read()
        cursor = conn.cursor()
        conn.execute("BEGIN TRANSACTION;")
        # if sql_commands is not None:
        #     print(f"running db modifications: {sql_commands}")
        #     cursor.executescript(sql_commands)
        #     print(f"finished db modifications")
        cursor.execute(sql)
        execution_res = deepcopy(cursor.fetchall())
        conn.rollback()
        conn.close()
        # print('Successfully executed')
        return db_file, sql, execution_res, 1
    except Exception as e:
        print(f"Error executing SQL: {e}, db file: {db_file}, SQL: {sql}")
        conn.rollback()
        conn.close()
        return db_file, sql, None, f"Error executing SQL: {e}, db file: {db_file}, SQL: {sql}"


def execute_sql_wrapper_single(db_file, sql, timeout, output_str):
    try:
        res = func_timeout(timeout, execute_sql_single, args=(db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"SQL:\n{sql}\nTime Out ({timeout}s)!")
        print("-" * 30)
        res = (db_file, sql, None, f"SQL:\n{sql}\nTime Out!")
    except Exception as e:
        print(f"Error executing SQL: {e}, db_file: {db_file}, SQL: {sql}")
        res = (db_file, sql, None, f"Error executing SQL: {e}, db_file: {db_file}")

    # Append the output to the tuple
    if isinstance(res, tuple):
        res = res + (output_str,)

    return res