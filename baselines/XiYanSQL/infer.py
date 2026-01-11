from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json, re
import argparse
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut
from sql_utils import verify_format_and_extract, execute_sql_wrapper_single

nl2sqlite_template_cn = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

nl2sqlite_template_multiturn_cn = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

说明：
- 请确保仅输出问题中要求的信息。若问题要求特定列，请务必在SELECT子句中仅包含该列，不得添加其他内容。
- 生成的查询语句应完整返回问题中要求的所有信息，不得遗漏或添加额外内容。
- 在生成最终SQL查询前，请系统规划编写步骤。该过程需包含详细考量：分析问题要点、
归纳相关发现、构思新思路、验证当前步骤准确性、修正潜在错误、规划SQL工具调用方式，并复盘前期步骤。


格式规范：
- 每次获得新观察结果或信息时，请在<think>...</think>块内进行思考。
- 可使用单个<sql>your sql</sql>块内的SQL工具进行探索或验证。SQL工具输出将以数据框形式显示在<observation>...</observation>内。基于此观察结果，可再次思考并优化。
- 若观察结果过长，返回的数据框将截断为50行。
- 若确认无需进一步探索或达到最大轮次，必须直接在<solution>...</solution>内提供最终SQL查询方案。

----------------------- - 示例开始 ------------------------
【用户问题】
how many pigs are in the farm? 

【数据库schema】
【DB_ID】 debit_card_specializing
【Schema】
# Table: main.animals
[
(id:INTEGER, Primary Key, Examples: [3, 5, 6]),
(species:TEXT, Examples: [chicken, pig, cow]),
(age:TEXT, Examples: [2, 3, 4]),
(name:TEXT, Examples: [Jack, Mike, Tom])]
【Foreign keys】

【参考信息】
Pig is a species of domesticated animal commonly found on farms.

【用户问题】
how many pigs are in the farm? 

<think>I am querying how many pigs are in the farm. I will begin by checking if the 'animals' table exists and contains entries with species = 'pig'.</think>
<sql>SELECT COUNT(*) FROM animals WHERE species = 'pig';</sql>
<observation>
+----------+
| COUNT(*) |
+----------+
|   12     |
+----------+
</observation>
<think>The result indicates that there are 12 pigs in the farm. Since the question asks for how many pigs, I can now output the final SQL as the solution.</think>
<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>
------------------------ 示例结束 ------------------------

"""

error_prompt = "SQL execution error: {error}"
result_prompt = "SQL execution results: {result}"
long_result_prompt = "Truncated to 50 lines since returned response too long: {result}"
multiturn_prompt = "You have {n_turn_left} turns left to complete the task."
invalid_format_prompt = "Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."

def prepare_prompts(data: list, db_base_path: str, n_data: int = -1, schemas: dict = None, prompt_style: str = "xiyan") -> list:
    prompts = []
    ground_truths = []
    question_ids = []
    db_ids = []
    for i, correction in enumerate(data):
        if n_data != -1 and i >= n_data:
            break
        db_id = correction['db_id']
        question = correction['question']
        evidence = correction['evidence']
        question_id = correction['question_id']
        SQL = correction['SQL']
        schema, dialect = schemas[db_id]['schema'], schemas[db_id]['dialect']
        if prompt_style == "xiyan":
            prompt = nl2sqlite_template_cn.format(
                dialect=dialect, 
                db_schema=schema, 
                question=question, 
                evidence=evidence
            )
        elif prompt_style == "skyrl-sql":
            prompt = nl2sqlite_template_multiturn_cn.format(
                dialect=dialect, 
                db_schema=schema, 
                question=question, 
                evidence=evidence
            )
        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")
        prompts.append([{'role': 'user', 'content': prompt}])
        ground_truths.append(SQL)
        question_ids.append(question_id)
        db_ids.append(db_id)
    return prompts, ground_truths, question_ids, db_ids


def get_next_turn_skyrl(generation: str, db_file: str):
    if generation.endswith('</sql>'):
        match = re.search(r"<sql>(.*?)</sql>", generation, re.DOTALL)
        sql_query = match.group(1) if match else None
        sql_output = execute_sql_wrapper_single(db_file, sql_query, 30, generation)
        _, _, pred_results, error, _ = sql_output
        if pred_results is None:
            print(f"SQL execution error: {error}")
            return error, False, "Time Out!" in error
        else:
            df = pd.DataFrame(pred_results)
            res = df.to_string(index=False)
            if len(res) > 9000:
                # just truncate
                truncated_df = df.head(50)
                res = "Truncated to 50 lines since returned response too long: " + truncated_df.to_string(
                    index=False
                )  # or index=True if you want row numbers
            else:
                res = "SQL execution results: " + res


            return res, False, False
    else:
        is_valid, _, sql, _ = verify_format_and_extract(generation)
        if not is_valid:
            return invalid_format_prompt, False, False
        return sql, True, False
    
def infer(raw_data, db_base_path: str, n_data: int = -1, schemas: dict = None, output_path: str = "xiyan_eval_results.json", prompt_style: str = "xiyan"):
    prompts, ground_truths, question_ids, db_ids = prepare_prompts(
        data=raw_data, db_base_path=db_base_path, n_data=n_data, schemas=schemas, prompt_style=prompt_style)
    

    model_path = "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(
        n=1,
        temperature=0.1,
        max_tokens=1024
    )

    done_all = [False for _ in range(len(prompts))]
    queries = [None for _ in range(len(prompts))]
    n_output_tokens_all = 0
    n_input_tokens_all = 0
    llm = LLM(model=model_path, tensor_parallel_size=8)
    if prompt_style == "skyrl-sql":
        for i in range(5): # max 5 turns

            texts = []
            for prompt, done in zip(prompts, done_all):
                if done:
                    continue
                text = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)

            print(f"Number of examples: {len(texts)}")

            outputs = llm.generate(texts, sampling_params=sampling_params)
            
            generations = [
                output.outputs[0].text for output in outputs
            ]

            n_output_tokens = [
                len(output.outputs[0].token_ids) for output in outputs
            ]
            n_output_tokens_all += sum(n_output_tokens)
            n_input_tokens_all += sum([len(output.prompt_token_ids) for output in outputs])

            not_done_idx = [j for j, done in enumerate(done_all) if not done]
            n_timeouts = 0
            executions = 0

            for j, generation in zip(not_done_idx, generations):
                obs, done, timeout = get_next_turn_skyrl(
                    generation=generation, 
                    db_file=f"{db_base_path}/{db_ids[j]}/{db_ids[j]}.sqlite"
                )
                if generation.endswith('</sql>'):
                    executions += 1
                if timeout:
                    n_timeouts += 1
                if not done:
                    obs += "\n" + multiturn_prompt.format(n_turn_left=4 - i)
                    prompts[j] += [
                        {'role': 'assistant', 'content': generation},
                        {'role': 'user', 'content': obs}
                    ]
                else:
                    done_all[j] = True
                    queries[j] = obs

            print("---- Turn {} completed ----".format(i+1))
            print(f"# questions done: {sum(done_all)}/{len(done_all)}")
            print(f"Input tokens: {n_input_tokens_all}")
            print(f"Output tokens: {n_output_tokens_all}")
            print(f"Executed SQLs: {executions}, Timeouts: {n_timeouts}")
            if all(done_all):
                break
    elif prompt_style == "xiyan":
        texts = []
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        print(f"Number of examples: {len(texts)}")

        outputs = llm.generate(texts, sampling_params=sampling_params)
        
        generations = [
            output.outputs[0].text for output in outputs
        ]

        n_output_tokens = [
            len(output.outputs[0].token_ids) for output in outputs
        ]
        n_output_tokens_all += sum(n_output_tokens)
        n_input_tokens_all += sum([len(output.prompt_token_ids) for output in outputs])

        for j, generation in enumerate(generations):
            queries[j] = generation

        print("---- Inference completed ----")
        print(f"Input tokens: {n_input_tokens_all}")
        print(f"Output tokens: {n_output_tokens_all}")
    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    results = []
    for qid, gt, pred, prompt in zip(question_ids, ground_truths, queries, prompts):
        results.append({
            'question_id': qid,
            'ground_truth': gt,
            'prediction': pred,
            'prompt': prompt[0]['content']
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/bird_minidev.json", help="Path to the input data file.")
    parser.add_argument("--db_base_path", type=str, default="data/bird_dbs", help="Base path to the database files.")
    parser.add_argument("--n_data", type=int, default=-1, help="Number of data samples to process. Use -1 for all.")
    parser.add_argument("--schema_path", type=str, default="schemas.json", help="Path to the pre-constructed schema file.")
    parser.add_argument("--output_path", type=str, default="xiyan_eval_results.json", help="Path to save the output results.")
    parser.add_argument("--prompt_style", type=str, default="xiyan", help="Prompt style to use")
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        raw_data = json.load(f)

    with open(args.schema_path, 'r') as f:
        schemas = json.load(f)

    infer(raw_data, db_base_path=args.db_base_path, n_data=args.n_data, schemas=schemas, output_path=args.output_path, prompt_style=args.prompt_style)