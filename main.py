import asyncio
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import aiofiles
from openai import AsyncOpenAI
from dotenv import load_dotenv

from executors import evaluate_rs
from extract_rust import extract_rust_code
from prompts import base_prompt, retry_prompt



load_dotenv()
open_router_key = os.getenv("OPEN_ROUTER_KEY")

filename = "results/2024-11-21b.jsonl"
problems_file = "problems.jsonl"
retry_on_error = True

model = "mistralai/mistral-large-2411"
model = "qwen/qwen-2.5-coder-32b-instruct"
model = "anthropic/claude-3-5-haiku"

model = "anthropic/claude-3.5-sonnet"
#model = "x-ai/grok-beta"
#model = "openai/o1-mini-2024-09-12"
#model = "openai/o1-preview-2024-09-12"
#model = "openai/chatgpt-4o-latest"
#model = "openai/gpt-4o-mini"
#model = "openai/gpt-3.5-turbo-0125"
#model = "mistralai/codestral-mamba"
#model = "google/gemini-flash-1.5" # Tripe backtick, easy
#model = "google/gemini-pro-1.5"
#model = "meta-llama/llama-3.1-70b-instruct", # same
#model = "meta-llama/llama-3.1-405b-instruct"
#model = "nousresearch/hermes-3-llama-3.1-405b", # Tripe backtick, easy
#model = "microsoft/phi-3.5-mini-128k-instruct"
#model = "liquid/lfm-40b:free"
#model = "deepseek/deepseek-chat"

models = ["anthropic/claude-3-5-haiku", "qwen/qwen-2.5-coder-32b-instruct", "openai/gpt-4o-mini"]*3

client = AsyncOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=open_router_key,
)

def get_forbidden_ids(main_model):
    filename = "results/merged.jsonl"
    problem_ids = []
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            model = data["model"]
            problem_id = data["problem_id"]
            if model == main_model:
                problem_ids.append(problem_id)
    return problem_ids
           

async def write_jsonl(filename: str, data: Dict[str, Any]) -> None:
    async with aiofiles.open(filename, mode='a') as file:
        json_line = json.dumps(data) + '\n'
        await file.write(json_line)

async def attempt_solution(model: str, prompt: str, executor: ProcessPoolExecutor) -> tuple[str, str, bool]:
    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    code = extract_rust_code(completion.choices[0].message.content)
    if code is None:
        return None, None, False, False
    error_msg, success = await evaluate_rs(code, executor)
    return code, error_msg, success, True

async def process_one_problem(
    model: str,
    problem_id: str,
    problem: str,
    retry_on_error: bool,
    index: int,
    semaphore: asyncio.Semaphore,
    executor: ProcessPoolExecutor
) -> None:
   try:
       async with semaphore:
           print(f"Starting index {index}")

           prompt = base_prompt + "\n" + problem
           rust_code, error_message, success, code_extracted = await attempt_solution(model, prompt, executor)
           if not code_extracted: return
           if success or not retry_on_error:
               await write_jsonl(filename, {
                   "model": model,
                   "problem_id": problem_id, 
                   "code": rust_code,
                   "error_message": error_message,
                   "success": success,
                   "retry_on_error": retry_on_error
               })
               return

           prompt = retry_prompt.format(
               problem_statement=prompt,
               code=rust_code,
               error_message=error_message
           )
           rust_code, second_error_message, second_success, code_extracted = await attempt_solution(model, prompt, executor)
           if not code_extracted: return
           
           await write_jsonl(filename, {
               "model": model,
               "problem_id": problem_id,
               "code": rust_code,
               "success": success,
               "error_message": error_message,
               "second_error_message": second_error_message,
               "second_success": second_success,
               "retry_on_error": retry_on_error
           })
           
           print(f"Finished index {index}")
           
   except Exception as e:
       print(f"Error processing index {index}: {str(e)}")

async def load_problems(filename: str):
    problems = []
    async with aiofiles.open(filename, 'r') as f:
        async for line in f:
            problem_data = json.loads(line)
            problems.append((problem_data['id'], problem_data['problem_statement']))
    return problems

async def main():
    """Execute model evaluations on problem statements."""
    max_problems = 83
    max_concurrent_tasks = 8  # Limit concurrent tasks

    semaphore = asyncio.Semaphore(max_concurrent_tasks*3)
    problems = await load_problems(problems_file) 
    random.shuffle(problems)

    # Use a ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        tasks = []
        for model in models:
            skip_ids = get_forbidden_ids(model)
            for index, (problem_id, problem) in enumerate(problems[:max_problems]):
                if False and problem_id in skip_ids:
                    print(f"Skipping problem {problem_id}")
                    continue
                task = process_one_problem(model, problem_id, problem, retry_on_error, index, semaphore, executor)
                tasks.append(task)
    
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"Total time: {time.time() - t0}")
