import asyncio
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import aiofiles
import aiohttp
from openai import AsyncOpenAI
from dotenv import load_dotenv

from executors import evaluate_rs
from extract_rust import extract_rust_code
from prompts import base_prompt, retry_prompt



load_dotenv()
open_router_key = os.getenv("OPEN_ROUTER_KEY")
t00 = time.time()

filename = "results/2024-11-26.jsonl"
problems_file = "problems.jsonl"
retry_on_error = True
provider = "local_llama"

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
#model = "google/gemini-flash-1.5"
#model = "google/gemini-pro-1.5"
#model = "meta-llama/llama-3.1-70b-instruct",
#model = "meta-llama/llama-3.1-405b-instruct"
#model = "nousresearch/hermes-3-llama-3.1-405b",
#model = "microsoft/phi-3.5-mini-128k-instruct"
#model = "liquid/lfm-40b:free"
#model = "liquid/lfm-40b"
#model = "deepseek/deepseek-chat"

models = ["Codestral-22B-v0.1-Q8_0.gguf"]

client = AsyncOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=open_router_key,
)


async def write_jsonl(filename: str, data: Dict[str, Any]) -> None:
    async with aiofiles.open(filename, mode='a') as file:
        json_line = json.dumps(data) + '\n'
        await file.write(json_line)

async def make_completion_call(prompt: str, provider: str, model: str = None, **kwargs) -> str:
    """
    Make API call to specified provider.
    
    Args:
        prompt: Input prompt
        provider: API provider ("open_router" or "local_llama")
        model: Model identifier (required for open_router, ignored for local_llama)
        **kwargs: Additional arguments for the API call (only used for open_router)
    """
    if provider == "open_router":
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return completion.choices[0].message.content
    elif provider == "local_llama":
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8080/completion",
                json={"prompt": prompt}
            ) as response:
                result = await response.json()
                return result["content"]
    else:
        raise ValueError(f"Unsupported provider: {provider}")

async def attempt_solution(
    model: str,
    prompt: str, 
    executor: ProcessPoolExecutor, 
    provider: str = "open_router",
    **kwargs
) -> tuple[str, str, bool, bool]:
    """
    Attempt to generate and evaluate a solution using specified provider.
    
    Args:
        prompt: Input prompt
        executor: Process pool executor
        provider: API provider ("open_router" or "local_llama")
        model: Model identifier (required for open_router, ignored for local_llama)
        **kwargs: Additional arguments for the API call (only used for open_router)
        
    Returns:
        tuple containing (code, error_message, success, attempted)
    """
    try:
        content = await make_completion_call(prompt, provider, model, **kwargs)
        code = extract_rust_code(content)
        if code is None:
            print(f"content = {content}")
            return None, None, False, False
            
        error_msg, success = await evaluate_rs(code, executor)
        return code, error_msg, success, True
        
    except Exception as e:
        return None, str(e), False, False

async def process_one_problem(
    model: str,
    problem_id: str,
    problem: str,
    retry_on_error: bool,
    index: int,
    semaphore: asyncio.Semaphore,
    executor: ProcessPoolExecutor,
    **kwargs
) -> None:
    try:
        async with semaphore:
            print(f"Starting index {index}")

            prompt = base_prompt + "\n" + problem
            rust_code, error_message, success, code_extracted = await attempt_solution(model, prompt, executor, **kwargs)
            if not code_extracted:
                return

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

            print(f"Retrying index {index}, time {time.time() - t00}")
            prompt = retry_prompt.format(
                problem_statement=prompt,
                code=rust_code,
                error_message=error_message
            )
            rust_code, second_error_message, second_success, code_extracted = await attempt_solution(model, prompt, executor, **kwargs)
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
    max_problems = 20
    max_concurrent_tasks = 1  # Limit concurrent tasks
    max_process_tasks = 8

    if provider == "local_llama" and len(models) > 1:
        raise ValueError("Local LLAMA only supports one model at a time")

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    problems = await load_problems(problems_file) 
    random.shuffle(problems)

    # Use a ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_process_tasks) as executor:
        tasks = []
        for model in models:
            for index, (problem_id, problem) in enumerate(problems[:max_problems]):
                task = process_one_problem(model, problem_id, problem, retry_on_error, index, semaphore, executor, provider = provider, temperature=0.5)
                tasks.append(task)
    
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"Total time: {time.time() - t0}")
