import asyncio
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional
import argparse

import aiofiles
import aiohttp
from openai import AsyncOpenAI
from dotenv import load_dotenv

from executors import evaluate_rs
from extract_rust import extract_rust_code
from prompts import base_prompt, retry_prompt

# Define arguments for the script
ap = argparse.ArgumentParser(description="Rust code generation and evaluation benchmark.")
ap.add_argument("--eval_existing", type=str, help="Path to a JSONL file with existing solutions to evaluate.")
ap.add_argument("--output", type=str, help="Path to the output JSONL file for results.")
ap.add_argument("--actual_model_id", type=str, help="The ID of the model that generated the solutions (used with --eval_existing).")
ap.add_argument("--max_problems", type=int, default=100, help="Maximum number of problems to process.")
ap.add_argument("--max_concurrent_tasks", type=int, default=8, help="Limit concurrent LLM API calls.")
ap.add_argument("--max_process_tasks", type=int, default=8, help="Max worker processes for Rust evaluation.")
ap.add_argument("--retry_on_error", action="store_true", help="Retry failed generations once.")
ap.add_argument("--provider", type=str, default="local_llama", help="API provider ('open_router' or 'local_llama').")
ap.add_argument("--model", type=str, help="Specific model identifier for API calls (e.g., 'mistralai/mistral-large-2411').") # For generation mode
ap.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature for generation.")
# Add other sampling parameters if needed for generation mode
# ap.add_argument("--max_tokens", type=int, default=1536)
# ap.add_argument("--top_p", type=float, default=0.95)
args = ap.parse_args()

# Use arguments for configuration
filename = args.output if args.output else "results/default_output.jsonl" # Default if not provided
problems_file = "problems.jsonl" # Still hardcoded, but can be made an arg too if needed
retry_on_error = args.retry_on_error
provider = args.provider

# Models list for generation mode (not used with --eval_existing)
# If --model is provided, use that, otherwise use a default list
if args.model:
    models = [args.model]
else:
    models = ["model"] # Default for generation if no --model is specified


load_dotenv()
open_router_key = os.getenv("OPEN_ROUTER_KEY")
t00 = time.time()

# Conditionally initialize the client based on the 'provider' setting
client = None
if provider == "open_router":
    client = AsyncOpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=open_router_key,
    )


async def write_jsonl(filename: str, data: Dict[str, Any]) -> None:
    async with aiofiles.open(filename, mode='a') as file:
        json_line = json.dumps(data) + '\n'
        await file.write(json_line)

async def make_completion_call(prompt: str, provider: str, model: str = "local", **kwargs) -> str:
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
    model: str, # This 'model' param will now be the actual_model_id in eval mode
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
        # Only make API call if not in evaluation mode and client is available
        if args.eval_existing: # If evaluating existing, no generation call needed
            # This branch should not be reached if --eval_existing is used
            # because process_one_problem will read from file directly.
            # However, if it somehow is, we need to handle it.
            raise RuntimeError("[RUST-BENCH]: Attempted to generate solution in --eval_existing mode.")

        # If not evaluating existing, proceed with generation
        if client is None and provider == "open_router":
            raise ValueError("[RUST-BENCH]: OpenRouter client not initialized. Check API key or provider setting.")

        content = await make_completion_call(prompt, provider, model, **kwargs)

        code = extract_rust_code(content)
        if code is None:
            print(f"[RUST-BENCH]: content = {content}")
            return None, None, False, False

        error_msg, success = await evaluate_rs(code, executor)
        return code, error_msg, success, True

    except Exception as e:
        return None, str(e), False, False

async def process_one_problem(
    model: str, # This is now the 'actual_model_id' or 'model_from_list'
    problem_id: str,
    problem: str,
    retry_on_error_single: bool,
    index: int,
    semaphore: asyncio.Semaphore,
    executor: ProcessPoolExecutor,
    **kwargs
) -> None:
    try:
        async with semaphore:
            print(f"[RUST-BENCH]: Starting index {index}")

            rust_code = None
            error_message = None
            success = False
            code_extracted = False

            if args.eval_existing:
                # In evaluation mode, read solutions from the input file
                # The 'problem' here is actually the entire entry from bench1_generated_solutions.jsonl
                # We need to parse it to get the code and the original model_used_id

                # Assume 'problem' is the dict loaded from problems.jsonl
                # The actual code to evaluate is in problem['solution'] from the input file
                # The model_id is in problem['model_used_id']

                # The 'problem' argument passed to process_one_problem when --eval_existing is used
                # is actually the full parsed JSON object from bench1_generated_solutions.jsonl.
                # So we need to extract `solution` and `model_used_id` from it.

                # The `problem` variable in this function is actually the `problem_data`
                # from `bench1_formatted_for_eval.jsonl` when `eval_existing` is true.
                # It contains 'code' and 'model_used_id' already.

                # Corrected logic for --eval_existing mode:
                # The `problem` argument passed to `process_one_problem` is already the
                # full dictionary from the formatted evaluation file.

                # This `model` argument is the `actual_model_id` passed from `runbmo`
                # when in `--eval_existing` mode, or the model from the `models` list
                # when in generation mode.

                # We need to ensure that the `model` in the output JSONL is the one
                # that *generated* the solution, which is `problem['model_used_id']`
                # from the input file.

                # For `eval_existing`, the `problem` variable is actually the
                # parsed dict from `bench1_formatted_for_eval.jsonl`, which already
                # contains the 'code' and 'problem_id' and 'model_used_id'.
                # The `model` parameter to this function will be `args.actual_model_id`.

                # Use the 'code' from the input file directly
                rust_code = problem.get("solution")
                problem_id_from_file = problem.get("task_id") # Get problem_id from file
                model_used_in_generation = problem.get("model_used_id", model) # Use model_used_id from file, fallback to current model param

                if rust_code is None:
                    print(f"[RUST-BENCH]: Skipping problem {problem_id_from_file}: No code found in input file.")
                    return # Skip if no code to evaluate

                # Evaluate the existing code
                error_message, success = await evaluate_rs(rust_code, executor)
                code_extracted = True # Code was extracted from the input file

                await write_jsonl(args.output, { # Use args.output for the filename
                    "model": model_used_in_generation, # Use the model that generated it
                    "problem_id": problem_id_from_file,
                    "code": rust_code,
                    "error_message": error_message,
                    "success": success,
                    "retry_on_error": False, # No retry in eval_existing mode
                    "evaluation_only": True
                })
                print(f"[RUST-BENCH]: Finished evaluation for {problem_id_from_file}")
                return # Done with this problem in eval mode

            else: # Original generation mode
                prompt_text = base_prompt + "\n" + problem
                rust_code, error_message, success, code_extracted = await attempt_solution(model, prompt_text, executor, **kwargs)
                if not code_extracted:
                    return

                if success or not retry_on_error_single:
                    await write_jsonl(filename, {
                        "model": model,
                        "problem_id": problem_id,
                        "code": rust_code,
                        "error_message": error_message,
                        "success": success,
                        "retry_on_error": retry_on_error_single
                    })
                    print(f"[RUST-BENCH]: Finished index {index}")
                    return

                print(f"[RUST-BENCH]: Retrying index {index}, time {time.time() - t00}")
                prompt_text = retry_prompt.format(
                    problem_statement=prompt_text,
                    code=rust_code,
                    error_message=error_message
                )
                rust_code, second_error_message, second_success, code_extracted = await attempt_solution(model, prompt_text, executor, **kwargs)
                if not code_extracted: return

                await write_jsonl(filename, {
                    "model": model,
                    "problem_id": problem_id,
                    "code": rust_code,
                    "success": success,
                    "error_message": error_message,
                    "second_error_message": second_error_message,
                    "second_success": second_success,
                    "retry_on_error": retry_on_error_single
                })
                print(f"[RUST-BENCH]: Finished index {index}")

    except Exception as e:
        print(f"[RUST-BENCH]: Error processing index {index}: {str(e)}")

async def load_problems(file_name: str, eval_existing: Optional[str] = None):
    problems = []
    async with aiofiles.open(file_name, 'r') as f:
        async for line in f:
            problem_data = json.loads(line)
            if eval_existing:
                # When evaluating existing, the 'problem' is the entire entry from the solution file
                problems.append(problem_data)
            else:
                # When generating, extract id and problem statement
                problems.append((problem_data['id'], problem_data['problem_statement']))
    return problems

async def main_async():
    """Execute model evaluations on problem statements."""

    # Use arguments for configuration, overriding hardcoded values
    max_problems = args.max_problems
    max_concurrent_tasks = args.max_concurrent_tasks
    max_process_tasks = args.max_process_tasks

    if args.eval_existing:
        # When evaluating existing solutions, load from the provided file
        problems_to_process = await load_problems(args.eval_existing, eval_existing=Non)
        # In eval mode, we only process problems from the eval_existing file
        # The 'model' to pass to process_one_problem is the one provided via --actual_model_id
        models_to_iterate = [args.actual_model_id] if args.actual_model_id else ["unknown_model_id"]
        # Limit to max_problems if specified
        problems_to_process = problems_to_process[:max_problems]

    else:
        # When generating new solutions, load from problems_file
        if provider == "local_llama" and len(models) > 1:
            raise ValueError("[RUST-BENCH]: Local LLAMA only supports one model at a time when generating.")
        problems_to_process = await load_problems(problems_file)
        random.shuffle(problems_to_process)
        problems_to_process = problems_to_process[:max_problems]
        models_to_iterate = models # Use the internal 'models' list for generation

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # Use a ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_process_tasks) as executor:
        tasks = []
        for model_id_for_loop in models_to_iterate: # Iterate over the correct models list
            for index, problem_entry in enumerate(problems_to_process):
                # If eval_existing, problem_entry is a dict. If generating, it's (id, statement)
                if args.eval_existing:
                    # In eval_existing mode, problem_entry is the full dict from the solutions file
                    # We pass the original problem_id from the file, but the model_id_for_loop
                    # is the --actual_model_id we want to use for the output report.
                    task = process_one_problem(
                        model_id_for_loop, # This will be the model ID for the output report
                        problem_entry.get('problem_id'), # Original problem ID
                        problem_entry, # Pass the whole dict as 'problem'
                        retry_on_error,
                        index,
                        semaphore,
                        executor,
                        provider = provider,
                        temperature=args.temperature # Pass through other args
                    )
                else:
                    # Original generation mode
                    problem_id, problem_statement = problem_entry
                    task = process_one_problem(
                        model_id_for_loop,
                        problem_id,
                        problem_statement,
                        retry_on_error,
                        index,
                        semaphore,
                        executor,
                        provider = provider,
                        temperature=args.temperature # Pass through other args
                        # Add other sampling params here if they are args
                        # max_tokens=args.max_tokens, top_p=args.top_p
                    )
                tasks.append(task)

        await asyncio.gather(*tasks)

if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main_async())
    print(f"[RUST-BENCH]: Total time: {time.time() - t0}")