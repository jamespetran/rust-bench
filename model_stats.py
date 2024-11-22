from collections import defaultdict
from scipy.special import comb
import json

def calculate_pass_at_k(n: int, m: int, k: int) -> float:
    """
    n: total number of samples
    m: number of correct samples
    k: number of draws
    """
    if k > n:
        raise ValueError("k should be less than or equal to n")
    
    if m == 0:
        return 0.0
    
    if n-m <= k:
        return 1.0
    
    # Probability of NOT drawing any correct solutions
    p_fail = comb(n - m, k) / comb(n, k)
    
    # Probability of drawing at least one correct solution
    return 1 - p_fail

pass_3_models = ["anthropic/claude-3.5-sonnet", "qwen/qwen-2.5-coder-32b-instruct", "openai/gpt-4o-mini", "anthropic/claude-3-5-haiku", "google/gemini-flash-1.5"]
pass_7_models = ["qwen/qwen-2.5-coder-32b-instruct", "openai/gpt-4o-mini", "anthropic/claude-3-5-haiku"]

stats_first = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [total, successes]
stats_retry = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # [total, successes_first, successes_second]

filename = "results/merged.jsonl"
problem_file = "problems.jsonl"
with open(filename, "r") as f:
    for line in f:
        data = json.loads(line)
        model = data["model"]
        problem_id = data["problem_id"]
        stats_first[model][problem_id][0] += 1  # increment total
        if data["success"]:
            stats_first[model][problem_id][1] += 1  # increment successes

        if data["retry_on_error"]:
            stats_retry[model][problem_id][0] += 1
            if "second_success" in data and data["second_success"]:
                stats_retry[model][problem_id][2] += 1
            elif data["success"]:
                stats_retry[model][problem_id][1] += 1



problem_rates = defaultdict(lambda: [0, 0])  
for model, problem_dict in stats_first.items():
    for problem_id, (total, successes) in problem_dict.items():
        problem_rates[problem_id][0] += total
        problem_rates[problem_id][1] += successes

problem_rates = sorted(problem_rates.items(), key=lambda x: x[1][1]/x[1][0])
hard_problems, easy_problems = problem_rates[:40], problem_rates[40:]
hard_problems = set(list(zip(*hard_problems))[0])
easy_problems = set(list(zip(*easy_problems))[0])

for model, problem_dict in stats_first.items():
    total_problems = len(problem_dict)
    mean_success_rate = 0
    mean_pass_3_rate, hard_3_rate, easy_3_rate = 0, 0, 0
    mean_pass_7_rate, hard_7_rate, easy_7_rate = 0, 0, 0
    total_attempts = 0

    total_hard_problems, success_mean_hard = len(problem_dict.keys() & hard_problems), 0
    total_easy_problems, success_mean_easy = len(problem_dict.keys() & easy_problems), 0

    for problem_id, (total, successes) in problem_dict.items():
        total_attempts += total
        success_rate = (successes / total * 100) if total > 0 else 0
        mean_success_rate += success_rate*(1/total_problems)
        if model in pass_3_models:
            mean_pass_3_rate += calculate_pass_at_k(total, successes, 3)*(1/total_problems)*100
        if model in pass_7_models:
            mean_pass_7_rate += calculate_pass_at_k(total, successes, 7)*(1/total_problems)*100

        if problem_id in hard_problems:
            success_mean_hard += success_rate*(1/total_hard_problems)
            if model in pass_3_models:
                hard_3_rate += calculate_pass_at_k(total, successes, 3)*(1/total_hard_problems)*100
            if model in pass_7_models:
                hard_7_rate += calculate_pass_at_k(total, successes, 7)*(1/total_hard_problems)*100
        elif problem_id in easy_problems:
            success_mean_easy += success_rate*(1/total_easy_problems)
            if model in pass_3_models:
                easy_3_rate += calculate_pass_at_k(total, successes, 3)*(1/total_easy_problems)*100
            if model in pass_7_models:
                easy_7_rate += calculate_pass_at_k(total, successes, 7)*(1/total_easy_problems)*100

        
    print(f"{model}: {mean_success_rate:.1f}%, hard -> {success_mean_hard:.1f}%, easy -> {success_mean_easy:.1f}%, total attempts: {total_attempts}")
    if model in pass_3_models:
        print(f"Pass at 3: {mean_pass_3_rate:.1f}%, hard -> {hard_3_rate:.1f}%, easy -> {easy_3_rate:.1f}%")
    if model in pass_7_models:
        print(f"Pass at 7: {mean_pass_7_rate:.1f}%, hard -> {hard_7_rate:.1f}%, easy -> {easy_7_rate:.1f}%")
    print("-------------------")

print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
print("")
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
for model, problem_dict in stats_retry.items():
    total_problems = len(problem_dict)
    mean_first_success = 0
    mean_second_success = 0
    mean_pass_3_rate, hard_3_rate, easy_3_rate = 0, 0, 0
    mean_pass_7_rate, hard_7_rate, easy_7_rate = 0, 0, 0
    total_attempts = 0

    total_hard_problems, success_mean_hard = len(problem_dict.keys() & hard_problems), 0
    total_easy_problems, success_mean_easy = len(problem_dict.keys() & easy_problems), 0

    for problem_id, (total, first_successes, second_successes) in problem_dict.items():
        total_attempts += total
        first_succes_rate = (first_successes / total * 100) if total > 0 else 0 
        second_succes_rate = ((second_successes / total * 100) if total > 0 else 0) + first_succes_rate

        mean_first_success += first_succes_rate*(1/total_problems)
        mean_second_success += second_succes_rate*(1/total_problems)

        if model in pass_3_models:
            mean_pass_3_rate += calculate_pass_at_k(total, second_successes+first_successes, 3)*(1/total_problems)*100
        if model in pass_7_models:
            mean_pass_7_rate += calculate_pass_at_k(total, second_successes+first_successes, 7)*(1/total_problems)*100

        if problem_id in hard_problems:
            success_mean_hard += second_succes_rate*(1/total_hard_problems)
            if model in pass_3_models:
                hard_3_rate += calculate_pass_at_k(total, second_successes+first_successes, 3)*(1/total_hard_problems)*100
            if model in pass_7_models:
                hard_7_rate += calculate_pass_at_k(total, second_successes+first_successes, 7)*(1/total_hard_problems)*100
        elif problem_id in easy_problems:
            success_mean_easy += second_succes_rate*(1/total_easy_problems)
            if model in pass_3_models:
                easy_3_rate += calculate_pass_at_k(total, second_successes+first_successes, 3)*(1/total_easy_problems)*100
            if model in pass_7_models:
                easy_7_rate += calculate_pass_at_k(total, second_successes+first_successes, 7)*(1/total_easy_problems)*100

    print(f"{model}: 1st -> {mean_first_success:.1f}%,  2nd ->{mean_second_success:.1f}% | hard -> {success_mean_hard:.1f}%, easy -> {success_mean_easy:.1f}%, total attempts: {total_attempts}")
    if model in pass_3_models:
        print(f"Pass at 3: {mean_pass_3_rate:.1f}%, hard -> {hard_3_rate:.1f}%, easy -> {easy_3_rate:.1f}%")
    if model in pass_7_models:
        print(f"Pass at 7: {mean_pass_7_rate:.1f}%, hard -> {hard_7_rate:.1f}%, easy -> {easy_7_rate:.1f}%")
    print("-------------------")

problem_statements = dict()
with open(problem_file, "r") as f:
    for line in f:
        data = json.loads(line)
        problem_statements[data["id"]] = data["problem_statement"]

print(f"HARDEST PROBLEMS")
for problem_id, (total, successes) in problem_rates[:2]:
    success_rate = (successes / total * 100) if total > 0 else 0
    print(f"{problem_id}: {successes}/{total} ({success_rate:.1f}%)")
    problem_prompt = problem_statements[problem_id]
    print(problem_prompt[:50])
    print()
    print("-------------------")

print(f"EASIEST PROBLEMS")
for problem_id, (total, successes) in problem_rates[::-1][:1]:
    success_rate = (successes / total * 100) if total > 0 else 0
    print(f"{problem_id}: {successes}/{total} ({success_rate:.1f}%)")
    problem_prompt = problem_statements[problem_id]
    print(problem_prompt[:50])
    print()
    print("-------------------")