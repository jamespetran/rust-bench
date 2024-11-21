"""
This code is based on the Human-Eval Rust evaluation implementation from MultiPL-E
(https://arxiv.org/abs/2208.08227). Whith some modifications.

Original authors: 
@misc{cassano2022multiple,
    title={MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation},
    author={Federico Cassano and John Gouwar and Daniel Nguyen and Sydney Nguyen
            and Luna Phipps-Costin and Donald Pinckney and Ming-Ho Yee and Yangtian Zi
            and Carolyn Jane Anderson and Molly Q Feldman and Arjun Guha
            and Michael Greenberg and Abhinav Jangda},
    year={2022},
    eprint={2208.08227},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""
import tempfile
import shutil
import os
import signal
import subprocess
import json
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Tuple, List, Optional

cargo_harness_dir = "rust_execution"


def run_with_timeout(cmd: str, tmp_cargo_path: str, timeout: int = 5, print_debug: bool = False) -> Optional[Tuple[str, str]]:
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmp_cargo_path, preexec_fn=os.setsid
    )
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the process group
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        out, err = p.communicate()
        return None
    out = out.decode("utf-8")
    err = err.decode("utf-8")
    if print_debug:
        print("## RUN OUTPUTS ##")
        print("STDOUT:")
        print(out)
        print("STDERR:")
        print(err, flush=True)
    return out, err


def grab_compile_errs(inp: str) -> List[str]:
    errors = []
    for line in inp.splitlines():
        if line == "":
            continue
        try:
            o = json.loads(line)
            if o is not None and o.get("reason") == "compiler-message":
                message = o.get("message", {})
                if message.get("level") == "error" and message.get("spans"):
                    errors.append(message.get("rendered", ""))
        except json.JSONDecodeError:
            continue
    return errors


def write_to_file_toplevel(path: str, code: str):
    with open(path, "w") as f:
        f.write(code)


def evaluate_rs_sync(func: str, timeout: int = 25) -> Tuple[str, bool]:
    with tempfile.TemporaryDirectory(prefix='cargo_harness_') as tmp_dir:
        # Copy cargo harness into tmp_dir
        shutil.copytree(cargo_harness_dir, tmp_dir, dirs_exist_ok=True)
        tmp_path = os.path.join(tmp_dir, "src", "main.rs")
        write_to_file_toplevel(tmp_path, func)
        
        res = run_with_timeout(
            "cargo check --message-format=json", tmp_dir, timeout=timeout, print_debug=False)
        if res is None:
            return "Timeout in cargo check", False
        out, err = res

        errors = grab_compile_errs(out)
        return "\n".join(errors), len(errors) == 0


async def evaluate_rs(func: str, executor: ProcessPoolExecutor, timeout: int = 125) -> Tuple[str, bool]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, partial(evaluate_rs_sync, func, timeout))

