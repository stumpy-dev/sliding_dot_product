#!/usr/bin/env python

import argparse
import pkgutil
import ast
import importlib
import numpy as np
import numpy.testing as npt
import sdp
import time
import warnings


def func_exists(mod_path, func_name):
    try:
        with open(mod_path, "r") as file:
            module_content = file.read()
    except FileNotFoundError:
        return False  # Module file not found

    try:
        tree = ast.parse(module_content)
    except SyntaxError:
        return False  # Syntax error in module

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return True
    return False


def import_sdp_mods(include=None, ignore=None):
    mods = []
    for m in sorted(list(pkgutil.iter_modules(sdp.__path__))):
        mod_path = f"sdp/{m[1]}.py"
        if (
            include is not None
            and len(include)
            and not any(mod in mod_path for mod in include)
        ):
            continue
        if (
            ignore is not None
            and len(ignore)
            and any(mod in mod_path for mod in ignore)
        ):
            continue

        if (
            "sdp" in m[1]
            and func_exists(mod_path, "sliding_dot_product")
            and func_exists(mod_path, "setup")
        ):
            mod_name = f"sdp.{m[1]}"
            mod = importlib.import_module(mod_name)
            mods.append(mod)

    return mods


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="./test.py -noheader -pmin 6 -pmax 23 -pdiff 3 pyfftw challenger"
    )
    parser.add_argument("-noheader", default=False, action="store_true")
    parser.add_argument(
        "-timeout",
        default=5.0,
        type=float,
        help="Number of seconds to wait for a run before timing out",
    )
    parser.add_argument(
        "-pequal", default=False, action="store_true", help="Compute `len(Q) == len(T)`"
    )
    parser.add_argument(
        "-niter", default=4, type=int, help="Number of iterations to run"
    )
    parser.add_argument("-pmin", default=6, type=int, help="Minimum 2^p to use")
    parser.add_argument("-pmax", default=27, type=int, help="Maximum 2^p to use")
    parser.add_argument(
        "-pdiff",
        default=100,
        type=int,
        help="Maximum deviation from the minimum 2^p allowed",
    )
    parser.add_argument(
        "-ignore",
        default=None,
        nargs="*",
        help="Keyword of modules to match and ignore",
    )
    parser.add_argument(
        "include",
        default=None,
        nargs="*",
        help="Keyword of modules to match and include",
    )
    args = parser.parse_args()

    modules = import_sdp_mods(args.include, args.ignore)

    noheader = args.noheader
    timeout = args.timeout
    if args.pequal:
        skip_p_equal = 0
    else:
        skip_p_equal = 1
    n_iter = args.niter
    p_min = args.pmin
    p_max = args.pmax
    p_diff = args.pdiff

    if not noheader:
        print(f"module,len_Q,len_T,n_iter,time", flush=True)

    start_timing = time.time()
    for mod in modules:
        mod_name = mod.__name__.removeprefix("sdp.").removesuffix("_sdp")
        for i in range(p_min, p_max + 1):
            Q = np.random.rand(2**i)
            break_Q = False
            for j in range(i + skip_p_equal, min(i + p_diff + 1, p_max + 1)):
                T = np.random.rand(2**j)
                break_T = False

                mod.setup(Q, T)

                elapsed_times = []
                for _ in range(n_iter):
                    start = time.time()
                    mod.sliding_dot_product(Q, T)
                    diff = time.time() - start
                    if diff > timeout:
                        break_T = True
                        warnings.warn(f"SKIPPED: {mod_name},{len(Q)},{len(T)},{diff})")
                        break
                    else:
                        elapsed_times.append(diff)

                if break_T:
                    if j == i + 1:
                        break_Q = True
                    break

                print(
                    f"{mod_name},{len(Q)},{len(T)},{len(elapsed_times)},{sum(elapsed_times) / len(elapsed_times)}",
                    flush=True,
                )

            if break_Q:
                warnings.warn(f"SKIPPED: {mod_name},{len(Q)},>{len(T)},{diff})")
                break

    elapsed_timing = np.round((time.time() - start_timing) / 60.0, 2)
    warnings.warn(f"Test completed in {elapsed_timing} min")
