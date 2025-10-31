#!/usr/bin/env python

import argparse
import numpy as np
import time
import warnings

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="./timing.py -noheader -pmin 6 -pmax 23 -pdiff 3 pyfftw challenger"
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

    modules = utils.import_sdp_mods(args.include, args.ignore)

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
        print("module,len_Q,len_T,n_iter,time", flush=True)

    start_timing = time.time()
    for mod in modules:
        mod_name = mod.__name__.removeprefix("sdp.").removesuffix("_sdp")        
        mod.setup(np.random.rand(2), np.random.rand(2))
        
        for i in range(p_min, p_max + 1):
            Q = np.random.rand(2**i)
            
            j_range = range(i + skip_p_equal, min(i + p_diff + 1, p_max + 1))
            timing = np.zeros(len(j_range), dtype=np.float64)
            for _ in range(n_iter):
                lst = []
                for j in j_range:
                    T = np.random.rand(2**j)
                
                    start = time.time()
                    mod.sliding_dot_product(Q, T)
                    diff = time.time() - start
                    lst.append(diff)
                
                timing += np.array(lst) 
            
            timing /= n_iter 


            for j_index, j in enumerate(j_range):
                T = np.random.rand(2**j)
                info = (
                    f"{mod_name},{len(Q)},{len(T)},{n_iter}"
                    + f",{timing[j_index]}"
                )
                print(info, flush=True)

            
    elapsed_timing = np.round((time.time() - start_timing) / 60.0, 2)
    warnings.warn(f"Test completed in {elapsed_timing} min")
