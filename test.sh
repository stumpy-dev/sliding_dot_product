#!/bin/bash

check_errs()
{
  # Function. Parameter 1 is the return code
  if [[ $1 -ne "0" && $1 -ne "5" ]]; then
    echo "Error: Test execution encountered exit code $1"
    # as a bonus, make our script exit with the right error code.
    exit $1
  fi
}

clean_up()
{
    echo "Cleaning Up"
    rm -rf "__pycache__/"
    rm -rf "sdp/__pycache__/"
}


check_black()
{
    echo "Checking Black Code Formatting"
    black --check --exclude=".*\.ipynb" --extend-exclude=".venv" --diff ./
    check_errs $?
}

check_flake()
{
    echo "Checking Flake8 Style Guide Enforcement"
    flake8 --extend-exclude=.venv ./
    check_errs $?
}


test_unit()
{
    echo "Testing Functions"
    SECONDS=0
    pytest -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning -W ignore::UserWarning test.py
    check_errs $?
    duration=$SECONDS
    echo "Elapsed Time: $((duration / 60)) minutes and $((duration % 60)) seconds" 
}

test_coverage()
{
    echo "Disabling Numba JIT and CUDA Compiled Functions"
    export NUMBA_DISABLE_JIT=1

    echo "Testing Code Coverage"
    coverage erase

    SECONDS=0
    coverage run --append --source=. -m pytest -rsx -W ignore::RuntimeWarning -W ignore::DeprecationWarning -W ignore::UserWarning test.py
    check_errs $?            
    duration=$SECONDS
  
    echo "Elapsed Time: $((duration / 60)) minutes and $((duration % 60)) seconds"
    coverage report -m --fail-under=100 --skip-covered --omit=timing.py,utils.py
}


clean_up
check_black
check_flake
test_unit

clean_up
test_coverage

clean_up