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

get_safe_python_version()
{
    SAFE_PYTHON=$(curl -v --location https://devguide.python.org/versions | xmllint --html --xpath '//section[@id="supported-versions"]//table/tbody/tr[count(//section[@id="supported-versions"]//table/tbody/tr[td[.="security"]]/preceding-sibling::*)]/td[1]/p/text()' - 2> /dev/null)
    check_errs $?
}

get_safe_python_version
echo $SAFE_PYTHON