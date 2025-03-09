#!/bin/bash

get_safe_python_version()
{
    SAFE_PYTHON=$(curl --location https://devguide.python.org/versions | xmllint --html --xpath '//section[@id="supported-versions"]//table/tbody/tr[count(//section[@id="supported-versions"]//table/tbody/tr[td[.="security"]]/preceding-sibling::*)]/td[1]/p/text()' - 2>/dev/null)
}

get_safe_python_version
echo $SAFE_PYTHON