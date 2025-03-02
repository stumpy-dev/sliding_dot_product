#!/bin/bash

pytest test_modules.py 
rm -rf sdp/__pycache__
./test.py > timing.csv
rm -rf sdp/__pycache__
