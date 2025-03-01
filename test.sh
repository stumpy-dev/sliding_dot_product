#!/bin/bash
python -m pytest test_challenger.py 
rm -rf sdp/__pycache__
./test.py > timing.csv
rm -rf sdp/__pycache__
