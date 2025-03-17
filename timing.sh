#!/bin/bash

./test.sh
rm -rf sdp/__pycache__
./timing.py > timing.csv
rm -rf sdp/__pycache__
