#!/bin/bash

rm -rf sdp/__pycache__
./test.py > timing.csv
rm -rf sdp/__pycache__
