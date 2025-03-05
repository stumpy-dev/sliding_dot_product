#!/bin/bash

pytest test.py 
rm -rf sdp/__pycache__
./timing.py > timing.csv
rm -rf sdp/__pycache__
