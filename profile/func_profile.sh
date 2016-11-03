#!/bin/sh
# USAGE: ./func_profile.sh <script to profile>

filename=$(basename "${1}")      # file minus path
rootname=${filename%.*}         # file minus extension

python -m cProfile -o ${rootname}_profile.cprof ${1}
pyprof2calltree -i ${rootname}_profile.cprof -o callgrind.${rootname}_profile.cprof

# open using QCachegrind GUI
