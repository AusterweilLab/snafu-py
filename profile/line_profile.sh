#!/bin/sh
# USAGE: Prefix any function in code with decorator @profile before using

filename=$(basename "${1}")      # file minus path

kernprof -l ${1}
python -m line_profiler ${filename}.lprof
