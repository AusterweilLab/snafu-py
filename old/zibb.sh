#/bin/sh
for filename in $(ls *zibb*.pickle)
do
    ./check_half.py $filename >> zibb.csv
done
