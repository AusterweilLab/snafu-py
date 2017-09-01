#/bin/sh
for filename in $(ls *.pickle)
do
    echo $filename
    ./check_half.py $filename >> pickles.csv
done
