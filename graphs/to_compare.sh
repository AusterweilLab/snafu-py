#!/bin/sh

sort ${1} > ${1}_sorted
sort ${2} > ${2}_sorted 

# common links
COMMON=`comm -12 ${1}_sorted ${2}_sorted`

# left only (nohidden)
LEFT=`comm -23 ${1}_sorted ${2}_sorted`

# right only (our method)
RIGHT=`comm -13 ${1}_sorted ${2}_sorted`

# add 'color=red' etc to certain edges

COMMON=${COMMON/strict graph/}
COMMON=${COMMON/{/}
COMMON=${COMMON/\}/}
COMMON=${COMMON//]/,color=blue]}
LEFT=${LEFT//]/,color=red]}
RIGHT=${RIGHT//]/,color=green]}

echo "strict graph {" $COMMON $LEFT $RIGHT "}" > ${1}_colors.dot

rm ${1}_sorted
rm ${2}_sorted
