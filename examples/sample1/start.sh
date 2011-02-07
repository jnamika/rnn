#!/bin/sh

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt
    exit
fi

cat <<EOS | python > target.txt
import sys
sys.path.append('..')
import gen_target
gen_target.print_sin_curve(500, 20)
EOS

rnn-learn -e 5000 target.txt
rnn-generate rnn.dat > orbit.log

