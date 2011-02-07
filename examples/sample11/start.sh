#!/bin/sh

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt
    exit
fi

cat <<EOS | python > target.txt
import sys
sys.path.append('..')
import gen_target
gen_target.print_van_der_Pol_attractor(1000,truncate_length=1000)
EOS

rnn-learn -c config.txt target.txt
rnn-generate -n 2000 rnn.dat > orbit.log

