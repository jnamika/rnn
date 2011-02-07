#!/bin/sh

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt *.scale *.restore
    exit
fi

cat <<EOS | python > target.txt
import sys
sys.path.append('..')
import gen_target
gen_target.print_Lorenz_attractor(1000,truncate_length=1000)
EOS

rnn-scale target.txt
rnn-learn -c config.txt target.txt.scale
rnn-generate -n 2000 rnn.dat > orbit.log
rnn-scale-restore orbit.log

