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

if [ x`which rnn-learn` == x ]; then
    path1=../../src/rnn-learn/
else
    path1=
fi
if [ x`which rnn-generate` == x ]; then
    path2=../../src/rnn-generate/
else
    path2=
fi
if [ x`which rnn-scale` == x ]; then
    path3=../../src/python/
else
    path3=
fi
${path3}rnn-scale target.txt
${path1}rnn-learn -c config.txt target.txt.scale
${path2}rnn-generate -n 2000 rnn.dat > orbit.log
${path3}rnn-scale-restore orbit.log

