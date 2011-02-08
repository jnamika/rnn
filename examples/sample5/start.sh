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
${path1}rnn-learn -e 5000 -c config.txt target.txt
${path2}rnn-generate rnn.dat > orbit.log

