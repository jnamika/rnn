#!/bin/sh

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt
    exit
fi

cat <<EOS | python > target.txt
import sys
sys.path.append('..')
import gen_target
for n in xrange(10):
    gen_target.print_golden_mean_shift(20, 100, output_type='softmax')
    print '\n'
EOS

rnn-learn -c config.txt target.txt
rnn-generate -n 500 rnn.dat > orbit.log

