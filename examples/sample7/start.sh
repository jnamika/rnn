#!/bin/sh

my_path=../../bin

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt
    exit
fi

cat <<EOS | python > target.txt
import sys
sys.path.append('$my_path')
import gen_target
for n in xrange(10):
    gen_target.print_golden_mean_shift(20, 100, output_type='softmax')
    print '\n'
EOS

$my_path/rnn-learn -c config.txt target.txt
$my_path/rnn-generate -n 500 rnn.dat > orbit.log

