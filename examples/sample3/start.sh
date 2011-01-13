#!/bin/sh

my_path=../../bin

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt
    exit
fi

echo "
import sys
sys.path.append('$my_path')
import gen_target
for n in xrange(10):
    gen_target.print_golden_mean_shift(20, 100, output_type='softmax')
    print '\n'
" | python > target.txt

$my_path/rnn-learn -e 10000 -n 20 -t 2.0 -k 1 -a target.txt
$my_path/rnn-generate -n 500 rnn.dat > orbit.log

