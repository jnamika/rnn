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
gen_target.print_sin_curve(500, 20)
EOS

$my_path/rnn-learn -e 5000 target.txt
$my_path/rnn-generate rnn.dat > orbit.log

