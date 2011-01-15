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
gen_target.print_van_der_Pol_attractor(1000,truncate_length=1000)
EOS

$my_path/rnn-learn -c config.txt target.txt
$my_path/rnn-generate -n 2000 rnn.dat > orbit.log

