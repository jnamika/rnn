#!/bin/sh

my_path=../../bin

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt *.scale *.restore
    exit
fi

cat <<EOS | python > target.txt
import sys
sys.path.append('$my_path')
import gen_target
gen_target.print_Lorenz_attractor(1000,truncate_length=1000)
EOS

$my_path/rnn-scale target.txt
$my_path/rnn-learn -c config.txt target.txt.scale
$my_path/rnn-generate -n 2000 rnn.dat > orbit.log
$my_path/rnn-scale-restore orbit.log

