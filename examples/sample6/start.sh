#!/bin/sh

if [ "$1" = clean ]; then
    rm -f *.log *.dat target.txt
    exit
fi

cat <<EOS | python > target.txt
import sys
sys.path.append('..')
import gen_target
for n in xrange(3):
    gen_target.print_henon_map(500, 100)
    print '\n'
EOS

rnn-learn -c config.txt target.txt
rnn-generate -n 1500 rnn.dat > orbit.log

