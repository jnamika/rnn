#!/bin/sh

if [ "$1" = clean ]; then
    rm -f *.log *.dat target*.txt
    exit
fi

cat <<EOS | python
import sys
import re
sys.path.append('..')
import gen_target
for n in xrange(4):
    file = re.compile('XXXXXX').sub('%06d' % n, 'targetXXXXXX.txt')
    sys.stdout = open(file, 'w')
    gen_target.print_Reber_grammar(50, 100)
    sys.stdout = sys.__stdout__
EOS

rnn-learn -c config.txt target*.txt
rnn-generate -n 2000 rnn.dat > orbit.log

