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
${path1}rnn-learn -c config.txt target*.txt
${path2}rnn-generate -n 2000 rnn.dat > orbit.log

