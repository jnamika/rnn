#!/bin/sh

if [ "$1" != clean ]; then
    dir=`dirname $0`

    config_file=$dir/config.txt

    target_num=4
    target_file=targetXXXXXX.txt

    cat << EOS | python
import sys
import re
sys.path.append('../../bin')
import gen_target
for n in xrange($target_num):
    file = re.compile('XXXXXX').sub('%06d' % n, '$target_file')
    sys.stdout = open(file, 'w')
    gen_target.print_Reber_grammar(500, 100)
    sys.stdout = sys.__stdout__
EOS

    time ../../bin/rnn-learn -c $config_file target*.txt
else
    rm -f *.log *.dat target*.txt
fi


