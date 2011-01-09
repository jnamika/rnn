# -*- coding:utf-8 -*-

import sys
import re
import math
import datetime
import rnn_runner

def append_sequence_to_frequency(sequence, block_length=1, frequency={}):
    for n in xrange(len(sequence) - block_length + 1):
        s = tuple(sequence[n:n+block_length])
        if s in frequency:
            frequency[s] += 1
        else:
            frequency[s] = 1
    return frequency

def frequency2probability(frequency):
    sum = 0
    for k,v in frequency.iteritems():
        sum += v
    for k,v in frequency.iteritems():
        frequency[k] = v / float(sum)
    return frequency

def kullback_leibler_divergence(f1, f2):
    lower = 0.001
    sum = 0
    p = frequency2probability(f1)
    q = frequency2probability(f2)
    kl_div = 0
    for k,v in p.iteritems():
        if (k in q):
            kl_div += v * math.log(v/q[k])
        else:
            kl_div += v * math.log(v/lower)
    for k,v in q.iteritems():
        if (k not in p):
            kl_div += lower * math.log(lower/v)
    return kl_div

def main():
    seed, steps, samples, truncate_length, block_length, divide_num = \
            map(lambda x: int(x) if str.isdigit(x) else 0, sys.argv[1:7])
    p = re.compile(r'(^#)|(^$)')
    rnn_file = sys.argv[7]
    if seed == 0:
        now = datetime.datetime.utcnow()
        seed = ((now.hour * 3600 + now.minute * 60 + now.second) *
                now.microsecond)
    rnn_runner.init_genrand(seed % 4294967295)
    runner = rnn_runner.rnn_runner()
    runner.init(rnn_file)
    def divide(x):
        s = int(math.floor(divide_num * x))
        if (s == divide_num):
            s = divide_num - 1
        return s
    output_type = runner.output_type()
    if (output_type == 0):
        func = lambda x: divide(0.5 * (x+1))
    else:
        func = lambda x: divide(x)
    f1 = {}
    for i in xrange(samples):
        runner.set_time_series_id(-1)
        for x,y in runner.closed_loop(truncate_length):
            pass
        sequence = []
        for x,y in runner.closed_loop(steps):
            s = tuple(map(func, x))
            sequence.append(s)
        f1 = append_sequence_to_frequency(sequence, block_length, f1)
    args = sys.argv[8:]
    f2 = {}
    for arg in args:
        sequence = []
        for line in open(arg, 'r'):
            if (p.match(line) == None):
                input = map(float, line[:-1].split())
                s = tuple(map(func, input))
                sequence.append(s)
        f2 = append_sequence_to_frequency(sequence, block_length, f2)
    print kullback_leibler_divergence(f1, f2)


if __name__ == "__main__":
    main()

