# -*- coding:utf-8 -*-

import sys
import re
import math
import rnn_runner

def append_sequence_to_frequency(sequence, block_length=1, frequency={}):
    for n in range(len(sequence) - block_length + 1):
        s = tuple(sequence[n:n+block_length])
        if s in frequency:
            frequency[s] += 1
        else:
            frequency[s] = 1
    return frequency

def frequency2probability(frequency):
    sum = 0
    for k,v in frequency.items():
        sum += v
    for k,v in frequency.items():
        frequency[k] = v / float(sum)
    return frequency

def kullback_leibler_divergence(f1, f2):
    lower = 0.001
    sum = 0
    p = frequency2probability(f1)
    q = frequency2probability(f2)
    kl_div = 0
    for k,v in p.items():
        if k in q:
            kl_div += v * math.log(v/q[k])
        else:
            kl_div += v * math.log(v/lower)
    for k,v in q.items():
        if k not in p:
            kl_div += lower * math.log(lower/v)
    return kl_div

def get_KL_div(length, samples, truncate_length, block_length, divide_num,
        runner, files):
    def divide(x):
        s = int(math.floor(divide_num * x))
        if s == divide_num:
            s = divide_num - 1
        return s
    output_type = runner.output_type()
    if output_type == 0:
        func = lambda x: divide(0.5 * (x+1))
    else:
        func = lambda x: divide(x)
    f1 = {}
    for i in range(samples):
        runner.set_time_series_id(-1)
        for x in runner.closed_loop(truncate_length):
            pass
        sequence = []
        for x in runner.closed_loop(length):
            s = tuple(map(func, x[0]))
            sequence.append(s)
        f1 = append_sequence_to_frequency(sequence, block_length, f1)
    p = re.compile(r'(^#)|(^$)')
    f2 = {}
    for file in files:
        sequence = []
        for line in open(file, 'r'):
            if p.match(line) == None:
                input = list(map(float, line[:-1].split()))
                s = tuple(map(func, input))
                sequence.append(s)
        f2 = append_sequence_to_frequency(sequence, block_length, f2)
    return kullback_leibler_divergence(f1, f2)

def main():
    seed, length, samples, truncate_length, block_length, divide_num = \
            [int(x) if str.isdigit(x) else 0 for x in sys.argv[1:7]]
    rnn_file = sys.argv[7]
    rnn_runner.init_genrand(seed)
    runner = rnn_runner.RNNRunner()
    runner.init(rnn_file)
    print(get_KL_div(length, samples, truncate_length, block_length,
            divide_num, runner, sys.argv[8:]))


if __name__ == '__main__':
    main()

