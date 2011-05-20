# -*- coding:utf-8 -*-

import sys
import os
import re
import rnn_runner

def main():
    seed = int(sys.argv[1]) if str.isdigit(sys.argv[1]) else 0
    ignore_index = [int(x) for x in sys.argv[2].split(',') if str.isdigit(x)]
    type = sys.argv[3]
    rnn_file = sys.argv[4]
    sequence_file = sys.argv[5]

    rnn_runner.init_genrand(seed)
    runner = rnn_runner.RNNRunner()
    runner.init(rnn_file)
    runner.set_time_series_id()

    p = re.compile(r'(^#)|(^$)')
    out_state_queue = []
    for line in open(sequence_file, 'r'):
        if p.match(line) == None:
            input = map(float, line[:-1].split())
            if len(out_state_queue) >= runner.delay_length():
                out_state = out_state_queue.pop(0)
                for i in ignore_index:
                    input[i] = out_state[i]
            runner.in_state(input)
            runner.update()
            out_state = runner.out_state()
            if type == 'o':
                print '\t'.join([str(x) for x in out_state])
            elif type == 'c':
                c_state = runner.c_state()
                print '\t'.join([str(x) for x in c_state])
            elif type == 'a':
                c_state = runner.c_state()
                print '\t'.join([str(x) for x in out_state + c_state])
            out_state_queue.append(out_state)

if __name__ == '__main__':
    main()

