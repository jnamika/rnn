# -*- coding:utf-8 -*-

import sys
import re

p = re.compile(r'^$')

def get_standard_scale_parameter(files):
    upper, lower, count, dim = None, None, 0, 1
    for file in files:
        for line in open(file, 'r'):
            if not p.match(line):
                input = list(map(float, line[:-1].split()))
                d = len(input)
                if count > 0 and d != dim:
                    raise
                dim = d
                if count == 0:
                    upper = input
                    lower = input
                else:
                    upper = list(map(max, upper, input))
                    lower = list(map(min, lower, input))
                count += 1
    if count == 0:
        raise
    s = 0.8
    a = list(map(lambda u,l: (2 * s) / (u - l), upper, lower))
    b = list(map(lambda u,l: s * ((u + l) / (u - l)), upper, lower))
    return a, b

def get_softmax_scale_parameter(files):
    s, m, count, dim = 0, 0, 0, 1
    for file in files:
        for line in open(file, 'r'):
            if not p.match(line):
                input = list(map(float, line[:-1].split()))
                d = len(input)
                if count > 0 and d != dim:
                    raise
                dim = d
                s += sum(input)
                x = min(input)
                m = x if count == 0 else min([m, x])
                count += 1
    if count == 0:
        raise
    s = s / count
    s -= m * dim
    a = [1 / (s - m)] * dim
    b = [m / (s - m)] * dim
    return a, b

def scale(output_type, parameter_file, files):
    if output_type == 0:
        a,b = get_standard_scale_parameter(files)
    elif output_type == 1:
        a, b = get_softmax_scale_parameter(files)
    else:
        return
    func = lambda x,a,b: a * x - b
    conv = lambda x: list(map(func, x, a, b))
    for file in files:
        f = open('%s.scale' % file, 'w')
        for line in open(file, 'r'):
            if not p.match(line):
                input = list(map(float, line[:-1].split()))
                output = conv(input)
                f.write('%s\n' % '\t'.join([str(x) for x in output]))
            else:
                f.write(line)
        f.close()
    f = open(parameter_file, 'w')
    f.write('%s\n' % '\t'.join([str(x) for x in a]))
    f.write('%s\n' % '\t'.join([str(x) for x in b]))
    f.close()

def restore(parameter_file, files):
    f = open(parameter_file, 'r')
    a = list(map(float, f.readline()[:-1].split()))
    b = list(map(float, f.readline()[:-1].split()))
    f.close()
    func = lambda x,a,b: (x + b) / a
    conv = lambda x: list(map(func, x, a, b))
    for file in files:
        f = open('%s.restore' % file, 'w')
        for line in open(file, 'r'):
            if not p.match(line):
                input = list(map(float, line[:-1].split()))
                output = conv(input)
                f.write('%s\n' % '\t'.join([str(x) for x in output]))
            else:
                f.write(line)
        f.close()

def main():
    mode, output_type = [int(x) if str.isdigit(x) else 0 for x in
            sys.argv[1:3]]
    parameter_file = sys.argv[3]
    if mode == 0:
        scale(output_type, parameter_file, sys.argv[4:])
    else:
        restore(parameter_file, sys.argv[4:])


if __name__ == '__main__':
    main()

