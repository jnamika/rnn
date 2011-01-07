# -*- coding:utf-8 -*-

import sys

def scale(output_type, parameter_file, files):
    max_lst, min_lst = None, None
    for file in files:
        for line in open(file, 'r'):
            input = map(float, line[:-1].split())
            max_lst = input if not max_lst else map(lambda x,y: x if x > y
                    else y, max_lst, input)
            min_lst = input if not min_lst else map(lambda x,y: x if x < y
                    else y, min_lst, input)
    if output_type == 0:
        func = lambda x,y,z: 1.6 * (x - z)/(y - z) - 0.8
    elif output_type == 1:
        func = lambda x,y,z: (x - z)/(y - z)
    else:
        func = lambda x,y,z: x
    for file in files:
        f = open('%s.scale' % file, 'w')
        for line in open(file, 'r'):
            input = map(float, line[:-1].split())
            output = map(func, input, max_lst, min_lst)
            f.writelines(['\t'.join([str(x) for x in output]), '\n'])
        f.close()
    if max_lst != None and min_lst != None:
        f = open(parameter_file, 'w')
        f.writelines([str(output_type), '\n'])
        f.writelines(['\t'.join([str(x) for x in max_lst]), '\n'])
        f.writelines(['\t'.join([str(x) for x in min_lst]), '\n'])
        f.close()

def restore(parameter_file, files):
    f = open(parameter_file, 'r')
    output_type = int(f.readline())
    line = f.readline()
    max_lst = map(float, line[:-1].split())
    line = f.readline()
    min_lst = map(float, line[:-1].split())
    f.close()
    if output_type == 0:
        func = lambda x,y,z: ((x + 0.8) * (y - z)) / 1.6 + z
    elif output_type == 1:
        func = lambda x,y,z: x * (y - z) + z
    else:
        func = lambda x,y,z: x
    for file in files:
        f = open('%s.restore' % file, 'w')
        for line in open(file, 'r'):
            input = map(float, line[:-1].split())
            output = map(func, input, max_lst, min_lst)
            f.writelines(['\t'.join([str(x) for x in output]), '\n'])
        f.close()
    pass

def main():
    mode, output_type = map(lambda x: int(x) if str.isdigit(x) else 0,
            sys.argv[1:3])
    parameter_file = sys.argv[3]
    if mode == 0:
        scale(output_type, parameter_file, sys.argv[4:])
    else:
        restore(parameter_file, sys.argv[4:])


if __name__ == "__main__":
    main()

