# -*- coding:utf-8 -*-

import sys
import re

def tail_n(f, n=10, offset=0):
    avg_length = 74
    to_read = n + offset
    while 1:
        try:
            f.seek(-int(avg_length * to_read), 2)
        except IOError:
            f.seek(0)
        pos = f.tell()
        lines = f.read().splitlines()
        if len(lines) > to_read or pos == 0:
            return lines[-to_read:offset and -offset or None]
        avg_length *= 1.3

def read_parameter(f):
    r = {}
    r['in_state_size'] = re.compile(r'^# in_state_size')
    r['c_state_size'] = re.compile(r'^# c_state_size')
    r['out_state_size'] = re.compile(r'^# out_state_size')
    r['output_type'] = re.compile(r'^# output_type')
    r['delay_length'] = re.compile(r'^# delay_length')
    r['lyapunov_spectrum_size'] = re.compile(r'^# lyapunov_spectrum_size')
    r['target_num'] = re.compile(r'^# target_num')
    r['target'] = re.compile(r'^# target ([0-9]+)')
    r_comment = re.compile(r'^#')
    params = {}
    for line in f:
        for k,v in r.iteritems():
            if (v.match(line)):
                x = line.split('=')[1]
                if k == 'target':
                    m = int(v.match(line).group(1))
                    if (k in params):
                        params[k][m] = x
                    else:
                        params[k] = {m:x}
                else:
                    params[k] = x

        if (r_comment.match(line) == None):
            break
    f.seek(0)
    return params

def current_line(f, epoch=None):
    s = None
    if epoch == None:
        line = tail_n(f, 1)[0]
        s = line.split('\t')
    else:
        r = re.compile(r'(^#)|(^$)')
        for line in f:
            if (r.match(line) == None):
                x = line[:-1].split('\t')
                if (int(x[0]) == epoch):
                    s = x
    f.seek(0)
    return s

def current_target(f):
    r = re.compile(r'^# target:([0-9]+)')
    target = 0
    for line in f:
        m = r.match(line)
        if (m):
            target = int(m.group(1))
            break
    f.seek(0)
    return target



def print_state(f, epoch=None):
    r = re.compile(r'^# epoch')
    if epoch == None:
        params = read_parameter(f)
        t = current_target(f)
        length = int(params['target'][t])
        lines = tail_n(f, length + 3)
        flag = 0
        for line in lines:
            if (r.match(line)):
                flag = 1
            if (flag):
                print line
    else:
        current_epoch = -1
        for line in f:
            if (r.match(line)):
                current_epoch = int(line.split('=')[1])
            if current_epoch == epoch:
                print line[:-1]

def print_weight(f, epoch=None):
    params = read_parameter(f)
    in_state_size = int(params['in_state_size'])
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    s = current_line(f, epoch)
    if s != None:
        epoch = s[0]
        s = s[1:]
        w_i2c, w_c2c, w_c2o = [], [], []
        for i in xrange(c_state_size):
            w_i2c.append(s[:in_state_size])
            s = s[in_state_size:]
            w_c2c.append(s[:c_state_size])
            s = s[c_state_size:]
        for i in xrange(out_state_size):
            w_c2o.append(s[:c_state_size])
            s = s[c_state_size:]
        print 'epoch : %s' % epoch
        print 'weight (input to context)'
        for w in w_i2c:
            print '\t'.join([str(x) for x in w])
        print 'weight (context to context)'
        for w in w_c2c:
            print '\t'.join([str(x) for x in w])
        print 'weight (context to output)'
        for w in w_c2o:
            print '\t'.join([str(x) for x in w])


def print_threshold(f, epoch=None):
    params = read_parameter(f)
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    s = current_line(f, epoch)
    if s != None:
        epoch = s[0]
        s = s[1:]
        t_c = s[:c_state_size]
        s = s[c_state_size:]
        t_o = s[:out_state_size]
        print 'epoch : %s' % epoch
        print 'threshold (context)'
        print '\t'.join([str(x) for x in t_c])
        print 'threshold (output)'
        print '\t'.join([str(x) for x in t_o])

def print_tau(f, epoch=None):
    params = read_parameter(f)
    c_state_size = int(params['c_state_size'])
    s = current_line(f, epoch)
    if s != None:
        epoch = s[0]
        tau = s[1:]
        print 'epoch : %s' % epoch
        print 'time constant'
        print '\t'.join([str(x) for x in tau])

def print_sigma(f, epoch=None):
    s = current_line(f, epoch)
    if s != None:
        epoch = s[0]
        sigma = s[1]
        variance = s[2]
        print 'epoch : %s' % epoch
        print 'sigma : %s' % sigma
        print 'variance : %s' % variance


def print_init(f, epoch=None):
    params = read_parameter(f)
    target_num = int(params['target_num'])
    r = re.compile(r'^# epoch')
    if epoch == None:
        lines = tail_n(f, target_num + 1)
        flag = 0
        for line in lines:
            if (r.match(line)):
                flag = 1
            if (flag):
                print line
    else:
        current_epoch = -1
        for line in f:
            if (r.match(line)):
                current_epoch = int(line.split('=')[1])
            if current_epoch == epoch:
                print line[:-1]

def print_adapt_lr(f, epoch=None):
    s = current_line(f, epoch)
    if s != None:
        epoch = s[0]
        adapt_lr = s[1]
        print 'epoch : %s' % epoch
        print 'adaptive learning rate: %s' % adapt_lr

def print_error(f, epoch=None):
    s = current_line(f, epoch)
    if s != None:
        epoch = s[0]
        error = s[1:]
        print 'epoch : %s' % epoch
        print 'error / (length * dimension)'
        print '\t'.join([str(x) for x in error])

def print_lyapunov(f, epoch=None):
    params = read_parameter(f)
    target_num = int(params['target_num'])
    ls_size = int(params['lyapunov_spectrum_size'])
    s = current_line(f, epoch)
    if s != None:
        epoch = s[0]
        s = s[1:]
        print 'epoch : %s' % epoch
        line = ['target']
        for i in xrange(ls_size):
            line.append('lyapunov[%d]' % i)
        print '\t'.join(line)
        for i in xrange(target_num):
            print '%d\t%s' % (i, '\t'.join([str(x) for x in s[:ls_size]]))
            s = s[ls_size:]

def print_entropy(f, epoch=None):
    params = read_parameter(f)
    target_num = int(params['target_num'])
    s = current_line(f, epoch)
    if s != None:
        epoch = s[0]
        s = s[1:]
        print 'epoch : %s' % epoch
        line = ['target', 'KL-divergence', 'generation-rate',
                'entropy(target)', 'entropy(out)']
        print '\t'.join(line)
        for i in xrange(target_num):
            print '%d\t%s' % (i, '\t'.join([str(x) for x in s[:4]]))
            s = s[4:]

def print_period(f, epoch=None):
    s = current_line(f, epoch)
    if s != None:
        epoch = s[0]
        error = s[1:]
        print 'epoch : %s' % epoch
        print 'Period'
        print '\t'.join([str(x) for x in error])

def print_log(f, epoch=None):
    line = f.readline()
    if (re.compile(r'^# STATE FILE').match(line)):
        print_state(f, epoch)
    elif (re.compile(r'^# WEIGHT FILE').match(line)):
        print_weight(f, epoch)
    elif (re.compile(r'^# THRESHOLD FILE').match(line)):
        print_threshold(f, epoch)
    elif (re.compile(r'^# TAU FILE').match(line)):
        print_tau(f, epoch)
    elif (re.compile(r'^# SIGMA FILE').match(line)):
        print_sigma(f, epoch)
    elif (re.compile(r'^# INIT FILE').match(line)):
        print_init(f, epoch)
    elif (re.compile(r'^# ADAPT_LR FILE').match(line)):
        print_adapt_lr(f, epoch)
    elif (re.compile(r'^# ERROR FILE').match(line)):
        print_error(f, epoch)
    elif (re.compile(r'^# LYAPUNOV FILE').match(line)):
        print_lyapunov(f, epoch)
    elif (re.compile(r'^# ENTROPY FILE').match(line)):
        print_entropy(f, epoch)
    elif (re.compile(r'^# PERIOD FILE').match(line)):
        print_period(f, epoch)


def main():
    epoch = None
    if str.isdigit(sys.argv[1]):
        epoch = int(sys.argv[1])
    for file in sys.argv[2:]:
        f = open(file, 'r')
        print_log(f, epoch)
        f.close()


if __name__ == '__main__':
    main()

