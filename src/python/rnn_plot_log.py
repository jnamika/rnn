# -*- coding:utf-8 -*-

import sys
import re
import subprocess
import tempfile
import rnn_print_log

def plot_state(f, filename, epoch, multiplot=False):
    params = rnn_print_log.read_parameter(f)
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    output_type = params['output_type']
    if re.search('STANDARD_TYPE', output_type):
        range = '[][-1:1]'
        pass
    elif re.search('SOFTMAX_TYPE', output_type):
        range = '[][0:1]'
    else:
        range = ''
    tmp = tempfile.NamedTemporaryFile()
    sys.stdout = tmp
    rnn_print_log.print_state(f, epoch)
    sys.stdout.flush()
    type = []
    type.append(('Target', out_state_size, lambda x: 2 * x + 2, range))
    type.append(('Output', out_state_size, lambda x: 2 * x + 3, range))
    type.append(('Context', c_state_size, lambda x: x + 2 * out_state_size + 2,
        ''))
    if multiplot:
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        p.stdin.write('set nokey;')
        p.stdin.write("set multiplot layout 3,1 title 'Type=State  File=%s';" %
                filename)
        for v in type:
            p.stdin.write("set xlabel 'Time step';")
            p.stdin.write("set ylabel '%s';" % v[0])
            command = ['plot %s ' % v[3]]
            for i in xrange(v[1]):
                command.append("'%s' u 1:%d w l," % (tmp.name, v[2](i)))
            p.stdin.write(''.join(command)[:-1])
            p.stdin.write('\n')
        p.stdin.write('unset multiplot\n')
        p.stdin.write('exit\n')
        p.wait()
    else:
        for v in type:
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            p.stdin.write('set nokey;')
            p.stdin.write("set title 'Type=%s  File=%s';" % (v[0], filename))
            p.stdin.write("set xlabel 'Time step';")
            p.stdin.write("set ylabel '%s';" % v[0])
            command = ['plot %s ' % v[3]]
            for i in xrange(v[1]):
                command.append("'%s' u 1:%d w l," % (tmp.name, v[2](i)))
            p.stdin.write(''.join(command)[:-1])
            p.stdin.write('\n')
            p.stdin.write('exit\n')
            p.wait()
    sys.stdout = sys.__stdout__

def plot_weight(f, filename, multiplot=False):
    params = rnn_print_log.read_parameter(f)
    in_state_size = int(params['in_state_size'])
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    index_i2c, index_c2c, index_c2o = [], [], []
    s = [x+2 for x in xrange(c_state_size *
        (in_state_size + c_state_size + out_state_size))]
    for i in xrange(c_state_size):
        index_i2c.extend(s[:in_state_size])
        s = s[in_state_size:]
        index_c2c.extend(s[:c_state_size])
        s = s[c_state_size:]
    for i in xrange(out_state_size):
        index_c2o.extend(s[:c_state_size])
        s = s[c_state_size:]
    type = []
    type.append(('Weight (input to context)', index_i2c))
    type.append(('Weight (context to context)', index_c2c))
    type.append(('Weight (context to output)', index_c2o))
    if multiplot:
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        p.stdin.write('set nokey;')
        p.stdin.write("set multiplot layout 3,1 title 'Type=Weight File=%s';" %
                filename)
        for v in type:
            p.stdin.write("set xlabel 'Learning epoch';")
            p.stdin.write("set ylabel '%s';" % v[0])
            command = ['plot ']
            for i in v[1]:
                command.append("'%s' u 1:%d w l," % (filename, i))
            p.stdin.write(''.join(command)[:-1])
            p.stdin.write('\n')
        p.stdin.write('unset multiplot\n')
    else:
        for v in type:
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            p.stdin.write('set nokey;')
            p.stdin.write("set title 'Type=Weight  File=%s';" % filename)
            p.stdin.write("set xlabel 'Learning epoch';")
            p.stdin.write("set ylabel '%s';" % v[0])
            command = ['plot ']
            for i in v[1]:
                command.append("'%s' u 1:%d w l," % (filename, i))
            p.stdin.write(''.join(command)[:-1])
            p.stdin.write('\n')

def plot_threshold(f, filename, multiplot=False):
    params = rnn_print_log.read_parameter(f)
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    type = []
    type.append(('Threshold (context)', c_state_size, lambda x: x + 2))
    type.append(('Threshold (output)', out_state_size, lambda x: x +
        c_state_size + 2))
    if multiplot:
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        p.stdin.write('set nokey;')
        p.stdin.write('set multiplot layout 2,1 title ' +
                "'Type=Threshold File=%s';" % filename)
        for v in type:
            p.stdin.write("set xlabel 'Learning epoch';")
            p.stdin.write("set ylabel '%s';" % v[0])
            command = ['plot ']
            for i in xrange(v[1]):
                command.append("'%s' u 1:%d w l," % (filename, v[2](i)))
            p.stdin.write(''.join(command)[:-1])
            p.stdin.write('\n')
        p.stdin.write('unset multiplot\n')
    else:
        for v in type:
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            p.stdin.write('set nokey;')
            p.stdin.write("set title 'Type=Threshold  File=%s';" % filename)
            p.stdin.write("set xlabel 'Learning epoch';")
            p.stdin.write("set ylabel '%s';" % v[0])
            command = ['plot ']
            for i in xrange(v[1]):
                command.append("'%s' u 1:%d w l," % (filename, v[2](i)))
            p.stdin.write(''.join(command)[:-1])
            p.stdin.write('\n')

def plot_tau(f, filename):
    params = rnn_print_log.read_parameter(f)
    c_state_size = int(params['c_state_size'])
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    p.stdin.write('set nokey;')
    p.stdin.write("set title 'Type=Time-constant  File=%s';" % filename)
    p.stdin.write("set xlabel 'Learning epoch';")
    p.stdin.write("set ylabel 'Time constant';")
    command = ['plot ']
    for i in xrange(c_state_size):
        command.append("'%s' u 1:%d w l," % (filename, i+2))
    p.stdin.write(''.join(command)[:-1])
    p.stdin.write('\n')

def plot_sigma(f, filename):
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    p.stdin.write('set nokey;')
    p.stdin.write("set title 'Type=Variance  File=%s';" % filename)
    p.stdin.write("set xlabel 'Learning epoch';")
    p.stdin.write("set ylabel 'Variance';")
    p.stdin.write("plot '%s' u 1:3 w l" % filename)
    p.stdin.write('\n')

def plot_init(f, filename, epoch):
    params = rnn_print_log.read_parameter(f)
    c_state_size = int(params['c_state_size'])
    tmp = tempfile.NamedTemporaryFile()
    sys.stdout = tmp
    rnn_print_log.print_init(f, epoch)
    sys.stdout.flush()
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    p.stdin.write('set nokey;')
    p.stdin.write("set title 'Type=Init  File=%s';" % filename)
    p.stdin.write("set xlabel 'x';")
    p.stdin.write("set ylabel 'y';")
    p.stdin.write('set pointsize 3;')
    command = ['plot ']
    index = [(2*x,(2*x+1)%c_state_size) for x in xrange(c_state_size) if 2*x <
            c_state_size]
    for x in index:
        command.append("'%s' u %d:%d w p," % (tmp.name, x[0]+2, x[1]+2))
    p.stdin.write(''.join(command)[:-1])
    p.stdin.write('\n')
    p.stdin.write('exit\n')
    p.wait()
    sys.stdout = sys.__stdout__

def plot_adapt_lr(f, filename):
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    p.stdin.write('set nokey;')
    p.stdin.write('set logscale y;')
    p.stdin.write("set title 'Type=Learning-rate  File=%s';" % filename)
    p.stdin.write("set xlabel 'Learning epoch';")
    p.stdin.write(
            "set ylabel 'Current learning rate / Initial learning rate';")
    p.stdin.write("plot '%s' u 1:2 w l" % filename)
    p.stdin.write('\n')

def plot_error(f, filename):
    params = rnn_print_log.read_parameter(f)
    target_num = int(params['target_num'])
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    p.stdin.write('set nokey;')
    p.stdin.write('set logscale y;')
    p.stdin.write("set title 'Type=Error  File=%s';" % filename)
    p.stdin.write("set xlabel 'Learning epoch';")
    p.stdin.write("set ylabel 'Error / (Length times Dimension)';")
    command = ['plot ']
    for i in xrange(target_num):
        command.append("'%s' u 1:%d w l," % (filename, i+2))
    p.stdin.write(''.join(command)[:-1])
    p.stdin.write('\n')

def plot_lyapunov(f, filename):
    params = rnn_print_log.read_parameter(f)
    target_num = int(params['target_num'])
    ls_size = int(params['lyapunov_spectrum_size'])
    p = re.compile(r'(^#)|(^$)')
    tmp = tempfile.NamedTemporaryFile()
    for line in f:
        if p.match(line) == None:
            input = map(float, line[:-1].split())
            lyapunov = [input[0]]
            for i in xrange(ls_size):
                lyapunov.append(sum(input[i+1::ls_size]) / target_num)
            tmp.write('%s\n' % '\t'.join([str(x) for x in lyapunov]))
    tmp.flush()
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    p.stdin.write('set nokey;')
    p.stdin.write("set title 'Type=Lyapunov  File=%s';" % filename)
    p.stdin.write("set xlabel 'Learning epoch';")
    p.stdin.write("set ylabel 'Lyapunov';")
    command = ['plot 0 w l lt 0']
    for i in xrange(ls_size):
        command.append("'%s' u 1:%d w l lt %d" % (tmp.name, i+2, i+1))
    p.stdin.write(','.join(command))
    p.stdin.write('\n')
    p.stdin.write('exit\n')
    p.wait()

def plot_entropy(f, filename, multiplot=False):
    params = rnn_print_log.read_parameter(f)
    target_num = int(params['target_num'])
    type = ['KL-divergence', 'generation-rate', 'entropy(target)',
            'entropy(out)']
    if multiplot:
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        p.stdin.write('set nokey;')
        p.stdin.write('set multiplot layout 3,1 title ' +
                "'Type=Entropy File=%s';" % filename)
        for i in [0, 1, 3]:
            p.stdin.write("set xlabel 'Learning epoch';")
            p.stdin.write("set ylabel '%s';" % type[i])
            command = ['plot ']
            for j in xrange(target_num):
                command.append("'%s' u 1:%d w l," % (filename, i+j*4+2))
            p.stdin.write(''.join(command)[:-1])
            p.stdin.write('\n')
        p.stdin.write('unset multiplot\n')
    else:
        for i in [0, 1, 3]:
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            p.stdin.write('set nokey;')
            p.stdin.write("set title 'Type=%s  File=%s';" % (type[i], filename))
            p.stdin.write("set xlabel 'Learning epoch';")
            p.stdin.write("set ylabel '%s';" % type[i])
            command = ['plot ']
            for j in xrange(target_num):
                command.append("'%s' u 1:%d w l," % (filename, i+j*4+2))
            p.stdin.write(''.join(command)[:-1])
            p.stdin.write('\n')

def plot_period(f, filename):
    params = rnn_print_log.read_parameter(f)
    target_num = int(params['target_num'])
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    p.stdin.write('set nokey;')
    p.stdin.write('set logscale y;')
    p.stdin.write("set title 'Type=Period  File=%s';" % filename)
    p.stdin.write("set xlabel 'Learning epoch';")
    p.stdin.write("set ylabel 'Period';")
    command = ['plot ']
    for i in xrange(target_num):
        command.append("'%s' u 1:%d w l," % (filename, i+2))
    p.stdin.write(''.join(command)[:-1])
    p.stdin.write('\n')

def plot_unknown(f, filename):
    p = re.compile(r'(^#)|(^$)')
    columns = -1
    f.seek(0)
    for line in f:
        if p.match(line) == None:
            n = len(line[:-1].split())
            if columns == -1 or columns > n:
                columns = n
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    p.stdin.write('set nokey;')
    p.stdin.write("set title 'Type=Unknown  File=%s';" % filename)
    command = ['plot ']
    for i in xrange(columns):
        command.append("'%s' u %d w l," % (filename, i + 1))
    p.stdin.write(''.join(command)[:-1])
    p.stdin.write('\n')

def plot_log(f, file, epoch=None):
    multiplot=False
    try:
        p = subprocess.Popen(['gnuplot --version'], stdout=subprocess.PIPE,
                shell=True)
        version = p.communicate()[0].split()[1]
        if float(version) >= 4.2:
            multiplot=True
    except:
        pass
    line = f.readline()
    if re.compile(r'^# STATE FILE').match(line):
        plot_state(f, file, epoch, multiplot)
    elif re.compile(r'^# WEIGHT FILE').match(line):
        plot_weight(f, file, multiplot)
    elif re.compile(r'^# THRESHOLD FILE').match(line):
        plot_threshold(f, file, multiplot)
    elif re.compile(r'^# TAU FILE').match(line):
        plot_tau(f, file)
    elif re.compile(r'^# SIGMA FILE').match(line):
        plot_sigma(f, file)
    elif re.compile(r'^# INIT FILE').match(line):
        plot_init(f, file, epoch)
    elif re.compile(r'^# ADAPT_LR FILE').match(line):
        plot_adapt_lr(f, file)
    elif re.compile(r'^# ERROR FILE').match(line):
        plot_error(f, file)
    elif re.compile(r'^# LYAPUNOV FILE').match(line):
        plot_lyapunov(f, file)
    elif re.compile(r'^# ENTROPY FILE').match(line):
        plot_entropy(f, file, multiplot)
    elif re.compile(r'^# PERIOD FILE').match(line):
        plot_period(f, file)
    else:
        plot_unknown(f, file)


def main():
    epoch = None
    if str.isdigit(sys.argv[1]):
        epoch = int(sys.argv[1])
    for file in sys.argv[2:]:
        f = open(file, 'r')
        plot_log(f, file, epoch)
        f.close()


if __name__ == '__main__':
    main()

