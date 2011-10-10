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
        prange = '[][-1:1]'
        pass
    elif re.search('SOFTMAX_TYPE', output_type):
        prange = '[][0:1]'
    else:
        prange = ''
    tmp = tempfile.NamedTemporaryFile('w+')
    sys.stdout = tmp
    rnn_print_log.print_state(f, epoch)
    sys.stdout.flush()
    ptype = []
    ptype.append(('Target', out_state_size, lambda x: 2 * x + 2, prange))
    ptype.append(('Output', out_state_size, lambda x: 2 * x + 3, prange))
    ptype.append(('Context', c_state_size, lambda x: x + 2 * out_state_size +
        2, ''))
    if multiplot:
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        gnuplot = lambda s: p.stdin.write(s.encode())
        gnuplot('set nokey;')
        gnuplot("set multiplot layout 3,1 title 'Type=State  File=%s';" %
                filename)
        for v in ptype:
            gnuplot("set xlabel 'Time step';")
            gnuplot("set ylabel '%s';" % v[0])
            command = ['plot %s ' % v[3]]
            for i in range(v[1]):
                command.append("'%s' u 1:%d w l," % (tmp.name, v[2](i)))
            gnuplot(''.join(command)[:-1])
            gnuplot('\n')
        gnuplot('unset multiplot\n')
        gnuplot('exit\n')
        p.wait()
    else:
        for v in ptype:
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            gnuplot = lambda s: p.stdin.write(s.encode())
            gnuplot('set nokey;')
            gnuplot("set title 'Type=%s  File=%s';" % (v[0], filename))
            gnuplot("set xlabel 'Time step';")
            gnuplot("set ylabel '%s';" % v[0])
            command = ['plot %s ' % v[3]]
            for i in range(v[1]):
                command.append("'%s' u 1:%d w l," % (tmp.name, v[2](i)))
            gnuplot(''.join(command)[:-1])
            gnuplot('\n')
            gnuplot('exit\n')
            p.wait()
    sys.stdout = sys.__stdout__

def plot_weight(f, filename, multiplot=False):
    params = rnn_print_log.read_parameter(f)
    in_state_size = int(params['in_state_size'])
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    index_i2c, index_c2c, index_c2o = [], [], []
    s = [x+2 for x in range(c_state_size *
        (in_state_size + c_state_size + out_state_size))]
    for i in range(c_state_size):
        index_i2c.extend(s[:in_state_size])
        s = s[in_state_size:]
        index_c2c.extend(s[:c_state_size])
        s = s[c_state_size:]
    for i in range(out_state_size):
        index_c2o.extend(s[:c_state_size])
        s = s[c_state_size:]
    ptype = []
    ptype.append(('Weight (input to context)', index_i2c))
    ptype.append(('Weight (context to context)', index_c2c))
    ptype.append(('Weight (context to output)', index_c2o))
    if multiplot:
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        gnuplot = lambda s: p.stdin.write(s.encode())
        gnuplot('set nokey;')
        gnuplot("set multiplot layout 3,1 title 'Type=Weight File=%s';" %
                filename)
        for v in ptype:
            gnuplot("set xlabel 'Learning epoch';")
            gnuplot("set ylabel '%s';" % v[0])
            command = ['plot ']
            for i in v[1]:
                command.append("'%s' u 1:%d w l," % (filename, i))
            gnuplot(''.join(command)[:-1])
            gnuplot('\n')
        gnuplot('unset multiplot\n')
    else:
        for v in ptype:
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            gnuplot = lambda s: p.stdin.write(s.encode())
            gnuplot('set nokey;')
            gnuplot("set title 'Type=Weight  File=%s';" % filename)
            gnuplot("set xlabel 'Learning epoch';")
            gnuplot("set ylabel '%s';" % v[0])
            command = ['plot ']
            for i in v[1]:
                command.append("'%s' u 1:%d w l," % (filename, i))
            gnuplot(''.join(command)[:-1])
            gnuplot('\n')

def plot_threshold(f, filename, multiplot=False):
    params = rnn_print_log.read_parameter(f)
    c_state_size = int(params['c_state_size'])
    out_state_size = int(params['out_state_size'])
    ptype = []
    ptype.append(('Threshold (context)', c_state_size, lambda x: x + 2))
    ptype.append(('Threshold (output)', out_state_size, lambda x: x +
        c_state_size + 2))
    if multiplot:
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        gnuplot = lambda s: p.stdin.write(s.encode())
        gnuplot('set nokey;')
        gnuplot('set multiplot layout 2,1 title ' +
                "'Type=Threshold File=%s';" % filename)
        for v in ptype:
            gnuplot("set xlabel 'Learning epoch';")
            gnuplot("set ylabel '%s';" % v[0])
            command = ['plot ']
            for i in range(v[1]):
                command.append("'%s' u 1:%d w l," % (filename, v[2](i)))
            gnuplot(''.join(command)[:-1])
            gnuplot('\n')
        gnuplot('unset multiplot\n')
    else:
        for v in ptype:
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            gnuplot = lambda s: p.stdin.write(s.encode())
            gnuplot('set nokey;')
            gnuplot("set title 'Type=Threshold  File=%s';" % filename)
            gnuplot("set xlabel 'Learning epoch';")
            gnuplot("set ylabel '%s';" % v[0])
            command = ['plot ']
            for i in range(v[1]):
                command.append("'%s' u 1:%d w l," % (filename, v[2](i)))
            gnuplot(''.join(command)[:-1])
            gnuplot('\n')

def plot_tau(f, filename):
    params = rnn_print_log.read_parameter(f)
    c_state_size = int(params['c_state_size'])
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    gnuplot = lambda s: p.stdin.write(s.encode())
    gnuplot('set nokey;')
    gnuplot("set title 'Type=Time-constant  File=%s';" % filename)
    gnuplot("set xlabel 'Learning epoch';")
    gnuplot("set ylabel 'Time constant';")
    command = ['plot ']
    for i in range(c_state_size):
        command.append("'%s' u 1:%d w l," % (filename, i+2))
    gnuplot(''.join(command)[:-1])
    gnuplot('\n')

def plot_sigma(f, filename):
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    gnuplot = lambda s: p.stdin.write(s.encode())
    gnuplot('set nokey;')
    gnuplot("set title 'Type=Variance  File=%s';" % filename)
    gnuplot("set xlabel 'Learning epoch';")
    gnuplot("set ylabel 'Variance';")
    gnuplot("plot '%s' u 1:3 w l" % filename)
    gnuplot('\n')

def plot_init(f, filename, epoch):
    params = rnn_print_log.read_parameter(f)
    c_state_size = int(params['c_state_size'])
    tmp = tempfile.NamedTemporaryFile('w+')
    sys.stdout = tmp
    rnn_print_log.print_init(f, epoch)
    sys.stdout.flush()
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    gnuplot = lambda s: p.stdin.write(s.encode())
    gnuplot('set nokey;')
    gnuplot("set title 'Type=Init  File=%s';" % filename)
    gnuplot("set xlabel 'x';")
    gnuplot("set ylabel 'y';")
    gnuplot('set pointsize 3;')
    command = ['plot ']
    index = [(2*x,(2*x+1)%c_state_size) for x in range(c_state_size) if 2*x <
            c_state_size]
    for x in index:
        command.append("'%s' u %d:%d w p," % (tmp.name, x[0]+2, x[1]+2))
    gnuplot(''.join(command)[:-1])
    gnuplot('\n')
    gnuplot('exit\n')
    p.wait()
    sys.stdout = sys.__stdout__

def plot_adapt_lr(f, filename):
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    gnuplot = lambda s: p.stdin.write(s.encode())
    gnuplot('set nokey;')
    gnuplot('set logscale y;')
    gnuplot("set title 'Type=Learning-rate  File=%s';" % filename)
    gnuplot("set xlabel 'Learning epoch';")
    gnuplot("set ylabel 'Current learning rate / Initial learning rate';")
    gnuplot("plot '%s' u 1:2 w l" % filename)
    gnuplot('\n')

def plot_error(f, filename):
    params = rnn_print_log.read_parameter(f)
    target_num = int(params['target_num'])
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    gnuplot = lambda s: p.stdin.write(s.encode())
    gnuplot('set nokey;')
    gnuplot('set logscale y;')
    gnuplot("set title 'Type=Error  File=%s';" % filename)
    gnuplot("set xlabel 'Learning epoch';")
    gnuplot("set ylabel 'Error / (Length times Dimension)';")
    command = ['plot ']
    for i in range(target_num):
        command.append("'%s' u 1:%d w l," % (filename, i+2))
    gnuplot(''.join(command)[:-1])
    gnuplot('\n')

def plot_lyapunov(f, filename):
    params = rnn_print_log.read_parameter(f)
    target_num = int(params['target_num'])
    ls_size = int(params['lyapunov_spectrum_size'])
    p = re.compile(r'(^#)|(^$)')
    tmp = tempfile.NamedTemporaryFile('w+')
    for line in f:
        if p.match(line) == None:
            input = list(map(float, line[:-1].split()))
            lyapunov = [input[0]]
            for i in range(ls_size):
                lyapunov.append(sum(input[i+1::ls_size]) / target_num)
            tmp.write('%s\n' % '\t'.join([str(x) for x in lyapunov]))
    tmp.flush()
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    gnuplot = lambda s: p.stdin.write(s.encode())
    gnuplot('set nokey;')
    gnuplot("set title 'Type=Lyapunov  File=%s';" % filename)
    gnuplot("set xlabel 'Learning epoch';")
    gnuplot("set ylabel 'Lyapunov';")
    command = ['plot 0 w l lt 0']
    for i in range(ls_size):
        command.append("'%s' u 1:%d w l lt %d" % (tmp.name, i+2, i+1))
    gnuplot(','.join(command))
    gnuplot('\n')
    gnuplot('exit\n')
    p.wait()

def plot_entropy(f, filename, multiplot=False):
    params = rnn_print_log.read_parameter(f)
    target_num = int(params['target_num'])
    ptype = ['KL-divergence', 'generation-rate', 'entropy(target)',
            'entropy(out)']
    if multiplot:
        p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                shell=True)
        gnuplot = lambda s: p.stdin.write(s.encode())
        gnuplot('set nokey;')
        gnuplot('set multiplot layout 3,1 title ' +
                "'Type=Entropy File=%s';" % filename)
        for i in [0, 1, 3]:
            gnuplot("set xlabel 'Learning epoch';")
            gnuplot("set ylabel '%s';" % ptype[i])
            command = ['plot ']
            for j in range(target_num):
                command.append("'%s' u 1:%d w l," % (filename, i+j*4+2))
            gnuplot(''.join(command)[:-1])
            gnuplot('\n')
        gnuplot('unset multiplot\n')
    else:
        for i in [0, 1, 3]:
            p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
                    shell=True)
            gnuplot = lambda s: p.stdin.write(s.encode())
            gnuplot('set nokey;')
            gnuplot("set title 'Type=%s  File=%s';" % (ptype[i], filename))
            gnuplot("set xlabel 'Learning epoch';")
            gnuplot("set ylabel '%s';" % ptype[i])
            command = ['plot ']
            for j in range(target_num):
                command.append("'%s' u 1:%d w l," % (filename, i+j*4+2))
            gnuplot(''.join(command)[:-1])
            gnuplot('\n')

def plot_period(f, filename):
    params = rnn_print_log.read_parameter(f)
    target_num = int(params['target_num'])
    p = subprocess.Popen(['gnuplot -persist'], stdin=subprocess.PIPE,
            shell=True)
    gnuplot = lambda s: p.stdin.write(s.encode())
    gnuplot('set nokey;')
    gnuplot('set logscale y;')
    gnuplot("set title 'Type=Period  File=%s';" % filename)
    gnuplot("set xlabel 'Learning epoch';")
    gnuplot("set ylabel 'Period';")
    command = ['plot ']
    for i in range(target_num):
        command.append("'%s' u 1:%d w l," % (filename, i+2))
    gnuplot(''.join(command)[:-1])
    gnuplot('\n')

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
    gnuplot = lambda s: p.stdin.write(s.encode())
    gnuplot('set nokey;')
    gnuplot("set title 'Type=Unknown  File=%s';" % filename)
    command = ['plot ']
    for i in range(columns):
        command.append("'%s' u %d w l," % (filename, i + 1))
    gnuplot(''.join(command)[:-1])
    gnuplot('\n')

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

