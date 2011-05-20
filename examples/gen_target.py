# -*- coding:utf-8 -*-

import math
import random


def generate_Tn_torus(length, truncate_length=0, parameters=[(0,1,1,0)]):
    theta = [x[0] for x in parameters]
    inc = [x[1] for x in parameters]
    amplitude = [x[2] for x in parameters]
    base = [x[3] for x in parameters]
    for n in xrange(length + truncate_length):
        if (n >= truncate_length):
            yield map(lambda x, a, b: a * math.sin(x) + b,
                    theta, amplitude, base)
        theta = map(lambda x, y: math.fmod(x + y, 2 * math.pi), theta, inc)


def generate_logistic_map(length, truncate_length=0, a=4.0, x=None):
    if (x == None):
        x = random.random()
    for n in xrange(length + truncate_length):
        if (n >= truncate_length):
            yield x
        x = a * x * (1 - x)


def generate_henon_map(length, truncate_length=0, a=1.4, b=0.3, x=None,
        y=None):
    if (x == None):
        x = 0.625 * random.random()
    if (y == None):
        y = 0.3 * random.random()
    for n in xrange(length + truncate_length):
        if (n >= truncate_length):
            yield x, y
        next_x = y + 1 - a * x * x
        y = b * x
        x = next_x


def Runge_Kutta_4(df, x, delta, t):
    dx1 = df(x, t)
    y = map(lambda x, dx: x + delta * (dx / 2.0), x, dx1)
    dx2 = df(y, t + delta / 2.0)
    y = map(lambda x, dx: x + delta * (dx / 2.0), x, dx2)
    dx3 = df(y, t + delta / 2.0)
    y = map(lambda x, dx: x + delta * dx, x, dx3)
    dx4 = df(y, t + delta)
    y = map(lambda dx1, dx2, dx3, dx4: (dx1 / 6.0) + (dx2 / 3.0) +
            (dx3 / 3.0) + (dx4 / 6.0), dx1, dx2, dx3, dx4)
    return map(lambda x, y: x + delta * y, x, y)


def generate_Lorenz_attractor(length, truncate_length=0, p=10, r=28, b=8.0/3.0,
        tau=1, delta=0.1, x=None, y=None, z=None):
    if (x == None):
        x = 25 * random.random()
    if (y == None):
        y = 35 * random.random()
    if (z == None):
        z = 50 * (random.random() + 0.5)
    v = (x, y, z)
    def Lorenz_derivation(x, t):
        dx = []
        dx.append(p * (x[1] - x[0]))
        dx.append(-x[0] * x[2] + r * x[0] - x[1])
        dx.append(x[0] * x[1] - b * x[2])
        return map(lambda x: x / tau, dx)
    for n in xrange(length + truncate_length):
        if (n >= truncate_length):
            yield v
        v = tuple(Runge_Kutta_4(Lorenz_derivation, list(v), delta, n * delta))


def generate_van_der_Pol_attractor(length, truncate_length=0, a=1.0, b=0.25,
        delta=0.1, x=None, y=None):
    if (x == None):
        x = random.random()
    if (y == None):
        y = random.random()
    v = (x, y)
    def van_der_Pol_derivation(x, t):
        dx = []
        dx.append(a * x[1])
        dx.append(a * (b * (1 - x[0] * x[0]) * x[1] - x[0]))
        return dx
    for n in xrange(length + truncate_length):
        if (n >= truncate_length):
            yield v
        v = tuple(Runge_Kutta_4(van_der_Pol_derivation, list(v), delta,
            n * delta))


def generate_composite_dynamics(length, f, v, truncate_length=0, p_matrix=None,
        primitive_len=None,init_f=None):
    if (p_matrix == None):
        p_matrix = []
        for i in xrange(len(f)):
            p_matrix.append([])
            for j in xrange(len(f)):
                p_matrix[i].append(1.0/len(f))
    if (primitive_len == None):
        primitive_len = [1 for x in xrange(len(f))]
    if (init_f == None):
        init_f = random.randint(0, len(f)-1)
    c = init_f
    t = 0
    for n in xrange(length + truncate_length):
        v = f[c](v)
        if (n >= truncate_length):
            yield v
        t = t + 1
        if (t >= primitive_len[c]):
            t = t - primitive_len[c]
            p = random.random()
            q = 0
            for i in xrange(len(f)):
                q = q + p_matrix[c][i]
                if (p <= q):
                    c = i
                    break



def generate_golden_mean_shift(length, truncate_length=0, probability=0.5):
    c = 0
    for n in xrange(length + truncate_length):
        if (c == 0):
            p = random.random()
            if (p < probability):
                c = 0
            else:
                c = 1
        else:
            c = 0
        if (n >= truncate_length):
            yield c

def generate_01x_shift(length, truncate_length=0, probability=0.5):
    for n in xrange(length + truncate_length):
        c = n % 3
        if (c == 2):
            p = random.random()
            if (p < probability):
                c = 0
            else:
                c = 1
        if (n >= truncate_length):
            yield c


def generate_Morse_shift(length, truncate_length=0):
    s = [0]
    for n in xrange(length + truncate_length):
        if (n == len(s)):
            for i in xrange(len(s)):
                s.append(0 if s[i]==1 else 1)
        if (n >= truncate_length):
            yield s[n]


def generate_Reber_grammar(length, truncate_length=0, probability=0.5):
    c = 0
    trans= []
    trans.append([(1,'T'), (2, 'P')])
    trans.append([(1,'S'), (3, 'X')])
    trans.append([(2,'T'), (4, 'V')])
    trans.append([(2,'X'), (5, 'S')])
    trans.append([(3,'P'), (5, 'V')])
    trans.append([(6,'E'), (6, 'E')])
    trans.append([(0,'B'), (0, 'B')])
    for n in xrange(length + truncate_length):
        p = random.random()
        if p < probability:
            k = trans[c][0][1]
            c = trans[c][0][0]
        else:
            k = trans[c][1][1]
            c = trans[c][1][0]
        if (n >= truncate_length):
            yield (c, k)



def print_logistic_map(length, truncate_length=0, a=4.0, x=None):
    for x in generate_logistic_map(length, truncate_length, a, x):
        x = 0.8 * (2 * x - 1)
        print '%f' % x


def print_henon_map(length, truncate_length=0, a=1.4, b=0.3, x=None, y=None):
    for x, y in generate_henon_map(length, truncate_length, a, b, x, y):
        x = 0.625 * x
        print '%f\t%f' % (x, y)


def print_sin_curve(length, period, truncate_length=0):
    for v in generate_Tn_torus(length=length,
            parameters=[(0,(2*math.pi)/period,0.8,0)], \
                    truncate_length=truncate_length):
        print '\t'.join([str(x) for x in v])


def print_Lissajous_0curve(length, period, truncate_length=0):
    for v in generate_Tn_torus(length=length,
            parameters=[(0,(2*math.pi)/period,0.8,0),
                ((0.5*math.pi,(2*math.pi)/period,0.8,0))], \
                        truncate_length=truncate_length):
        print '\t'.join([str(x) for x in v])

def print_Lissajous_8curve(length, period, truncate_length=0):
    for v in generate_Tn_torus(length=length,
            parameters=[(0,(4*math.pi)/period,0.8,0),
                ((0,(2*math.pi)/period,0.8,0))], \
                        truncate_length=truncate_length):
        print '\t'.join([str(x) for x in v])


def print_Lorenz_attractor(length, tau=1, truncate_length=0):
    for v in generate_Lorenz_attractor(length=length, tau=tau,
            truncate_length=truncate_length):
        x = v[0] / 25
        y = v[1] / 35
        z = (v[2] / 50) - 0.5
        print '%f\t%f\t%f' % (x, y, z)


def print_van_der_Pol_attractor(length, a=1, truncate_length=0):
    for v in generate_van_der_Pol_attractor(length=length, a=a,
            truncate_length=truncate_length):
        x = 0.2 * v[0]
        y = 0.2 * v[1]
        print '%f\t%f' % (x, y)


def print_comp_Lissajous_08curves(length, period, probability=0.5,
        truncate_length=0):
    def Lissajous0(x):
        x[0] = 0.8 * math.sin(x[2])
        x[1] = 0.8 * math.sin(x[3])
        x[2] = math.fmod(x[2] + (2 * math.pi) / period, 2 * math.pi)
        x[3] = math.fmod(x[3] + (2 * math.pi) / period, 2 * math.pi)
        return x
    def Lissajous8(x):
        x[0] = 0.6 * math.sin(2 * x[2])
        x[1] = 0.8 * math.sin(x[3])
        x[2] = math.fmod(x[2] + (2 * math.pi) / period, 2 * math.pi)
        x[3] = math.fmod(x[3] + (2 * math.pi) / period, 2 * math.pi)
        return x
    for v in generate_composite_dynamics(length=length,
            f=[Lissajous0, Lissajous8],
            v=[0,0,0,0.5 * math.pi],
            primitive_len=[period,period],
            p_matrix=[[1-probability,probability],[probability,1-probability]],
            truncate_length=truncate_length):
        print '\t'.join([str(x) for x in v[:2]])


def print_comp_Lissajous_0LRcurves(length, period, probability=0.5,
        truncate_length=0):
    def Lissajous0L(x):
        x[0] = 0.4 * math.sin(x[2]) - 0.4
        x[1] = 0.4 * math.sin(x[3])
        x[2] = math.fmod(x[2] + (2 * math.pi) / period, 2 * math.pi)
        x[3] = math.fmod(x[3] + (2 * math.pi) / period, 2 * math.pi)
        return x
    def Lissajous0R(x):
        x[0] = -0.4 * math.sin(x[2]) + 0.4
        x[1] = 0.4 * math.sin(x[3])
        x[2] = math.fmod(x[2] + (2 * math.pi) / period, 2 * math.pi)
        x[3] = math.fmod(x[3] + (2 * math.pi) / period, 2 * math.pi)
        return x
    for v in generate_composite_dynamics(length=length,
            f=[Lissajous0L, Lissajous0R],
            v=[0,0,0.5 * math.pi,0],
            primitive_len=[period,period],
            p_matrix=[[1-probability,probability],[probability,1-probability]],
            truncate_length=truncate_length):
        print '\t'.join([str(x) for x in v[:2]])


def print_comp_Lissajous_0SLcurves(length, period, probability=0.5,
        truncate_length=0):
    def Lissajous0S(x):
        x[0] = 0.4 * math.sin(x[2]) + 0.4
        x[1] = 0.4 * math.sin(x[3])
        x[2] = math.fmod(x[2] + (2 * math.pi) / period, 2 * math.pi)
        x[3] = math.fmod(x[3] + (2 * math.pi) / period, 2 * math.pi)
        return x
    def Lissajous0L(x):
        x[0] = 0.8 * math.sin(x[2])
        x[1] = 0.8 * math.sin(x[3])
        x[2] = math.fmod(x[2] + (2 * math.pi) / period, 2 * math.pi)
        x[3] = math.fmod(x[3] + (2 * math.pi) / period, 2 * math.pi)
        return x
    for v in generate_composite_dynamics(length=length,
            f=[Lissajous0S, Lissajous0L],
            v=[0,0,0.5 * math.pi,0],
            primitive_len=[period,period],
            p_matrix=[[1-probability,probability],[probability,1-probability]],
            truncate_length=truncate_length):
        print '\t'.join([str(x) for x in v[:2]])


def print_comp_Lissajous_9curves(length, period, truncate_length=0):
    p4 = period / 4.0
    p = period
    primitive_len = [p4,p4,p4,p4,p,p,p,p,p,p,p,p]
    p_matrix = [
            [0,0.8,0,0,0.1,0.1,0,0,0,0,0,0],
            [0,0,0.8,0,0,0,0.1,0.1,0,0,0,0],
            [0,0,0,0.8,0,0,0,0,0.1,0.1,0,0],
            [0.8,0,0,0,0,0,0,0,0,0,0.1,0.1],
            [0,0.2,0,0,0.4,0.4,0,0,0,0,0,0],
            [0,0.2,0,0,0.4,0.4,0,0,0,0,0,0],
            [0,0,0.2,0,0,0,0.4,0.4,0,0,0,0],
            [0,0,0.2,0,0,0,0.4,0.4,0,0,0,0],
            [0,0,0,0.2,0,0,0,0,0.4,0.4,0,0],
            [0,0,0,0.2,0,0,0,0,0.4,0.4,0,0],
            [0.2,0,0,0,0,0,0,0,0,0,0.4,0.4],
            [0.2,0,0,0,0,0,0,0,0,0,0.4,0.4]]
    def Lissajous0(x):
        x[0] = math.sin(x[2])
        x[1] = math.sin(x[3])
        x[2] = math.fmod(x[2] + (2 * math.pi) / period, 2 * math.pi)
        x[3] = math.fmod(x[3] + (2 * math.pi) / period, 2 * math.pi)
        return x
    def Lissajous8(x):
        x[0] = 0.75 * math.sin(2 * x[2])
        x[1] = math.sin(x[3])
        x[2] = math.fmod(x[2] + (2 * math.pi) / period, 2 * math.pi)
        x[3] = math.fmod(x[3] + (2 * math.pi) / period, 2 * math.pi)
        return x
    def Lissajous0C(x):
        x = Lissajous0(x)
        x[0] = -0.2 * x[0]
        x[1] = 0.2 * x[1]
        return x
    def Lissajous0LL(x):
        x = Lissajous0(x)
        x[0] = 0.32 * x[0] - 0.52
        x[1] = 0.32 * x[1]
        return x
    def Lissajous0LS(x):
        x = Lissajous0(x)
        x[0] = 0.16 * x[0] - 0.52 + 0.16
        x[1] = 0.16 * x[1]
        return x
    def Lissajous0RL(x):
        x = Lissajous0(x)
        x[0] = 0.32 * x[0] + 0.52
        x[1] = 0.32 * x[1]
        return x
    def Lissajous0RS(x):
        x = Lissajous0(x)
        x[0] = 0.16 * x[0] + 0.52 - 0.16
        x[1] = 0.16 * x[1]
        return x
    def Lissajous0T(x):
        x = Lissajous0(x)
        x[0] = 0.32 * x[0]
        x[1] = -0.32 * x[1] + 0.52
        return x
    def Lissajous8T(x):
        x = Lissajous8(x)
        x[0] = 0.32 * x[0]
        x[1] = -0.32 * x[1] + 0.52
        return x
    def Lissajous0B(x):
        x = Lissajous0(x)
        x[0] = 0.32 * x[0]
        x[1] = -0.32 * x[1] - 0.52
        return x
    def Lissajous8B(x):
        x = Lissajous8(x)
        x[0] = 0.32 * x[0]
        x[1] = -0.32 * x[1] - 0.52
        return x
    f = [Lissajous0C, Lissajous0C, Lissajous0C, Lissajous0C, Lissajous0LL,
            Lissajous0LS, Lissajous0B, Lissajous8B, Lissajous0RL, Lissajous0RS,
            Lissajous0T, Lissajous8T]
    for v in generate_composite_dynamics(length=length, f=f,
            v=[0,0,0,0.5*math.pi], primitive_len=primitive_len,
            p_matrix=p_matrix, init_f=0, truncate_length=truncate_length):
        print '\t'.join([str(x) for x in v[:2]])



def print_golden_mean_shift(length, truncate_length=0, probability=0.5,
        output_type='tanh'):
    for x in generate_golden_mean_shift(length, truncate_length, probability):
        if (output_type == 'tanh'):
            print '%d' % 1 if x == 1 else -1
        elif (output_type == 'softmax'):
            y = 0 if x == 1 else 1
            print '%d\t%d' % (x, y)
        else:
            print '%d' % x

def print_01x_shift(length, truncate_length=0, probability=0.5,
        output_type='tanh'):
    for x in generate_01x_shift(length, truncate_length, probability):
        if (output_type == 'tanh'):
            print '%d' % 1 if x == 1 else -1
        elif (output_type == 'softmax'):
            y = 0 if x == 1 else 1
            print '%d\t%d' % (x, y)
        else:
            print '%d' % x


def print_Morse_shift(length, truncate_length=0, output_type='tanh'):
    for x in generate_Morse_shift(length, truncate_length=truncate_length):
        if (output_type == 'tanh'):
            print '%d' % 1 if x == 1 else -1
        elif (output_type == 'softmax'):
            y = 0 if x == 1 else 1
            print '%d\t%d' % (x, y)
        else:
            print '%d' % x

def print_Reber_grammar(length, truncate_length=0, probability=0.5):
    s = {'T':0, 'P':1, 'S':2, 'X':3, 'V':4, 'E':5, 'B':6}
    for x in generate_Reber_grammar(length, truncate_length, probability):
        output = map(lambda n: 1 if n == s[x[1]] else 0, xrange(len(s)))
        print '\t'.join([str(x) for x in output])


def main():
    print_logistic_map(100)
    print_henon_map(100)
    print_sin_curve(100, 20)
    print_Lissajous_0curve(100, 20)
    print_Lissajous_8curve(100, 20)
    print_Lorenz_attractor(1000,truncate_length=1000)
    print_van_der_Pol_attractor(1000,truncate_length=1000)
    print_comp_Lissajous_08curves(1000, 32)
    print_comp_Lissajous_0LRcurves(1000, 32)
    print_comp_Lissajous_0SLcurves(1000, 32)
    print_comp_Lissajous_9curves(10000, 25)
    print_golden_mean_shift(100)
    print_01x_shift(100)
    print_Morse_shift(100)
    print_Reber_grammar(100)


if __name__ == '__main__':
    main()

