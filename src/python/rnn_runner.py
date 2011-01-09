# -*- coding:utf-8 -*-

import sys
import os
import datetime
from ctypes import *

libc = cdll.LoadLibrary("libc.so.6")
librunner = cdll.LoadLibrary("%s/librunner.so" % os.path.dirname(sys.argv[0]))

libc.fopen.argtypes = [c_char_p, c_char_p]
libc.fopen.restype = c_void_p
libc.fclose.argtypes = [c_void_p]
libc.fwrite.argtypes = [c_void_p, c_uint, c_uint, c_void_p]
libc.fread.argtypes = [c_void_p, c_uint, c_uint, c_void_p]

librunner.init_genrand.argtype = c_ulong
librunner.rnn_in_state_from_runner.restype = POINTER(c_double)
librunner.rnn_c_state_from_runner.restype = POINTER(c_double)
librunner.rnn_c_inter_state_from_runner.restype = POINTER(c_double)
librunner.rnn_out_state_from_runner.restype = POINTER(c_double)


def init_genrand(seed):
    librunner.init_genrand(c_ulong(seed))


class rnn_runner:
    def __init__(self, librunner=librunner):
        self.runner = c_void_p()
        self.librunner = librunner
        self.librunner._new_rnn_runner(byref(self.runner))
        self.is_initialized = False

    def __del__(self):
        self.free()
        self.librunner._delete_rnn_runner(self.runner)

    def init(self, file_name):
        fp = libc.fopen(file_name, "r")
        if (fp):
            self.free()
            self.librunner.init_rnn_runner(self.runner, fp)
            self.is_initialized = True
            libc.fclose(fp)
        else:
            raise IOError("[Errno 2] No such file or directory: '%s'" %
                    file_name)

    def free(self):
        if (self.is_initialized):
            self.librunner.free_rnn_runner(self.runner)
        self.is_initialized = False

    def set_time_series_id(self, id=0):
        self.librunner.set_init_state_of_rnn_runner(self.runner, id)

    def target_num(self):
        return self.librunner.rnn_target_num_from_runner(self.runner)

    def in_state_size(self):
        return self.librunner.rnn_in_state_size_from_runner(self.runner)

    def c_state_size(self):
        return self.librunner.rnn_c_state_size_from_runner(self.runner)

    def out_state_size(self):
        return self.librunner.rnn_out_state_size_from_runner(self.runner)

    def delay_length(self):
        return self.librunner.rnn_delay_length_from_runner(self.runner)

    def output_type(self):
        return self.librunner.rnn_output_type_from_runner(self.runner)

    def update(self):
        self.librunner.update_rnn_runner(self.runner)

    def closed_loop(self, length):
        for n in xrange(length):
            self.update()
            yield self.out_state(), self.c_inter_state()

    def in_state(self, in_state=None):
        x = self.librunner.rnn_in_state_from_runner(self.runner)
        if in_state != None:
            for i in xrange(len(in_state)):
                x[i] = c_double(in_state[i])
        return [x[i] for i in xrange(self.in_state_size())]

    def c_state(self, c_state=None):
        x = self.librunner.rnn_c_state_from_runner(self.runner)
        if c_state != None:
            for i in xrange(len(c_state)):
                x[i] = c_double(c_state[i])
        return [x[i] for i in xrange(self.c_state_size())]

    def c_inter_state(self, c_inter_state=None):
        x = self.librunner.rnn_c_inter_state_from_runner(self.runner)
        if c_inter_state != None:
            for i in xrange(len(c_inter_state)):
                x[i] = c_double(c_inter_state[i])
        return [x[i] for i in xrange(self.c_state_size())]

    def out_state(self, out_state=None):
        x = self.librunner.rnn_out_state_from_runner(self.runner)
        if out_state != None:
            for i in xrange(len(out_state)):
                x[i] = c_double(out_state[i])
        return [x[i] for i in xrange(self.out_state_size())]


def main():
    seed, steps, index = map(lambda x: int(x) if str.isdigit(x) else 0,
            sys.argv[1:4])
    rnn_file = sys.argv[4]
    if seed == 0:
        now = datetime.datetime.utcnow()
        seed = ((now.hour * 3600 + now.minute * 60 + now.second) *
                now.microsecond)
    init_genrand(seed % 4294967295)
    runner = rnn_runner()
    runner.init(rnn_file)
    runner.set_time_series_id(index)
    for x,y in runner.closed_loop(steps):
        print '\t'.join([str(x) for x in x])

if __name__ == "__main__":
    main()

