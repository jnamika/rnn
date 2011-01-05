/*
    Copyright (c) 2010-2011, Jun Namikawa <jnamika@gmail.com>

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted, provided that the above
    copyright notice and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef RNN_RUNNER_H
#define RNN_RUNNER_H

#include "rnn.h"


typedef struct rnn_runner {
    int id;
    struct recurrent_neural_network rnn;
} rnn_runner;



int _new_rnn_runner (struct rnn_runner **runner);

void _delete_rnn_runner (struct rnn_runner *runner);


void init_rnn_runner (
        struct rnn_runner *runner,
        FILE *fp);

void free_rnn_runner (struct rnn_runner *runner);

void set_init_state_of_rnn_runner (
        struct rnn_runner *runner,
        int series_id);

void update_rnn_runner (struct rnn_runner *runner);


int rnn_in_state_size_from_runner (struct rnn_runner *runner);
int rnn_c_state_size_from_runner (struct rnn_runner *runner);
int rnn_out_state_size_from_runner (struct rnn_runner *runner);
int rnn_delay_length_from_runner (struct rnn_runner *runner);
int rnn_target_num_from_runner (struct rnn_runner *runner);
double* rnn_in_state_from_runner (struct rnn_runner *runner);
double* rnn_c_state_from_runner (struct rnn_runner *runner);
double* rnn_c_inter_state_from_runner (struct rnn_runner *runner);
double* rnn_out_state_from_runner (struct rnn_runner *runner);
struct rnn_state* rnn_state_from_runner (struct rnn_runner *runner);

#endif

