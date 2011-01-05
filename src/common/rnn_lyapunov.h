/*
    Copyright (c) 2009-2011, Jun Namikawa <jnamika@gmail.com>

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

#ifndef RNN_LYAPUNOV_H
#define RNN_LYAPUNOV_H

#include "rnn.h"


typedef struct rnn_lyapunov_info {
    const struct rnn_state *rnn_s;
    int delay_length;
    int transient_length;

    int dimension;
    double **tmp_matrix;

    int length;
    double **state;
} rnn_lyapunov_info;


void init_rnn_lyapunov_info (
        struct rnn_lyapunov_info *rl_info,
        const struct rnn_state *rnn_s,
        int delay_length,
        int transient_length);

void rnn_lyapunov_info_alloc (struct rnn_lyapunov_info *rl_info);

void free_rnn_lyapunov_info (struct rnn_lyapunov_info *rl_info);

void reset_rnn_lyapunov_info (struct rnn_lyapunov_info *rl_info);


double** rnn_jacobian_matrix_with_delay (
        double** matrix,
        double** tmp_matrix,
        int dimension,
        const struct rnn_parameters *rnn_p,
        const double *prev_c_state,
        const double *c_state,
        const double *out_state,
        int delay_length);


double* rnn_lyapunov_spectrum (
        struct rnn_lyapunov_info *rl_info,
        double *spectrum,
        int spectrum_num);

#endif

