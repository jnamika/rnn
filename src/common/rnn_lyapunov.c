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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "utils.h"
#include "solver.h"
#include "rnn_lyapunov.h"


void init_rnn_lyapunov_info (
        struct rnn_lyapunov_info *rl_info,
        const struct rnn_state *rnn_s,
        int delay_length,
        int truncate_length)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;

    assert(rnn_p->in_state_size == rnn_p->out_state_size ||
            rnn_p->in_state_size == 0);
    assert(delay_length > 0);
    assert(truncate_length >= 0);
    assert(rnn_s->length > truncate_length);

    rl_info->rnn_s = rnn_s;
    rl_info->delay_length = delay_length;
    rl_info->truncate_length = truncate_length;
    rl_info->length = rnn_s->length - truncate_length;
    rl_info->dimension = rnn_p->in_state_size * rl_info->delay_length +
        rnn_p->c_state_size;

    rnn_lyapunov_info_alloc(rl_info);
}

void rnn_lyapunov_info_alloc (struct rnn_lyapunov_info *rl_info)
{
    const struct rnn_parameters *rnn_p = rl_info->rnn_s->rnn_p;
    const int tmp_dimension = rnn_p->out_state_size + rnn_p->c_state_size;
    MALLOC2(rl_info->tmp_matrix, tmp_dimension, tmp_dimension);
    MALLOC2(rl_info->state, rl_info->length, rl_info->dimension);
}

void free_rnn_lyapunov_info (struct rnn_lyapunov_info *rl_info)
{
    FREE2(rl_info->tmp_matrix);
    FREE2(rl_info->state);
}


static double** jacobian_matrix_with_delay (
        double** matrix,
        double** tmp_matrix,
        int dimension,
        const struct rnn_parameters *rnn_p,
        const double *prev_c_state,
        const double *c_state,
        const double *out_state,
        int delay_length)
{
    rnn_jacobian_matrix(tmp_matrix, rnn_p, prev_c_state, c_state, out_state);

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            matrix[i][j] = 0.0;
        }
    }

    const int K = rnn_p->out_state_size * (delay_length - 1);
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        for (int j = 0; j < rnn_p->out_state_size + rnn_p->c_state_size; j++) {
            matrix[i][j+K] = tmp_matrix[i][j];
        }
    }
    for (int i = rnn_p->out_state_size;
            i < rnn_p->out_state_size + rnn_p->c_state_size; i++) {
        for (int j = 0; j < rnn_p->out_state_size + rnn_p->c_state_size; j++) {
            matrix[i+K][j+K] = tmp_matrix[i][j];
        }
    }

    for (int n = 1; n < delay_length; n++) {
        int I = rnn_p->out_state_size * n;
        int J = rnn_p->out_state_size * (n-1);
        for (int i = 0; i < rnn_p->out_state_size; i++) {
            matrix[i+I][i+J] = 1.0;
        }
    }
    return matrix;
}

static double** jacobian_matrix_without_input (
        double** matrix,
        double** tmp_matrix,
        int dimension,
        const struct rnn_parameters *rnn_p,
        const double *prev_c_state,
        const double *c_state,
        const double *out_state)
{
    rnn_jacobian_matrix(tmp_matrix, rnn_p, prev_c_state, c_state, out_state);
    for (int i = 0; i < dimension; i++) {
        int I = i + rnn_p->out_state_size;
        for (int j = 0; j < dimension; j++) {
            matrix[i][j] = tmp_matrix[I][j];
        }
    }
    return matrix;
}


double** rnn_jacobian_matrix_with_delay (
        double** matrix,
        double** tmp_matrix,
        int dimension,
        const struct rnn_parameters *rnn_p,
        const double *prev_c_state,
        const double *c_state,
        const double *out_state,
        int delay_length)
{
    if (rnn_p->in_state_size == 0) {
        jacobian_matrix_without_input(matrix, tmp_matrix, dimension, rnn_p,
                prev_c_state, c_state, out_state);
    } else if (delay_length > 1) {
        jacobian_matrix_with_delay(matrix, tmp_matrix, dimension, rnn_p,
                prev_c_state, c_state, out_state, delay_length);
    } else {
        rnn_jacobian_matrix(matrix, rnn_p, prev_c_state, c_state, out_state);
    }
    return matrix;
}


double** rnn_jacobian_for_lyapunov_spectrum (
        const double* vector,
        int n,
        int t,
        double** matrix,
        void *obj)
{
    struct rnn_lyapunov_info *rl_info = (rnn_lyapunov_info*)obj;
    const struct rnn_state *rnn_s = rl_info->rnn_s;
    const int T = t + rl_info->truncate_length;

    assert(t >= 0);
    assert(rnn_s->length > T);
    assert(vector == rl_info->state[t]);
    assert(n == rl_info->dimension);

    rnn_jacobian_matrix_with_delay(matrix, rl_info->tmp_matrix, n, rnn_s->rnn_p,
            (T==0) ? rnn_s->init_c_state : rnn_s->c_state[T-1],
            rnn_s->c_state[T], rnn_s->out_state[T], rl_info->delay_length);

    return matrix;
}


void reset_rnn_lyapunov_info (struct rnn_lyapunov_info *rl_info)
{
    const struct rnn_state *rnn_s = rl_info->rnn_s;
    const int length = rl_info->length;
    const int delay_length = rl_info->delay_length;
    const int truncate_length = rl_info->truncate_length;
    const int in_state_size = rnn_s->rnn_p->in_state_size;
    const int c_state_size = rnn_s->rnn_p->c_state_size;

    for (int n = 0; n < length; n++) {
        int I = 0;
        for (int k = delay_length-1; k >= 0; k--) {
            const int N = n + truncate_length + k;
            if (N < delay_length) {
                if (N < rnn_s->length) {
                    for (int i = 0; i < in_state_size; i++, I++) {
                        rl_info->state[n][I] = rnn_s->in_state[N][i];
                    }
                } else {
                    for (int i = 0; i < in_state_size; i++, I++) {
                        rl_info->state[n][I] = 0;
                    }
                }
            } else {
                for (int i = 0; i < in_state_size; i++, I++) {
                    rl_info->state[n][I] = rnn_s->out_state[N-delay_length][i];
                }
            }
        }
        if (n + truncate_length == 0) {
            for (int i = 0; i < c_state_size; i++, I++) {
                rl_info->state[n][I] = rnn_s->init_c_inter_state[i];
            }
        } else {
            const int N = n + truncate_length;
            for (int i = 0; i < c_state_size; i++, I++) {
                rl_info->state[n][I] = rnn_s->c_inter_state[N-1][i];
            }
        }
    }
}


/* this function returns the Lyapunov spectrum of RNN */
double* rnn_lyapunov_spectrum (
        struct rnn_lyapunov_info *rl_info,
        double *spectrum,
        int spectrum_size)
{
    reset_rnn_lyapunov_info(rl_info);
    return lyapunov_spectrum((const double* const*)rl_info->state,
            rl_info->length, spectrum_size, rl_info->dimension, 1,
            rnn_jacobian_for_lyapunov_spectrum, rl_info, spectrum, NULL, NULL);
}

