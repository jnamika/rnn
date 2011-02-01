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
#include <stddef.h>
#include <string.h>
#include <math.h>

#include "minunit.h"
#include "my_assert.h"
#include "utils.h"
#include "rnn_lyapunov.h"


/* assert functions */

void assert_reset_rnn_lyapunov_info (struct rnn_lyapunov_info *rl_info)
{
    const struct rnn_state *rnn_s;

    rnn_s = rl_info->rnn_s;
    reset_rnn_lyapunov_info(rl_info);

    for (int n = 0; n < rl_info->length; n++) {
        int is_equal = 1;
        int N = n + rl_info->truncate_length;
        for (int m = 0; m < rl_info->delay_length; m++) {
            int I = rnn_s->rnn_p->in_state_size *
                (rl_info->delay_length - (m+1));
            if (N+m < rl_info->delay_length) {
                if (memcmp(rnn_s->in_state[N+m], rl_info->state[n]+I,
                        rnn_s->rnn_p->in_state_size * sizeof(double))) {
                    is_equal = 0;
                }
            } else {
                if (memcmp(rnn_s->out_state[N+m - rl_info->delay_length],
                            rl_info->state[n]+I, rnn_s->rnn_p->in_state_size *
                            sizeof(double))) {
                    is_equal = 0;
                }
            }
        }
        int I = rnn_s->rnn_p->in_state_size * rl_info->delay_length;
        if (N == 0) {
            if (memcmp(rnn_s->init_c_inter_state, rl_info->state[n]+I,
                        rnn_s->rnn_p->c_state_size * sizeof(double))) {
                is_equal = 0;
            }
        } else {
            if (memcmp(rnn_s->c_inter_state[N-1], rl_info->state[n]+I,
                        rnn_s->rnn_p->c_state_size * sizeof(double))) {
                is_equal = 0;
            }
        }
        mu_assert(is_equal);
    }
}


/* test functions */

static void test_init_rnn_lyapunov_info (
        const struct rnn_state *rnn_s)
{
    struct rnn_lyapunov_info rl_info;
    assert_exit_call(init_rnn_lyapunov_info, &rl_info, rnn_s, 0, 0);
    assert_exit_call(init_rnn_lyapunov_info, &rl_info, rnn_s, 1, -1);
    assert_exit_call(init_rnn_lyapunov_info, &rl_info, rnn_s, 1, rnn_s->length);
    if (rnn_s->rnn_p->in_state_size == rnn_s->rnn_p->out_state_size ||
            rnn_s->rnn_p->in_state_size == 0) {
        assert_exit_nocall(init_rnn_lyapunov_info, &rl_info, rnn_s, 1, 0);
        free_rnn_lyapunov_info(&rl_info);
    }
}


double** rnn_jacobian_for_lyapunov_spectrum (const double* vector, int n, int t,
        double** matrix, void *obj);

static void test_rnn_jacobian_for_lyapunov_spectrum (struct rnn_state *rnn_s)
{
    if (rnn_s->rnn_p->in_state_size != rnn_s->rnn_p->out_state_size &&
            rnn_s->rnn_p->in_state_size != 0) {
        return;
    }

    int in_and_c_state_size, out_and_c_state_size;
    double **matrix, **rl_matrix;
    struct rnn_lyapunov_info rl_info;

    in_and_c_state_size = rnn_s->rnn_p->in_state_size +
        rnn_s->rnn_p->c_state_size;
    out_and_c_state_size = rnn_s->rnn_p->out_state_size +
        rnn_s->rnn_p->c_state_size;
    MALLOC(matrix, out_and_c_state_size);
    for (int i = 0; i < out_and_c_state_size; i++) {
        MALLOC(matrix[i], in_and_c_state_size);
    }

    init_rnn_lyapunov_info(&rl_info, rnn_s, 1, 0);
    MALLOC(rl_matrix, rl_info.dimension);
    for (int i = 0; i < rl_info.dimension; i++) {
        MALLOC(rl_matrix[i], rl_info.dimension);
    }

    assert_exit_call(rnn_jacobian_for_lyapunov_spectrum, NULL,
            rl_info.dimension, 0, rl_matrix, &rl_info);
    assert_exit_call(rnn_jacobian_for_lyapunov_spectrum, rl_info.state[1],
            rl_info.dimension, 0, rl_matrix, &rl_info);
    assert_exit_call(rnn_jacobian_for_lyapunov_spectrum, rl_info.state[0],
            rl_info.dimension+1, 0, rl_matrix, &rl_info);
    assert_exit_call(rnn_jacobian_for_lyapunov_spectrum, rl_info.state[0],
            rl_info.dimension-1, 0, rl_matrix, &rl_info);
    assert_exit_call(rnn_jacobian_for_lyapunov_spectrum, rl_info.state[0],
            rl_info.dimension, -1, rl_matrix, &rl_info);
    assert_exit_call(rnn_jacobian_for_lyapunov_spectrum, rl_info.state[0],
            rl_info.dimension, rnn_s->length, rl_matrix, &rl_info);
    assert_exit_nocall(rnn_jacobian_for_lyapunov_spectrum, rl_info.state[0],
            rl_info.dimension, 0, rl_matrix, &rl_info);
    assert_exit_nocall(rnn_jacobian_for_lyapunov_spectrum,
            rl_info.state[rnn_s->length-1], rl_info.dimension, rnn_s->length-1,
            rl_matrix, &rl_info);

    rnn_jacobian_for_lyapunov_spectrum(rl_info.state[0], rl_info.dimension, 0,
            rl_matrix, &rl_info);
    rnn_jacobian_matrix(matrix, rnn_s->rnn_p, rnn_s->init_c_state,
            rnn_s->c_state[0], rnn_s->out_state[0]);
    if (rnn_s->rnn_p->in_state_size != 0) {
        for (int i = 0; i < out_and_c_state_size; i++) {
            assert_equal_memory(matrix[i], out_and_c_state_size *
                    sizeof(double), rl_matrix[i], rl_info.dimension *
                    sizeof(double));
        }
    } else {
        for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++) {
            assert_equal_memory(matrix[i + rnn_s->rnn_p->out_state_size],
                    rnn_s->rnn_p->c_state_size * sizeof(double),
                    rl_matrix[i], rl_info.dimension * sizeof(double));
        }
    }
    for (int i = 0; i < rl_info.dimension; i++) {
        free(rl_matrix[i]);
    }
    free(rl_matrix);
    free_rnn_lyapunov_info(&rl_info);


    int I, J, K, L;
    init_rnn_lyapunov_info(&rl_info, rnn_s, 3, 0);
    MALLOC(rl_matrix, rl_info.dimension);
    for (int i = 0; i < rl_info.dimension; i++) {
        MALLOC(rl_matrix[i], rl_info.dimension);
    }
    mu_assert(rnn_jacobian_for_lyapunov_spectrum(rl_info.state[0],
                rl_info.dimension, 0, rl_matrix, &rl_info) != NULL);
    rnn_jacobian_matrix(matrix, rnn_s->rnn_p, rnn_s->init_c_state,
            rnn_s->c_state[0], rnn_s->out_state[0]);
    if (rnn_s->rnn_p->in_state_size != 0) {
        for (int n = 1; n < rl_info.delay_length; n++) {
            I = rnn_s->rnn_p->out_state_size * n;
            J = rnn_s->rnn_p->out_state_size * (n-1);
            for (int i = 0; i < rnn_s->rnn_p->out_state_size; i++) {
                for (int j = 0; j < rnn_s->rnn_p->in_state_size; j++) {
                    assert_equal_double((i==j) ? 1.0 : 0.0,
                            rl_matrix[i+I][j+J], 0);
                }
            }
        }
        for (int i = 0; i < rnn_s->rnn_p->out_state_size; i++) {
            I = 0;
            J = rnn_s->rnn_p->in_state_size * (rl_info.delay_length-1);
            assert_equal_memory(matrix[i]+I,
                    rnn_s->rnn_p->in_state_size * sizeof(double),
                    rl_matrix[i]+J,
                    rnn_s->rnn_p->in_state_size * sizeof(double));
            I = rnn_s->rnn_p->in_state_size;
            J = rnn_s->rnn_p->in_state_size * rl_info.delay_length;
            assert_equal_memory(matrix[i]+I,
                    rnn_s->rnn_p->c_state_size * sizeof(double),
                    rl_matrix[i]+J,
                    rnn_s->rnn_p->c_state_size * sizeof(double));
        }
    }
    I = rnn_s->rnn_p->out_state_size;
    if (rnn_s->rnn_p->in_state_size != 0) {
        J = rnn_s->rnn_p->out_state_size * rl_info.delay_length;
    } else {
        J = 0;
    }
    for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++) {
        K = 0;
        L = rnn_s->rnn_p->in_state_size * (rl_info.delay_length-1);
        assert_equal_memory(matrix[i+I]+K,
                rnn_s->rnn_p->in_state_size * sizeof(double),
                rl_matrix[i+J]+L,
                rnn_s->rnn_p->in_state_size * sizeof(double));
        K = rnn_s->rnn_p->in_state_size;
        L = rnn_s->rnn_p->in_state_size * rl_info.delay_length;
        assert_equal_memory(matrix[i+I]+K,
                rnn_s->rnn_p->c_state_size * sizeof(double),
                rl_matrix[i+J]+L,
                rnn_s->rnn_p->c_state_size * sizeof(double));
    }
    for (int i = 0; i < rl_info.dimension; i++) {
        free(rl_matrix[i]);
    }
    free(rl_matrix);
    free_rnn_lyapunov_info(&rl_info);

    for (int i = 0; i < out_and_c_state_size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}


static void test_reset_rnn_lyapunov_info (
        const struct rnn_state *rnn_s)
{
    if (rnn_s->rnn_p->in_state_size != rnn_s->rnn_p->out_state_size &&
            rnn_s->rnn_p->in_state_size != 0) {
        return;
    }
    struct rnn_lyapunov_info rl_info;

    init_rnn_lyapunov_info(&rl_info, rnn_s, 1, 0);
    assert_reset_rnn_lyapunov_info(&rl_info);
    free_rnn_lyapunov_info(&rl_info);

    init_rnn_lyapunov_info(&rl_info, rnn_s, 3, 0);
    assert_reset_rnn_lyapunov_info(&rl_info);
    free_rnn_lyapunov_info(&rl_info);

    init_rnn_lyapunov_info(&rl_info, rnn_s, 1, 20);
    assert_reset_rnn_lyapunov_info(&rl_info);
    free_rnn_lyapunov_info(&rl_info);

    init_rnn_lyapunov_info(&rl_info, rnn_s, 4, 50);
    assert_reset_rnn_lyapunov_info(&rl_info);
    free_rnn_lyapunov_info(&rl_info);

    init_rnn_lyapunov_info(&rl_info, rnn_s, rl_info.length/2, rl_info.length/2);
    assert_reset_rnn_lyapunov_info(&rl_info);
    free_rnn_lyapunov_info(&rl_info);
}

static void test_rnn_lyapunov_spectrum (
        struct rnn_state *rnn_s)
{

    if (rnn_s->rnn_p->in_state_size != rnn_s->rnn_p->out_state_size &&
            rnn_s->rnn_p->in_state_size != 0) {
        return;
    }

    struct rnn_lyapunov_info rl_info;
    init_rnn_lyapunov_info(&rl_info, rnn_s, 1, 0);

    double spectrum[rl_info.dimension];
    for (int n = 1; n < 10; n++) {
        rnn_set_uniform_tau(rnn_s->rnn_p, n);
        rnn_forward_dynamics_in_closed_loop(rnn_s, rl_info.delay_length);
        reset_rnn_lyapunov_info(&rl_info);
        rnn_lyapunov_spectrum(&rl_info, spectrum, rl_info.dimension);
        mu_assert(spectrum[0] < 0);
        for (int i = 1; i < rl_info.dimension; i++) {
            mu_assert(spectrum[i-1] >= spectrum[i]);
        }
    }

    free_rnn_lyapunov_info(&rl_info);
}


void test_rnn_state_setup (
        struct recurrent_neural_network *rnn,
        int target_num,
        int *target_length);

static void test_rnn_lyapunov_setup (
        struct recurrent_neural_network *rnn,
        unsigned long seed,
        int in_state_size,
        int c_state_size,
        int out_state_size)
{
    init_genrand(seed);
    init_recurrent_neural_network(rnn, in_state_size, c_state_size,
            out_state_size);

    test_rnn_state_setup(rnn, 1, (int[]){100});
}


void test_rnn_lyapunov (void)
{
    struct recurrent_neural_network rnn[3];
    test_rnn_lyapunov_setup(rnn, 5893L, 5, 12, 5);
    test_rnn_lyapunov_setup(rnn+1, 5893L, 0, 12, 3);
    test_rnn_lyapunov_setup(rnn+2, 5893L, 4, 12, 3);

    for (int i = 0; i < 3; i++) {
        mu_run_test_with_args(test_init_rnn_lyapunov_info, rnn[i].rnn_s);
        mu_run_test_with_args(test_rnn_jacobian_for_lyapunov_spectrum,
                rnn[i].rnn_s);
        mu_run_test_with_args(test_reset_rnn_lyapunov_info, rnn[i].rnn_s);
        mu_run_test_with_args(test_rnn_lyapunov_spectrum, rnn[i].rnn_s);
        free_recurrent_neural_network(rnn + i);
    }
}


