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
#include <string.h>
#include <math.h>

#define TEST_CODE
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "minunit.h"
#include "my_assert.h"
#include "utils.h"
#include "rnn.h"


#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif


/* assert functions */

void assert_equal_rnn_p (
        struct rnn_parameters *rnn1_p,
        struct rnn_parameters *rnn2_p)
{
    assert_equal_int(rnn1_p->in_state_size, rnn2_p->in_state_size);
    assert_equal_int(rnn1_p->c_state_size, rnn2_p->c_state_size);
    assert_equal_int(rnn1_p->out_state_size, rnn2_p->out_state_size);
    assert_equal_int(rnn1_p->output_type, rnn2_p->output_type);
    assert_equal_int(rnn1_p->fixed_weight, rnn2_p->fixed_weight);
    assert_equal_int(rnn1_p->fixed_threshold, rnn2_p->fixed_threshold);
    assert_equal_int(rnn1_p->fixed_tau, rnn2_p->fixed_tau);
    assert_equal_int(rnn1_p->fixed_init_c_state, rnn2_p->fixed_init_c_state);
    assert_equal_int(rnn1_p->fixed_sigma, rnn2_p->fixed_sigma);
    assert_equal_int(rnn1_p->softmax_group_num, rnn2_p->softmax_group_num);
    assert_equal_double(rnn1_p->sigma, rnn2_p->sigma, 0.0);
    assert_equal_double(rnn1_p->variance, rnn2_p->variance, 0.0);

    assert_equal_memory(rnn1_p->const_init_c, rnn1_p->c_state_size *
            sizeof(int), rnn2_p->const_init_c, rnn2_p->c_state_size *
            sizeof(int));
    assert_equal_memory(rnn1_p->softmax_group_id,
            rnn1_p->out_state_size * sizeof(int), rnn2_p->softmax_group_id,
            rnn2_p->out_state_size * sizeof(int));


    size_t in_msz1, in_msz2, c_msz1, c_msz2, out_msz1, out_msz2;

    in_msz1 = rnn1_p->in_state_size * sizeof(double);
    in_msz2 = rnn2_p->in_state_size * sizeof(double);
    c_msz1 = rnn1_p->c_state_size * sizeof(double);
    c_msz2 = rnn2_p->c_state_size * sizeof(double);
    out_msz1 = rnn1_p->out_state_size * sizeof(double);
    out_msz2 = rnn2_p->out_state_size * sizeof(double);

    for (int i = 0; i < rnn1_p->c_state_size; i++) {
        assert_equal_memory(rnn1_p->weight_ci[i], in_msz1,
                rnn2_p->weight_ci[i], in_msz2);
        assert_equal_memory(rnn1_p->weight_cc[i], c_msz1,
                rnn2_p->weight_cc[i], c_msz2);
    }
    for (int i = 0; i < rnn1_p->out_state_size; i++) {
        assert_equal_memory(rnn1_p->weight_oc[i], c_msz1,
                rnn2_p->weight_oc[i], c_msz2);
    }
    assert_equal_memory(rnn1_p->threshold_c, c_msz1, rnn2_p->threshold_c,
            c_msz2);
    assert_equal_memory(rnn1_p->threshold_o, out_msz1, rnn2_p->threshold_o,
            out_msz2);
    assert_equal_memory(rnn1_p->tau, c_msz1, rnn2_p->tau, c_msz2);
    assert_equal_memory(rnn1_p->eta, c_msz1, rnn2_p->eta, c_msz2);

    assert_equal_double(rnn1_p->delta_sigma, rnn2_p->delta_sigma, 0.0);
    for (int i = 0; i < rnn1_p->c_state_size; i++) {
        assert_equal_memory(rnn1_p->delta_weight_ci[i], in_msz1,
                rnn2_p->delta_weight_ci[i], in_msz2);
        assert_equal_memory(rnn1_p->delta_weight_cc[i], c_msz1,
                rnn2_p->delta_weight_cc[i], c_msz2);
    }
    for (int i = 0; i < rnn1_p->out_state_size; i++) {
        assert_equal_memory(rnn1_p->delta_weight_oc[i], c_msz1,
                rnn2_p->delta_weight_oc[i], c_msz2);
    }
    assert_equal_memory(rnn1_p->delta_threshold_c, c_msz1,
            rnn2_p->delta_threshold_c, c_msz2);
    assert_equal_memory(rnn1_p->delta_threshold_o, out_msz1,
            rnn2_p->delta_threshold_o, out_msz2);
    assert_equal_memory(rnn1_p->delta_tau, c_msz1, rnn2_p->delta_tau,
            c_msz2);

    assert_equal_double(rnn1_p->prior_strength, rnn2_p->prior_strength, 0.0);
    assert_equal_double(rnn1_p->prior_sigma, rnn2_p->prior_sigma, 0.0);
    for (int i = 0; i < rnn1_p->c_state_size; i++) {
        assert_equal_memory(rnn1_p->prior_weight_ci[i], in_msz1,
                rnn2_p->prior_weight_ci[i], in_msz2);
        assert_equal_memory(rnn1_p->prior_weight_cc[i], c_msz1,
                rnn2_p->prior_weight_cc[i], c_msz2);
    }
    for (int i = 0; i < rnn1_p->out_state_size; i++) {
        assert_equal_memory(rnn1_p->prior_weight_oc[i], c_msz1,
                rnn2_p->prior_weight_oc[i], c_msz2);
    }
    assert_equal_memory(rnn1_p->prior_threshold_c, c_msz1,
            rnn2_p->prior_threshold_c, c_msz2);
    assert_equal_memory(rnn1_p->prior_threshold_o, out_msz1,
            rnn2_p->prior_threshold_o, out_msz2);
    assert_equal_memory(rnn1_p->prior_tau, c_msz1, rnn2_p->prior_tau,
            c_msz2);

    for (int i = 0; i < rnn1_p->c_state_size; i++) {
        for (int j = 0; j <= rnn1_p->in_state_size; j++) {
            assert_equal_int(rnn1_p->connection_ci[i][j].begin,
                    rnn2_p->connection_ci[i][j].begin);
            assert_equal_int(rnn1_p->connection_ci[i][j].end,
                    rnn2_p->connection_ci[i][j].end);
        }
        for (int j = 0; j <= rnn1_p->c_state_size; j++) {
            assert_equal_int(rnn1_p->connection_cc[i][j].begin,
                    rnn2_p->connection_cc[i][j].begin);
            assert_equal_int(rnn1_p->connection_cc[i][j].end,
                    rnn2_p->connection_cc[i][j].end);
        }
    }
    for (int i = 0; i < rnn1_p->out_state_size; i++) {
        for (int j = 0; j <= rnn1_p->c_state_size; j++) {
            assert_equal_int(rnn1_p->connection_oc[i][j].begin,
                    rnn2_p->connection_oc[i][j].begin);
            assert_equal_int(rnn1_p->connection_oc[i][j].end,
                    rnn2_p->connection_oc[i][j].end);
        }
    }
}

void assert_equal_rnn_s (
        struct rnn_state *rnn1_s,
        struct rnn_state *rnn2_s)
{
    size_t in_msz1, in_msz2, c_msz1, c_msz2, out_msz1, out_msz2;

    in_msz1 = rnn1_s->rnn_p->in_state_size * sizeof(double);
    in_msz2 = rnn2_s->rnn_p->in_state_size * sizeof(double);
    c_msz1 = rnn1_s->rnn_p->c_state_size * sizeof(double);
    c_msz2 = rnn2_s->rnn_p->c_state_size * sizeof(double);
    out_msz1 = rnn1_s->rnn_p->out_state_size * sizeof(double);
    out_msz2 = rnn2_s->rnn_p->out_state_size * sizeof(double);

    assert_equal_int(rnn1_s->length, rnn2_s->length);

    assert_equal_memory(rnn1_s->init_c_inter_state, c_msz1,
            rnn2_s->init_c_inter_state, c_msz2);
    assert_equal_memory(rnn1_s->init_c_state, c_msz1, rnn2_s->init_c_state,
            c_msz2);
    assert_equal_memory(rnn1_s->delta_init_c_inter_state, c_msz1,
            rnn2_s->delta_init_c_inter_state, c_msz2);

    assert_equal_vector_sequence(rnn1_s->in_state, in_msz1, rnn1_s->length,
            rnn2_s->in_state, in_msz2, rnn2_s->length);
    assert_equal_vector_sequence(rnn1_s->teach_state, out_msz1, rnn1_s->length,
            rnn2_s->teach_state, out_msz2, rnn2_s->length);
}

void assert_rnn_forward_context_map (
        const struct rnn_parameters *rnn_p,
        const double *in_state,
        const double *prev_c_inter_state,
        const double *prev_c_state,
        const double *c_inputsum,
        const double *c_inter_state,
        const double *c_state)
{
    double x, w, c;
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        w = rnn_p->threshold_c[i];
        for (int j = 0; j < rnn_p->in_state_size; j++) {
            w += rnn_p->weight_ci[i][j] * in_state[j];
        }
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            w += rnn_p->weight_cc[i][j] * prev_c_state[j];
        }
        assert_equal_double(w, c_inputsum[i], 1e-12);
        x = (1 - (1.0/rnn_p->tau[i])) * prev_c_inter_state[i] +
            (1.0/rnn_p->tau[i]) * c_inputsum[i];
        assert_equal_double(x, c_inter_state[i], 1e-12);
        c = tanh(c_inter_state[i]);
        assert_equal_double(c, c_state[i], 1e-12);
    }
}


void assert_forward_output_map_for_standard (
        const struct rnn_parameters *rnn_p,
        const double *c_state,
        double *o_inter_state,
        double *out_state)
{
    double w, o;
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        w = rnn_p->threshold_o[i];
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            w += rnn_p->weight_oc[i][j] * c_state[j];
        }
        o = tanh(o_inter_state[i]);
        assert_equal_double(w, o_inter_state[i], 1e-12);
        assert_equal_double(o, out_state[i], 1e-12);
    }
}

void assert_forward_output_map_for_softmax (
        const struct rnn_parameters *rnn_p,
        const double *c_state,
        double *o_inter_state,
        double *out_state)
{
    double sum[rnn_p->softmax_group_num], x[rnn_p->out_state_size], w, o;
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        x[i] = exp(o_inter_state[i]);
    }
    for (int c = 0; c < rnn_p->softmax_group_num; c++) {
        sum[c] = 0;
        for (int i = 0; i < rnn_p->out_state_size; i++) {
            if (rnn_p->softmax_group_id[i] == c) {
                sum[c] += x[i];
            }
        }
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        w = rnn_p->threshold_o[i];
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            w += rnn_p->weight_oc[i][j] * c_state[j];
        }
        o = x[i]/sum[rnn_p->softmax_group_id[i]];
        assert_equal_double(w, o_inter_state[i], 1e-12);
        assert_equal_double(o, out_state[i], 1e-12);
    }
}


#ifdef ENABLE_ATTRACTION_OF_INIT_C
static double get_posterior_distribution (
        struct recurrent_neural_network *rnn,
        double **mean,
        double **variance)
#else
static double get_posterior_distribution (struct recurrent_neural_network *rnn)
#endif
{
    double post_dist = 0;
    for (int i = 0; i < rnn->series_num; i++) {
        rnn_set_likelihood(rnn->rnn_s + i);
        post_dist += rnn_get_likelihood(rnn->rnn_s + i);
    }
    double d = rnn->rnn_p.prior_sigma - rnn->rnn_p.sigma;
    post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
    for (int i = 0; i < rnn->rnn_p.c_state_size; i++) {
        for (int j = 0; j < rnn->rnn_p.in_state_size; j++) {
            d = rnn->rnn_p.prior_weight_ci[i][j] - rnn->rnn_p.weight_ci[i][j];
            post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
        }
        for (int j = 0; j < rnn->rnn_p.c_state_size; j++) {
            d = rnn->rnn_p.prior_weight_cc[i][j] - rnn->rnn_p.weight_cc[i][j];
            post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
        }
        d = rnn->rnn_p.prior_threshold_c[i] - rnn->rnn_p.threshold_c[i];
        post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
        d = rnn->rnn_p.prior_tau[i] - rnn->rnn_p.tau[i];
        post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
    }
    for (int i = 0; i < rnn->rnn_p.out_state_size; i++) {
        for (int j = 0; j < rnn->rnn_p.c_state_size; j++) {
            d = rnn->rnn_p.prior_weight_oc[i][j] - rnn->rnn_p.weight_oc[i][j];
            post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
        }
        d = rnn->rnn_p.prior_threshold_o[i] - rnn->rnn_p.threshold_o[i];
        post_dist -= 0.5 * (d * d) * rnn->rnn_p.prior_strength;
    }
#ifdef ENABLE_ATTRACTION_OF_INIT_C
    for (int i = 0; i < rnn->series_num; i++) {
        for (int j = 0; j < rnn->rnn_p.c_state_size; j++) {
            d = mean[i][j] - rnn->rnn_s[i].init_c_inter_state[j];
            post_dist -= (d * d) / (2 * variance[i][j]);
            post_dist -= 0.5 * log(2 * M_PI * variance[i][j]);
        }
    }
#endif
    return post_dist;
}

#ifdef ENABLE_ATTRACTION_OF_INIT_C
void get_mean_and_variance (
        struct rnn_state *rnn_s,
        double *mean,
        double *variance)
{
    for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++) {
        mean[i] = variance[i] = 0;
        for (int n = 0; n < rnn_s->length; n++) {
            mean[i] += rnn_s->c_inter_state[n][i];
        }
        mean[i] /= rnn_s->length;
        for (int n = 0; n < rnn_s->length; n++) {
            double d = mean[i] - rnn_s->c_inter_state[n][i];
            variance[i] += d * d;
        }
        variance[i] /= rnn_s->length;
        if (variance[i] < MIN_VARIANCE) {
            variance[i] = MIN_VARIANCE;
        }
    }
}
#endif


void assert_effect_rnn_learn (struct recurrent_neural_network *rnn)
{
    double post_dist, next_post_dist;

    rnn_forward_dynamics_forall(rnn);
#ifdef ENABLE_ATTRACTION_OF_INIT_C
    double **mean, **variance;
    MALLOC2(mean, rnn->series_num, rnn->rnn_p.c_state_size);
    MALLOC2(variance, rnn->series_num, rnn->rnn_p.c_state_size);
    for (int i = 0; i < rnn->series_num; i++) {
        get_mean_and_variance(rnn->rnn_s + i, mean[i], variance[i]);
    }
    post_dist = get_posterior_distribution(rnn, mean, variance);
#else
    post_dist = get_posterior_distribution(rnn);
#endif

    rnn_reset_delta_parameters(&(rnn->rnn_p));
    rnn_learn(rnn, 1e-6, 1e-6, 1e-6, 1e-6, 0.0);

    rnn_forward_dynamics_forall(rnn);
#ifdef ENABLE_ATTRACTION_OF_INIT_C
    next_post_dist = get_posterior_distribution(rnn, mean, variance);
    FREE2(mean);
    FREE2(variance);
#else
    next_post_dist = get_posterior_distribution(rnn);
#endif

    mu_assert(post_dist < next_post_dist);
}


void assert_effect_rnn_learn_with_adapt_lr (
        struct recurrent_neural_network *rnn,
        double *adapt_lr)
{
    double error, next_error;

    rnn_forward_dynamics_forall(rnn);
    error = 0;
    for (int i = 0; i < rnn->series_num; i++) {
        error += rnn_get_error(rnn->rnn_s + i);
    }

    rnn_reset_delta_parameters(&(rnn->rnn_p));
    *adapt_lr = rnn_learn_with_adapt_lr(rnn, *adapt_lr, 1000.0, 1000.0, 1000.0,
            0.0, 0.0);

    rnn_forward_dynamics_forall(rnn);
    next_error = 0;
    for (int i = 0; i < rnn->series_num; i++) {
        next_error += rnn_get_error(rnn->rnn_s + i);
    }
    mu_assert((next_error / error) < (1.0+1e-9));
}


static inline double DTANH_DX (double x)
{
    double y = tanh(x);
    return 1.0 - y * y;
}

void assert_jacobian_matrix_for_standard (
        struct rnn_parameters *rnn_p,
        double **matrix,
        double *prev_c_inter_state,
        double *c_inter_state,
        double *o_inter_state)
{
    int I, J;
    double val;

    I = rnn_p->out_state_size;
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        J = rnn_p->in_state_size;
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            val = rnn_p->eta[i] * rnn_p->weight_cc[i][j] *
                DTANH_DX(prev_c_inter_state[j]);
            if (i == j) {
                val += (1 - rnn_p->eta[i]);
            }
            assert_equal_double(val, matrix[i+I][j+J], 1e-10);
        }
        J = 0;
        for (int j = 0; j < rnn_p->in_state_size; j++) {
            val = rnn_p->eta[i] * rnn_p->weight_ci[i][j];
            assert_equal_double(val, matrix[i+I][j+J], 1e-10);
        }
    }
    I = 0;
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        J = rnn_p->in_state_size;
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            val = 0;
            for (int k = 0; k < rnn_p->c_state_size; k++) {
                if (k == j) {
                    val += DTANH_DX(o_inter_state[i]) * rnn_p->weight_oc[i][k] *
                        DTANH_DX(c_inter_state[k]) * ((1 - rnn_p->eta[k]) +
                                (rnn_p->eta[k] * rnn_p->weight_cc[k][j]) *
                                DTANH_DX(prev_c_inter_state[j]) );
                } else {
                    val += DTANH_DX(o_inter_state[i]) * rnn_p->weight_oc[i][k] *
                        DTANH_DX(c_inter_state[k]) * (rnn_p->eta[k] *
                                rnn_p->weight_cc[k][j] *
                                DTANH_DX(prev_c_inter_state[j]));
                }
            }
            assert_equal_double(val, matrix[i+I][j+J], 1e-10);
        }
        J = 0;
        for (int j = 0; j < rnn_p->in_state_size; j++) {
            val = 0;
            for (int k = 0; k < rnn_p->c_state_size; k++) {
                val += DTANH_DX(o_inter_state[i]) * rnn_p->weight_oc[i][k] *
                    DTANH_DX(c_inter_state[k]) * (rnn_p->eta[k] *
                            rnn_p->weight_ci[k][j]);
            }
            assert_equal_double(val, matrix[i+I][j+J], 1e-10);
        }
    }
}

void assert_jacobian_matrix_for_softmax (
        struct rnn_parameters *rnn_p,
        double **matrix,
        double *prev_c_inter_state,
        double *c_inter_state,
        double *out_state)
{
    int I, J;
    double val;

    I = rnn_p->out_state_size;
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        J = rnn_p->in_state_size;
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            val = rnn_p->eta[i] * rnn_p->weight_cc[i][j] *
                DTANH_DX(prev_c_inter_state[j]);
            if (i == j) {
                val += (1 - rnn_p->eta[i]);
            }
            assert_equal_double(val, matrix[i+I][j+J], 1e-10);
        }
        J = 0;
        for (int j = 0; j < rnn_p->in_state_size; j++) {
            val = rnn_p->eta[i] * rnn_p->weight_ci[i][j];
            assert_equal_double(val, matrix[i+I][j+J], 1e-10);
        }
    }
    I = 0;
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        J = rnn_p->in_state_size;
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            val = 0;
            for (int k = 0; k < rnn_p->c_state_size; k++) {
                for (int l = 0; l < rnn_p->out_state_size; l++) {
                    if (k == j && i == l) {
                        val += (out_state[l] - out_state[l] * out_state[i]) *
                            rnn_p->weight_oc[l][k] *
                            DTANH_DX(c_inter_state[k]) * ((1 - rnn_p->eta[k]) +
                                    (rnn_p->eta[k] * rnn_p->weight_cc[k][j] *
                                     DTANH_DX(prev_c_inter_state[j])));
                    } else if (k == j && rnn_p->softmax_group_id[i] ==
                            rnn_p->softmax_group_id[l]) {
                        val += (-out_state[l] * out_state[i]) *
                            rnn_p->weight_oc[l][k] *
                            DTANH_DX(c_inter_state[k]) * ((1 - rnn_p->eta[k]) +
                                    (rnn_p->eta[k] * rnn_p->weight_cc[k][j] *
                                     DTANH_DX(prev_c_inter_state[j])));
                    } else if (k != j && i == l) {
                        val += (out_state[l] - out_state[l] * out_state[i]) *
                            rnn_p->weight_oc[l][k] *
                            DTANH_DX(c_inter_state[k]) * (rnn_p->eta[k] *
                                    rnn_p->weight_cc[k][j] *
                                    DTANH_DX(prev_c_inter_state[j]));
                    } else if (k != j && rnn_p->softmax_group_id[i] ==
                            rnn_p->softmax_group_id[l]) {
                        val += (-out_state[l] * out_state[i]) *
                            rnn_p->weight_oc[l][k] *
                            DTANH_DX(c_inter_state[k]) * (rnn_p->eta[k] *
                                    rnn_p->weight_cc[k][j] *
                                    DTANH_DX(prev_c_inter_state[j]));
                    }
                }
            }
            assert_equal_double(val, matrix[i+I][j+J], 1e-10);
        }
        J = 0;
        for (int j = 0; j < rnn_p->in_state_size; j++) {
            val = 0;
            for (int k = 0; k < rnn_p->c_state_size; k++) {
                for (int l = 0; l < rnn_p->out_state_size; l++) {
                    if (i == l) {
                        val += (out_state[l] - out_state[l] * out_state[i]) *
                            rnn_p->weight_oc[l][k] *
                            DTANH_DX(c_inter_state[k]) *
                            (rnn_p->eta[k] * rnn_p->weight_ci[k][j]);
                    } else if (rnn_p->softmax_group_id[i] ==
                            rnn_p->softmax_group_id[l]){
                        val += (-out_state[l] * out_state[i]) *
                            rnn_p->weight_oc[l][k] *
                            DTANH_DX(c_inter_state[k]) *
                            (rnn_p->eta[k] * rnn_p->weight_ci[k][j]);
                    }
                }
            }
            assert_equal_double(val, matrix[i+I][j+J], 1e-10);
        }
    }
}


/* test functions */

static void test_init_rnn_parameters (void)
{
    struct rnn_parameters rnn_p;
    /*
     * RNN has to contain at least one context neuron and one output neuron.
     * An input neuron is not necessarily required.
     */
    assert_exit(init_rnn_parameters, &rnn_p, -1, 1, 1);
    assert_exit(init_rnn_parameters, &rnn_p, 0, 0, 1);
    assert_exit(init_rnn_parameters, &rnn_p, 0, 1, 0);
    assert_noexit(init_rnn_parameters, &rnn_p, 0, 1, 1);
    free_rnn_parameters(&rnn_p);
}

static void test_init_rnn_state (void)
{
    struct rnn_parameters rnn_p;
    struct rnn_state rnn_s;
    double **input, **target;
    int length, dim;

    init_rnn_parameters(&rnn_p, 1, 1, 1);
    length = 1;
    dim = 1;
    MALLOC2(input, length, dim);
    MALLOC2(target, length, dim);
    for (int n = 0; n < length; n++) {
        memset(input[n], 0, dim * sizeof(double));
        memset(target[n], 0, dim * sizeof(double));
    }

    assert_exit(init_rnn_state, &rnn_s, &rnn_p, 0, (const double* const*)input,
            (const double* const*)target);
    assert_noexit(init_rnn_state, &rnn_s, &rnn_p, 1,
            (const double* const*)input, (const double* const*)target);

    free_rnn_state(&rnn_s);
    free_rnn_parameters(&rnn_p);
    FREE2(input);
    FREE2(target);
}

static void test_rnn_get_connection (void)
{
    int size;
    struct connection_domain *connection;
    int *has_connection;

    size = 10;

    MALLOC(connection, size + 1);
    MALLOC(has_connection, size);
    for (int i = 0; i <= size; i++) {
        connection[i].begin = -1;
        connection[i].end = -1;
    }
    rnn_add_connection(size, connection, 1, 4);
    rnn_add_connection(size, connection, 7, 9);
    assert_equal_int(1, connection[0].begin);
    assert_equal_int(4, connection[0].end);
    assert_equal_int(7, connection[1].begin);
    assert_equal_int(9, connection[1].end);
    assert_equal_int(-1, connection[2].begin);
    rnn_get_connection(size, connection, has_connection);
    for (int i = 0; i < size; i++) {
        switch (i) {
            case 1: case 2: case 3: case 7: case 8:
                assert_equal_int(1, has_connection[i]);
                break;
            default:
                assert_equal_int(0, has_connection[i]);
                break;
        }
    }
    rnn_delete_connection(size, connection, 0, size);
    assert_equal_int(-1, connection[0].begin);
    rnn_set_connection(size, connection, has_connection);
    assert_equal_int(1, connection[0].begin);
    assert_equal_int(4, connection[0].end);
    assert_equal_int(7, connection[1].begin);
    assert_equal_int(9, connection[1].end);
    assert_equal_int(-1, connection[2].begin);
    FREE(connection);
    FREE(has_connection);
}



static void test_fwrite_recurrent_neural_network (
        struct recurrent_neural_network *rnn)
{
    struct recurrent_neural_network rnn2;
    FILE *fp;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_recurrent_neural_network(rnn, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_recurrent_neural_network(&rnn2, fp);
    fclose(fp);

    assert_equal_rnn_p(&(rnn->rnn_p), &(rnn2.rnn_p));

    assert_equal_int(rnn->series_num, rnn2.series_num);
    for (int i = 0; i < rnn->series_num; i++) {
        assert_equal_rnn_s(rnn->rnn_s + i, rnn2.rnn_s + i);
    }

    free_recurrent_neural_network(&rnn2);
}


static void test_rnn_set_uniform_tau (struct recurrent_neural_network *rnn)
{
    rnn_set_uniform_tau(&(rnn->rnn_p), 10.0);
    for (int i = 0; i < rnn->rnn_p.c_state_size; i++) {
        assert_equal_double(10.0, rnn->rnn_p.tau[i], 1e-14);
        assert_equal_double(0.1, rnn->rnn_p.eta[i], 1e-14);
    }
}


static void test_rnn_set_tau (struct recurrent_neural_network *rnn)
{
    double tau[rnn->rnn_p.c_state_size];
    for (int i = 0; i < rnn->rnn_p.c_state_size; i++) {
        tau[i] = i + 1;
    }
    rnn_set_tau(&(rnn->rnn_p), tau);
    for (int i = 0; i < rnn->rnn_p.c_state_size; i++) {
        assert_equal_double(tau[i], rnn->rnn_p.tau[i], 1e-14);
        assert_equal_double(1.0 / tau[i], rnn->rnn_p.eta[i], 1e-14);
    }
}


static void test_rnn_set_sigma (struct recurrent_neural_network *rnn)
{
    rnn_set_sigma(&(rnn->rnn_p), 1.0);
    assert_equal_double(1.0, rnn->rnn_p.sigma, 1e-14);
    double variance = pow(exp(1.0), 2) + MIN_VARIANCE;
    assert_equal_double(variance, rnn->rnn_p.variance, 1e-14);
}


static void test_rnn_get_total_length (
        struct recurrent_neural_network *rnn,
        int total_length)
{
    assert_equal_int(total_length, rnn_get_total_length(rnn));
}


static void test_rnn_get_error (struct recurrent_neural_network *rnn)
{
    rnn->rnn_p.output_type = STANDARD_TYPE;
    rnn_forward_dynamics_forall(rnn);
    for (int i = 0; i < rnn->series_num; i++) {
        double error = 0;
        for (int n = 0; n < rnn->rnn_s[i].length; n++) {
            for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
                double d = rnn->rnn_s[i].out_state[n][j] -
                    rnn->rnn_s[i].teach_state[n][j];
                error += 0.5 * d * d;
            }
        }
        assert_equal_double(rnn_get_error(rnn->rnn_s + i), error, 10e-10);
    }
    rnn->rnn_p.output_type = SOFTMAX_TYPE;
    rnn_forward_dynamics_forall(rnn);
    for (int i = 0; i < rnn->series_num; i++) {
        double error = 0;
        for (int n = 0; n < rnn->rnn_s[i].length; n++) {
            for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
                double p = rnn->rnn_s[i].teach_state[n][j];
                double q = rnn->rnn_s[i].out_state[n][j];
                error += p * log(p / q);
            }
        }
        assert_equal_double(rnn_get_error(rnn->rnn_s + i), error, 10e-10);
    }
}

static void test_rnn_get_total_error (struct recurrent_neural_network *rnn)
{
    double total_error;
    rnn->rnn_p.output_type = STANDARD_TYPE;
    rnn_forward_dynamics_forall(rnn);
    total_error = 0;
    for (int i = 0; i < rnn->series_num; i++) {
        for (int n = 0; n < rnn->rnn_s[i].length; n++) {
            for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
                double d = rnn->rnn_s[i].out_state[n][j] -
                    rnn->rnn_s[i].teach_state[n][j];
                total_error += 0.5 * d * d;
            }
        }
    }
    assert_equal_double(rnn_get_total_error(rnn), total_error, 10e-10);
    rnn->rnn_p.output_type = SOFTMAX_TYPE;
    rnn_forward_dynamics_forall(rnn);
    total_error = 0;
    for (int i = 0; i < rnn->series_num; i++) {
        for (int n = 0; n < rnn->rnn_s[i].length; n++) {
            for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
                double p = rnn->rnn_s[i].teach_state[n][j];
                double q = rnn->rnn_s[i].out_state[n][j];
                total_error += p * log(p / q);
            }
        }
    }
    assert_equal_double(rnn_get_total_error(rnn), total_error, 10e-10);
}

static void test_rnn_get_likelihood (struct recurrent_neural_network *rnn)
{
    rnn->rnn_p.output_type = STANDARD_TYPE;
    rnn_forward_dynamics_forall(rnn);
    for (int i = 0; i < rnn->series_num; i++) {
        double likelihood = 0;
        for (int n = 0; n < rnn->rnn_s[i].length; n++) {
            for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
                double d = rnn->rnn_s[i].out_state[n][j] -
                    rnn->rnn_s[i].teach_state[n][j];
                likelihood -= (d * d) / (2 * rnn->rnn_p.variance);
                likelihood -= 0.5 * log(2 * M_PI * rnn->rnn_p.variance);
            }
        }
        rnn_set_likelihood(rnn->rnn_s + i);
        assert_equal_double(rnn_get_likelihood(rnn->rnn_s + i), likelihood,
                10e-10);
    }
    rnn->rnn_p.output_type = SOFTMAX_TYPE;
    rnn_forward_dynamics_forall(rnn);
    for (int i = 0; i < rnn->series_num; i++) {
        double likelihood = 0;
        for (int n = 0; n < rnn->rnn_s[i].length; n++) {
            for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
                double p = rnn->rnn_s[i].teach_state[n][j];
                double q = rnn->rnn_s[i].out_state[n][j];
                likelihood += p * log(q);
            }
        }
        rnn_set_likelihood(rnn->rnn_s + i);
        assert_equal_double(rnn_get_likelihood(rnn->rnn_s + i), likelihood,
                10e-10);
    }
}

static void test_rnn_get_total_likelihood (struct recurrent_neural_network *rnn)
{
    double total_likelihood;
    rnn->rnn_p.output_type = STANDARD_TYPE;
    rnn_forward_backward_dynamics_forall(rnn);
    total_likelihood = 0;
    for (int i = 0; i < rnn->series_num; i++) {
        for (int n = 0; n < rnn->rnn_s[i].length; n++) {
            for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
                double d = rnn->rnn_s[i].out_state[n][j] -
                    rnn->rnn_s[i].teach_state[n][j];
                total_likelihood -= (d * d) / (2 * rnn->rnn_p.variance);
                total_likelihood -= 0.5 * log(2 * M_PI * rnn->rnn_p.variance);
            }
        }
    }
    assert_equal_double(rnn_get_total_likelihood(rnn), total_likelihood,
            10e-10);
    rnn->rnn_p.output_type = SOFTMAX_TYPE;
    rnn_forward_backward_dynamics_forall(rnn);
    total_likelihood = 0;
    for (int i = 0; i < rnn->series_num; i++) {
        for (int n = 0; n < rnn->rnn_s[i].length; n++) {
            for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
                double p = rnn->rnn_s[i].teach_state[n][j];
                double q = rnn->rnn_s[i].out_state[n][j];
                total_likelihood += p * log(q);
            }
        }
    }
    assert_equal_double(rnn_get_total_likelihood(rnn), total_likelihood,
            10e-10);
}


static void test_rnn_clean_target (struct recurrent_neural_network *rnn)
{
    rnn_clean_target(rnn);
    assert_equal_int(0, rnn->series_num);
    assert_equal_pointer(NULL, rnn->rnn_s);
}


static void test_rnn_forward_context_map (struct rnn_parameters *rnn_p)
{
    double in_state[rnn_p->in_state_size];
    double prev_c_inter_state[rnn_p->c_state_size];
    double prev_c_state[rnn_p->c_state_size];
    double c_inputsum[rnn_p->c_state_size];
    double c_inter_state[rnn_p->c_state_size];
    double c_state[rnn_p->c_state_size];

    for (int i = 0; i < rnn_p->in_state_size; i++) {
        in_state[i] = 0;
    }
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        prev_c_inter_state[i] = 0;
        prev_c_state[i] = 0;
    }

    rnn_forward_context_map(rnn_p, in_state, prev_c_inter_state, prev_c_state,
            c_inputsum, c_inter_state, c_state);
    assert_rnn_forward_context_map(rnn_p, in_state, prev_c_inter_state,
            prev_c_state, c_inputsum, c_inter_state, c_state);


    for (int i = 0; i < rnn_p->in_state_size; i++) {
        in_state[i] = 1;
    }
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        prev_c_inter_state[i] = 1;
        prev_c_state[i] = 1;
    }
    rnn_forward_context_map(rnn_p, in_state, prev_c_inter_state, prev_c_state,
            c_inputsum, c_inter_state, c_state);
    assert_rnn_forward_context_map(rnn_p, in_state, prev_c_inter_state,
            prev_c_state, c_inputsum, c_inter_state, c_state);
}

static void test_rnn_forward_output_map (struct rnn_parameters *rnn_p)
{
    double c_state[rnn_p->c_state_size];
    double o_inter_state[rnn_p->out_state_size];
    double out_state[rnn_p->out_state_size];

    for (int i = 0; i < rnn_p->c_state_size; i++) {
        c_state[i] = 0;
    }

    rnn_p->output_type = STANDARD_TYPE;
    rnn_forward_output_map(rnn_p, c_state, o_inter_state, out_state);
    assert_forward_output_map_for_standard(rnn_p, c_state, o_inter_state,
            out_state);

    rnn_p->output_type = SOFTMAX_TYPE;
    rnn_forward_output_map(rnn_p, c_state, o_inter_state, out_state);
    assert_forward_output_map_for_softmax(rnn_p, c_state, o_inter_state,
            out_state);

    for (int i = 0; i < rnn_p->c_state_size; i++) {
        c_state[i] = 1;
    }

    rnn_p->output_type = STANDARD_TYPE;
    rnn_forward_output_map(rnn_p, c_state, o_inter_state, out_state);
    assert_forward_output_map_for_standard(rnn_p, c_state, o_inter_state,
            out_state);

    rnn_p->output_type = SOFTMAX_TYPE;
    rnn_forward_output_map(rnn_p, c_state, o_inter_state, out_state);
    assert_forward_output_map_for_softmax(rnn_p, c_state, o_inter_state,
            out_state);
}


static void test_rnn_forward_dynamics_forall (
        struct recurrent_neural_network *rnn)
{
    struct recurrent_neural_network rnn2;
    FILE *fp;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_recurrent_neural_network(rnn, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_recurrent_neural_network(&rnn2, fp);
    fclose(fp);

    size_t c_msz1, c_msz2, out_msz1, out_msz2;
    c_msz1 = rnn->rnn_p.c_state_size * sizeof(double);
    c_msz2 = rnn2.rnn_p.c_state_size * sizeof(double);
    out_msz1 = rnn->rnn_p.out_state_size * sizeof(double);
    out_msz2 = rnn2.rnn_p.out_state_size * sizeof(double);

    rnn->rnn_p.output_type = rnn2.rnn_p.output_type = STANDARD_TYPE;
    rnn_forward_dynamics_forall(rnn);
    struct rnn_state *rnn_s, *rnn2_s;
    for (int i = 0; i < rnn->series_num; i++) {
        rnn_s = rnn->rnn_s + i;
        rnn2_s = rnn2.rnn_s + i;
        rnn_forward_dynamics(rnn2_s);
        assert_equal_vector_sequence(rnn2_s->c_state, c_msz2, rnn2_s->length,
                rnn_s->c_state, c_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->out_state, out_msz2,
                rnn2_s->length, rnn_s->out_state, out_msz1, rnn_s->length);
    }

    rnn->rnn_p.output_type = rnn2.rnn_p.output_type = SOFTMAX_TYPE;
    rnn_forward_dynamics_forall(rnn);
    for (int i = 0; i < rnn->series_num; i++) {
        rnn_s = rnn->rnn_s + i;
        rnn2_s = rnn2.rnn_s + i;
        rnn_forward_dynamics(rnn2_s);
        assert_equal_vector_sequence(rnn2_s->c_state, c_msz2, rnn2_s->length,
                rnn_s->c_state, c_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->out_state, out_msz2,
                rnn2_s->length, rnn_s->out_state, out_msz1, rnn_s->length);
    }

    free_recurrent_neural_network(&rnn2);
}

static void test_rnn_forward_backward_dynamics_forall (
        struct recurrent_neural_network *rnn)
{
    struct recurrent_neural_network rnn2;
    FILE *fp;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_recurrent_neural_network(rnn, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_recurrent_neural_network(&rnn2, fp);
    fclose(fp);

    size_t in_msz1, in_msz2, c_msz1, c_msz2, out_msz1, out_msz2;
    in_msz1 = rnn->rnn_p.in_state_size * sizeof(double);
    in_msz2 = rnn2.rnn_p.in_state_size * sizeof(double);
    c_msz1 = rnn->rnn_p.c_state_size * sizeof(double);
    c_msz2 = rnn2.rnn_p.c_state_size * sizeof(double);
    out_msz1 = rnn->rnn_p.out_state_size * sizeof(double);
    out_msz2 = rnn2.rnn_p.out_state_size * sizeof(double);


    for (int i = 0; i < rnn->series_num; i++) {
        for (int j = 0; j < rnn->rnn_p.c_state_size; j++) {
            memset(rnn->rnn_s[i].delta_w_ci[j], 0, in_msz1);
            memset(rnn->rnn_s[i].delta_w_cc[j], 0, c_msz1);
        }
        for (int j = 0; j < rnn->rnn_p.out_state_size; j++) {
            memset(rnn->rnn_s[i].delta_w_oc[j], 0, c_msz1);
        }
        for (int j = 0; j < rnn2.rnn_p.c_state_size; j++) {
            memset(rnn2.rnn_s[i].delta_w_ci[j], 0, in_msz2);
            memset(rnn2.rnn_s[i].delta_w_cc[j], 0, c_msz2);
        }
        for (int j = 0; j < rnn2.rnn_p.out_state_size; j++) {
            memset(rnn2.rnn_s[i].delta_w_oc[j], 0, c_msz2);
        }
        memset(rnn->rnn_s[i].delta_t_c, 0, c_msz1);
        memset(rnn2.rnn_s[i].delta_t_c, 0, c_msz2);
        memset(rnn->rnn_s[i].delta_t_o, 0, out_msz1);
        memset(rnn2.rnn_s[i].delta_t_o, 0, out_msz2);
        memset(rnn->rnn_s[i].delta_tau, 0, c_msz1);
        memset(rnn2.rnn_s[i].delta_tau, 0, c_msz2);
    }

    rnn->rnn_p.output_type = rnn2.rnn_p.output_type = STANDARD_TYPE;
    rnn_forward_backward_dynamics_forall(rnn);
    struct rnn_state *rnn_s, *rnn2_s;
    for (int i = 0; i < rnn->series_num; i++) {
        rnn_s = rnn->rnn_s + i;
        rnn2_s = rnn2.rnn_s + i;
        rnn_forward_backward_dynamics(rnn2_s);
        assert_equal_vector_sequence(rnn2_s->c_state, c_msz2, rnn2_s->length,
                rnn_s->c_state, c_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->out_state, out_msz2,
                rnn2_s->length, rnn_s->out_state, out_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->likelihood, out_msz2,
                rnn2_s->length, rnn_s->likelihood, out_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->delta_likelihood, out_msz2,
                rnn2_s->length, rnn_s->delta_likelihood, out_msz1,
                rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->delta_c_inter, c_msz2,
                rnn2_s->length, rnn_s->delta_c_inter, c_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->delta_o_inter, out_msz2,
                rnn2_s->length, rnn_s->delta_o_inter, out_msz1, rnn_s->length);
        assert_equal_double(rnn2_s->delta_s, rnn_s->delta_s, 0);
        assert_equal_vector_sequence(rnn2_s->delta_w_ci, in_msz2,
                rnn2_s->rnn_p->c_state_size, rnn_s->delta_w_ci, in_msz1,
                rnn_s->rnn_p->c_state_size);
        assert_equal_vector_sequence(rnn2_s->delta_w_cc, c_msz2,
                rnn2_s->rnn_p->c_state_size, rnn_s->delta_w_cc, c_msz1,
                rnn_s->rnn_p->c_state_size);
        assert_equal_vector_sequence(rnn2_s->delta_w_oc, c_msz2,
                rnn2_s->rnn_p->out_state_size, rnn_s->delta_w_oc, c_msz1,
                rnn_s->rnn_p->out_state_size);
        assert_equal_memory(rnn2_s->delta_t_c, c_msz2, rnn_s->delta_t_c,
                c_msz1);
        assert_equal_memory(rnn2_s->delta_t_o, out_msz2, rnn_s->delta_t_o,
                out_msz1);
        assert_equal_memory(rnn2_s->delta_tau, c_msz2, rnn_s->delta_tau,
                c_msz1);
    }

    rnn->rnn_p.output_type = rnn2.rnn_p.output_type = SOFTMAX_TYPE;
    rnn_forward_backward_dynamics_forall(rnn);
    for (int i = 0; i < rnn->series_num; i++) {
        rnn_s = rnn->rnn_s + i;
        rnn2_s = rnn2.rnn_s + i;
        rnn_forward_backward_dynamics(rnn2_s);
        assert_equal_vector_sequence(rnn2_s->c_state, c_msz2, rnn2_s->length,
                rnn_s->c_state, c_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->out_state, out_msz2,
                rnn2_s->length, rnn_s->out_state, out_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->likelihood, out_msz2,
                rnn2_s->length, rnn_s->likelihood, out_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->delta_likelihood, out_msz2,
                rnn2_s->length, rnn_s->delta_likelihood, out_msz1,
                rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->delta_c_inter, c_msz2,
                rnn2_s->length, rnn_s->delta_c_inter, c_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->delta_o_inter, out_msz2,
                rnn2_s->length, rnn_s->delta_o_inter, out_msz1, rnn_s->length);
        assert_equal_double(rnn2_s->delta_s, rnn_s->delta_s, 0);
        assert_equal_vector_sequence(rnn2_s->delta_w_ci, in_msz2,
                rnn2_s->rnn_p->c_state_size, rnn_s->delta_w_ci, in_msz1,
                rnn_s->rnn_p->c_state_size);
        assert_equal_vector_sequence(rnn2_s->delta_w_cc, c_msz2,
                rnn2_s->rnn_p->c_state_size, rnn_s->delta_w_cc, c_msz1,
                rnn_s->rnn_p->c_state_size);
        assert_equal_vector_sequence(rnn2_s->delta_w_oc, c_msz2,
                rnn2_s->rnn_p->out_state_size, rnn_s->delta_w_oc, c_msz1,
                rnn_s->rnn_p->out_state_size);
        assert_equal_memory(rnn2_s->delta_t_c, c_msz2, rnn_s->delta_t_c,
                c_msz1);
        assert_equal_memory(rnn2_s->delta_t_o, out_msz2, rnn_s->delta_t_o,
                out_msz1);
        assert_equal_memory(rnn2_s->delta_tau, c_msz2, rnn_s->delta_tau,
                c_msz1);
    }

    free_recurrent_neural_network(&rnn2);
}

static void test_rnn_forward_dynamics_in_closed_loop_forall (
        struct recurrent_neural_network *rnn)
{
    struct recurrent_neural_network rnn2;
    FILE *fp;

    if (rnn->rnn_p.in_state_size != rnn->rnn_p.out_state_size &&
            rnn->rnn_p.in_state_size != 0) {
        return;
    }

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_recurrent_neural_network(rnn, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_recurrent_neural_network(&rnn2, fp);
    fclose(fp);

    size_t c_msz1, c_msz2, out_msz1, out_msz2;
    c_msz1 = rnn->rnn_p.c_state_size * sizeof(double);
    c_msz2 = rnn2.rnn_p.c_state_size * sizeof(double);
    out_msz1 = rnn->rnn_p.out_state_size * sizeof(double);
    out_msz2 = rnn2.rnn_p.out_state_size * sizeof(double);

    rnn->rnn_p.output_type = rnn2.rnn_p.output_type = STANDARD_TYPE;
    rnn_forward_dynamics_in_closed_loop_forall(rnn, 1);
    struct rnn_state *rnn_s, *rnn2_s;
    for (int i = 0; i < rnn->series_num; i++) {
        rnn_s = rnn->rnn_s + i;
        rnn2_s = rnn2.rnn_s + i;
        rnn_forward_dynamics_in_closed_loop(rnn2_s, 1);
        assert_equal_vector_sequence(rnn2_s->c_state, c_msz2, rnn2_s->length,
                rnn_s->c_state, c_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->out_state, out_msz2,
                rnn2_s->length, rnn_s->out_state, out_msz1, rnn_s->length);
    }

    rnn->rnn_p.output_type = rnn2.rnn_p.output_type = SOFTMAX_TYPE;
    rnn_forward_dynamics_in_closed_loop_forall(rnn, 1);
    for (int i = 0; i < rnn->series_num; i++) {
        rnn_s = rnn->rnn_s + i;
        rnn2_s = rnn2.rnn_s + i;
        rnn_forward_dynamics_in_closed_loop(rnn2_s, 1);
        assert_equal_vector_sequence(rnn2_s->c_state, c_msz2, rnn2_s->length,
                rnn_s->c_state, c_msz1, rnn_s->length);
        assert_equal_vector_sequence(rnn2_s->out_state, out_msz2,
                rnn2_s->length, rnn_s->out_state, out_msz1, rnn_s->length);
    }

    free_recurrent_neural_network(&rnn2);
}


static void test_rnn_learn (struct recurrent_neural_network *rnn)
{
    for (int n = 0; n < 3; n++) {
        rnn->rnn_p.prior_strength = 0.01 * n;
        for (int i = 0; i < 5; i++) {
            rnn->rnn_p.fixed_weight = (i==0)?0:1;
            rnn->rnn_p.fixed_threshold = (i==1)?0:1;
            rnn->rnn_p.fixed_tau = (i==2)?0:1;
            rnn->rnn_p.fixed_init_c_state = (i==3)?0:1;
            rnn->rnn_p.fixed_sigma = (i==4)?0:1;

            rnn->rnn_p.output_type = STANDARD_TYPE;
            assert_effect_rnn_learn(rnn);
            if (rnn->rnn_p.fixed_sigma) {
                rnn->rnn_p.output_type = SOFTMAX_TYPE;
                assert_effect_rnn_learn(rnn);
            }
        }

        rnn->rnn_p.fixed_weight = 0;
        rnn->rnn_p.fixed_threshold = 0;
        rnn->rnn_p.fixed_tau = 0;
        rnn->rnn_p.fixed_init_c_state = 0;
        rnn->rnn_p.fixed_sigma = 0;

        rnn->rnn_p.output_type = STANDARD_TYPE;
        assert_effect_rnn_learn(rnn);
        rnn->rnn_p.output_type = SOFTMAX_TYPE;
        assert_effect_rnn_learn(rnn);
    }
}

static void test_rnn_learn_s (struct recurrent_neural_network *rnn)
{
    struct recurrent_neural_network rnn2;
    FILE *fp;

    rnn->rnn_p.fixed_weight = 0;
    rnn->rnn_p.fixed_threshold = 0;
    rnn->rnn_p.fixed_tau = 0;
    rnn->rnn_p.fixed_init_c_state = 0;
    rnn->rnn_p.fixed_sigma = 0;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_recurrent_neural_network(rnn, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_recurrent_neural_network(&rnn2, fp);
    fclose(fp);

    int total_length = rnn_get_total_length(rnn);
    double rho = 1e-8;
    double rho_weight = rho / (total_length * rnn->rnn_p.out_state_size);
    double rho_tau = rho / (total_length * rnn->rnn_p.out_state_size);
    double rho_sigma = rho / (total_length * rnn->rnn_p.out_state_size);
    double rho_init = rho / rnn->rnn_p.out_state_size;
    double adapt_lr = 1.0;

    for (int n = 0; n < 4; n++) {
        if (n % 2 == 0) {
            rnn->rnn_p.output_type = STANDARD_TYPE;
            rnn2.rnn_p.output_type = STANDARD_TYPE;
        } else {
            rnn->rnn_p.output_type = SOFTMAX_TYPE;
            rnn2.rnn_p.output_type = SOFTMAX_TYPE;
        }
        if (n < 2) {
            rnn_learn(rnn, rho_weight, rho_tau, rho_init, rho_sigma, 0);
            rnn_learn_s(&rnn2, rho, 0);
        } else {
            rnn_learn_with_adapt_lr(rnn, adapt_lr, rho_weight, rho_tau,
                    rho_init, rho_sigma, 0);
            rnn_learn_s_with_adapt_lr(&rnn2, adapt_lr, rho, 0);
        }
        assert_equal_rnn_p(&rnn->rnn_p, &rnn2.rnn_p);
        for (int i = 0; i < rnn->series_num; i++) {
            assert_equal_rnn_s(rnn->rnn_s + i, rnn2.rnn_s + i);
        }
    }

    free_recurrent_neural_network(&rnn2);
}

static void test_rnn_backup_learning_parameters (
        struct recurrent_neural_network *rnn)
{
    struct recurrent_neural_network tmp_rnn;
    FILE *fp;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fwrite_recurrent_neural_network(rnn, fp);
    fseek(fp, 0L, SEEK_SET);
    fread_recurrent_neural_network(&tmp_rnn, fp);
    fclose(fp);

    size_t in_msz, c_msz, out_msz;
    in_msz = rnn->rnn_p.in_state_size * sizeof(double);
    c_msz = rnn->rnn_p.c_state_size * sizeof(double);
    out_msz = rnn->rnn_p.out_state_size * sizeof(double);

    rnn_backup_learning_parameters(rnn);

    rnn->rnn_p.sigma = 0;
    rnn->rnn_p.variance = 1;
    for (int i = 0; i < rnn->rnn_p.c_state_size; i++) {
        memset(rnn->rnn_p.weight_ci[i], 0, in_msz);
        memset(rnn->rnn_p.weight_cc[i], 0, c_msz);
    }
    for (int i = 0; i < rnn->rnn_p.out_state_size; i++) {
        memset(rnn->rnn_p.weight_oc[i], 0, c_msz);
    }
    memset(rnn->rnn_p.threshold_c, 0, c_msz);
    memset(rnn->rnn_p.threshold_o, 0, out_msz);
    memset(rnn->rnn_p.tau, 0, c_msz);
    memset(rnn->rnn_p.eta, 0, c_msz);
    for (int i = 0; i < rnn->series_num; i++) {
        memset(rnn->rnn_s[i].init_c_inter_state, 0, c_msz);
        memset(rnn->rnn_s[i].init_c_state, 0, c_msz);
    }

    rnn_restore_learning_parameters(rnn);

    assert_equal_double(tmp_rnn.rnn_p.sigma, rnn->rnn_p.sigma, 0.0);
    assert_equal_double(tmp_rnn.rnn_p.variance, rnn->rnn_p.variance, 0.0);
    for (int i = 0; i < rnn->rnn_p.c_state_size; i++) {
        assert_equal_memory(tmp_rnn.rnn_p.weight_ci[i], in_msz,
                rnn->rnn_p.weight_ci[i], in_msz);
        assert_equal_memory(tmp_rnn.rnn_p.weight_cc[i], c_msz,
                rnn->rnn_p.weight_cc[i], c_msz);
    }
    for (int i = 0; i < rnn->rnn_p.out_state_size; i++) {
        assert_equal_memory(tmp_rnn.rnn_p.weight_oc[i], c_msz,
                rnn->rnn_p.weight_oc[i], c_msz);
    }
    assert_equal_memory(tmp_rnn.rnn_p.threshold_c, c_msz,
            rnn->rnn_p.threshold_c, c_msz);
    assert_equal_memory(tmp_rnn.rnn_p.threshold_o, out_msz,
            rnn->rnn_p.threshold_o, out_msz);
    assert_equal_memory(tmp_rnn.rnn_p.tau, c_msz, rnn->rnn_p.tau, c_msz);
    assert_equal_memory(tmp_rnn.rnn_p.eta, c_msz, rnn->rnn_p.eta, c_msz);
    for (int i = 0; i < rnn->series_num; i++) {
        assert_equal_memory(tmp_rnn.rnn_s[i].init_c_inter_state, c_msz,
                rnn->rnn_s[i].init_c_inter_state, c_msz);
        assert_equal_memory(tmp_rnn.rnn_s[i].init_c_state, c_msz,
                rnn->rnn_s[i].init_c_state, c_msz);
    }

    free_recurrent_neural_network(&tmp_rnn);
}

static void test_rnn_learn_with_adapt_lr (
        struct recurrent_neural_network *rnn)
{
    double adapt_lr = 1.0;
    rnn->rnn_p.prior_strength = 0;
    for (int i = 0; i < 4; i++) {
        rnn->rnn_p.fixed_weight = (i==0)?0:1;
        rnn->rnn_p.fixed_threshold = (i==1)?0:1;
        rnn->rnn_p.fixed_tau = (i==2)?0:1;
        rnn->rnn_p.fixed_init_c_state = (i==3)?0:1;

        rnn->rnn_p.output_type = STANDARD_TYPE;
        assert_effect_rnn_learn_with_adapt_lr(rnn, &adapt_lr);
        rnn->rnn_p.output_type = SOFTMAX_TYPE;
        assert_effect_rnn_learn_with_adapt_lr(rnn, &adapt_lr);
    }

    rnn->rnn_p.fixed_weight = 0;
    rnn->rnn_p.fixed_threshold = 0;
    rnn->rnn_p.fixed_tau = 0;
    rnn->rnn_p.fixed_init_c_state = 0;

    rnn->rnn_p.output_type = STANDARD_TYPE;
    assert_effect_rnn_learn_with_adapt_lr(rnn, &adapt_lr);
    rnn->rnn_p.output_type = SOFTMAX_TYPE;
    assert_effect_rnn_learn_with_adapt_lr(rnn, &adapt_lr);
}


static void test_rnn_jacobian_matrix (struct rnn_parameters *rnn_p)
{
    double **matrix;
    double prev_c_inter_state[rnn_p->c_state_size],
           prev_c_state[rnn_p->c_state_size],
           c_inter_state[rnn_p->c_state_size],
           c_state[rnn_p->c_state_size],
           o_inter_state[rnn_p->out_state_size],
           out_state[rnn_p->out_state_size];
    double **tmp_p;

    MALLOC2(matrix, rnn_p->out_state_size + rnn_p->c_state_size,
            rnn_p->in_state_size + rnn_p->c_state_size);
    rnn_p->output_type = STANDARD_TYPE;
    init_genrand(584937L);
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        prev_c_inter_state[i] = 2* genrand_real1()-1;
        prev_c_state[i] = tanh(prev_c_inter_state[i]);
        c_inter_state[i] = 2* genrand_real1()-1;
        c_state[i] = tanh(c_inter_state[i]);
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        o_inter_state[i] = 2 * genrand_real1()-1;
        out_state[i] = tanh(o_inter_state[i]);
    }
    tmp_p = rnn_jacobian_matrix(matrix, rnn_p, prev_c_state, c_state,
            out_state);
    assert_equal_pointer(matrix, tmp_p);
    assert_jacobian_matrix_for_standard(rnn_p, matrix, prev_c_inter_state,
            c_inter_state, o_inter_state);

    rnn_p->output_type = SOFTMAX_TYPE;
    init_genrand(537L);
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        prev_c_inter_state[i] = 2* genrand_real1()-1;
        prev_c_state[i] = tanh(prev_c_inter_state[i]);
        c_inter_state[i] = 2* genrand_real1()-1;
        c_state[i] = tanh(c_inter_state[i]);
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        o_inter_state[i] = 2 * genrand_real1()-1;
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        double sum = 0;
        for (int j = 0; j < rnn_p->out_state_size; j++) {
            sum += o_inter_state[j];
        }
        out_state[i] = o_inter_state[i] / sum;
    }
    tmp_p = rnn_jacobian_matrix(matrix, rnn_p, prev_c_state, c_state,
            out_state);
    assert_equal_pointer(matrix, tmp_p);
    assert_jacobian_matrix_for_softmax(rnn_p, matrix, prev_c_inter_state,
            c_inter_state, out_state);

    FREE2(matrix);
}


static void test_rnn_update_prior_strength (
        struct recurrent_neural_network *rnn)
{
    double tmp_value = rnn->rnn_p.prior_strength;
    double lambda, alpha, total_length, value;
    lambda = 0.9;
    alpha = 1.0;
    total_length = 0;
    for (int i = 0; i < rnn->series_num; i++) {
        total_length += rnn->rnn_s[i].length;
    }
    rnn_update_prior_strength(rnn, lambda, alpha);
    value = lambda * tmp_value + alpha * total_length;
    assert_equal_double(value, rnn->rnn_p.prior_strength, 1e-10);
}





void test_rnn_state_setup (
        struct recurrent_neural_network *rnn,
        int target_num,
        int *target_length)
{
    const int in_state_size = rnn->rnn_p.in_state_size;
    const int out_state_size = rnn->rnn_p.out_state_size;
    for (int i = 0; i < target_num; i++) {
        double **input, **target;
        MALLOC2(input, target_length[i], in_state_size);
        MALLOC2(target, target_length[i], out_state_size);
        for (int n = 0; n < target_length[i]; n++) {
            for (int j = 0; j < in_state_size; j++) {
                input[n][j] = genrand_real1();
            }
            for (int j = 0; j < out_state_size; j++) {
                target[n][j] = genrand_real1();
            }
        }
        rnn_add_target(rnn, target_length[i], (const double* const*)input,
                (const double* const*)target);
        FREE2(input);
        FREE2(target);
    }
}


typedef struct test_rnn_data {
    struct recurrent_neural_network rnn;
    int target_num;
    int total_length;
} test_rnn_data;

static void test_rnn_data_setup (
        struct test_rnn_data *t_data,
        unsigned long seed,
        int in_state_size,
        int c_state_size,
        int out_state_size,
        int target_num,
        int *target_length)
{
    struct recurrent_neural_network *rnn = &t_data->rnn;

    init_genrand(seed);

    init_recurrent_neural_network(rnn, in_state_size, c_state_size,
            out_state_size);

    t_data->target_num = target_num;
    t_data->total_length = 0;
    for (int i = 0; i < target_num; i++) {
        t_data->total_length += target_length[i];
    }
    test_rnn_state_setup(rnn, target_num, target_length);
}


void test_rnn (void)
{
    mu_run_test(test_init_rnn_parameters);
    mu_run_test(test_init_rnn_state);
    mu_run_test(test_rnn_get_connection);

    struct test_rnn_data t_data[5];
    test_rnn_data_setup(t_data, 429837L, 10, 15, 7, 3, (int[]){100,100,50});
    test_rnn_data_setup(t_data+1, 3837L, 4, 10, 4, 2, (int[]){100,100});
    test_rnn_data_setup(t_data+2, 1181L, 0, 10, 5, 4, (int[]){75,50, 200, 35});
    test_rnn_data_setup(t_data+3, 400L, 5, 8, 5, 2, (int[]){75,50});
    t_data[3].rnn.rnn_p.softmax_group_num = 2;
    for (int i = 0; i < t_data[3].rnn.rnn_p.out_state_size; i++) {
        t_data[3].rnn.rnn_p.softmax_group_id[i] = i % 2;
    }
    test_rnn_data_setup(t_data+4, 400L, 0, 8, 6, 3, (int[]){75,100,100});
    t_data[4].rnn.rnn_p.softmax_group_num = 3;
    t_data[4].rnn.rnn_p.softmax_group_id[0] =
        t_data[4].rnn.rnn_p.softmax_group_id[1] = 0;
    t_data[4].rnn.rnn_p.softmax_group_id[2] =
        t_data[4].rnn.rnn_p.softmax_group_id[3] = 2;
    t_data[4].rnn.rnn_p.softmax_group_id[4] =
        t_data[4].rnn.rnn_p.softmax_group_id[5] = 1;
    rnn_delete_connection(t_data[0].rnn.rnn_p.in_state_size,
            t_data[0].rnn.rnn_p.connection_ci[4], 2, 3);
    rnn_delete_connection(t_data[0].rnn.rnn_p.c_state_size,
            t_data[0].rnn.rnn_p.connection_cc[2], 3, 15);
    rnn_delete_connection(t_data[0].rnn.rnn_p.c_state_size,
            t_data[0].rnn.rnn_p.connection_cc[8], 0, 7);
    rnn_delete_connection(t_data[0].rnn.rnn_p.c_state_size,
            t_data[0].rnn.rnn_p.connection_oc[3], 3, 9);
    rnn_reset_weight_by_connection(&t_data[0].rnn.rnn_p);

    rnn_delete_connection(t_data[4].rnn.rnn_p.c_state_size,
            t_data[4].rnn.rnn_p.connection_cc[4], 1, 2);
    rnn_delete_connection(t_data[4].rnn.rnn_p.c_state_size,
            t_data[4].rnn.rnn_p.connection_oc[4], 4, 8);
    rnn_reset_weight_by_connection(&t_data[4].rnn.rnn_p);
    for (int i = 0; i < 5; i++) {
        mu_run_test_with_args(test_fwrite_recurrent_neural_network,
                &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_set_uniform_tau, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_set_tau, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_set_sigma, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_get_total_length, &t_data[i].rnn,
                t_data[i].total_length);
        mu_run_test_with_args(test_rnn_get_error, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_get_total_error, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_get_likelihood, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_get_total_likelihood, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_forward_context_map,
                &t_data[i].rnn.rnn_p);
        mu_run_test_with_args(test_rnn_forward_output_map,
                &t_data[i].rnn.rnn_p);
        mu_run_test_with_args(test_rnn_forward_dynamics_forall, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_forward_backward_dynamics_forall,
                &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_forward_dynamics_in_closed_loop_forall,
                &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_learn, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_learn_s, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_backup_learning_parameters,
                &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_learn_with_adapt_lr, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_jacobian_matrix, &t_data[i].rnn.rnn_p);
        mu_run_test_with_args(test_rnn_update_prior_strength, &t_data[i].rnn);
        mu_run_test_with_args(test_rnn_clean_target, &t_data[i].rnn);
    }
    for (int i = 0; i < 5; i++) {
        free_recurrent_neural_network(&t_data[i].rnn);
    }
}


