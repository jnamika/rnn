/*
    Copyright (c) 2011, Jun Namikawa <jnamika@gmail.com>

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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "utils.h"
#include "lyapunov.h"
#include "rnn_runner.h"
#include "rnn_lyapunov.h"


static void memcopy_rnn_state (
        struct rnn_state *dst,
        struct rnn_state *src,
        int dst_step,
        int src_step)
{
    const struct rnn_parameters *rnn_p = src->rnn_p;
    memcpy(dst->in_state[dst_step], src->in_state[src_step],
            rnn_p->in_state_size * sizeof(double));
    memcpy(dst->c_inputsum[dst_step], src->c_inputsum[src_step],
            rnn_p->c_state_size * sizeof(double));
    memcpy(dst->c_inter_state[dst_step], src->c_inter_state[src_step],
            rnn_p->c_state_size * sizeof(double));
    memcpy(dst->c_state[dst_step], src->c_state[src_step],
            rnn_p->c_state_size * sizeof(double));
    memcpy(dst->o_inter_state[dst_step], src->o_inter_state[src_step],
            rnn_p->out_state_size * sizeof(double));
    memcpy(dst->out_state[dst_step], src->out_state[src_step],
            rnn_p->out_state_size * sizeof(double));
    if (dst_step == 0 && src_step == 0) {
        memcpy(dst->init_c_inter_state, src->init_c_inter_state,
                rnn_p->c_state_size * sizeof(double));
        memcpy(dst->init_c_state, src->init_c_state,
                rnn_p->c_state_size * sizeof(double));
    }
}

static double gauss_dev()
{
    static int iset = 0;
    static double gset;
    double fac, rsq, v1, v2;
    if (iset == 0) {
        do {
            v1 = 2.0 * genrand_real1() - 1.0;
            v2 = 2.0 * genrand_real1() - 1.0;
            rsq = v1*v1 + v2*v2;
        } while (rsq >= 1.0 || fpclassify(rsq) == FP_ZERO);
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        iset = 1;
        return v2 * fac;
    } else {
        iset = 0;
        return gset;
    }
}

static void update_rnn_runner_with_noise (
        struct rnn_runner *runner,
        double noise_deviation)
{
    double *in_state = rnn_in_state_from_runner(runner);
    if (fpclassify(noise_deviation) != FP_ZERO) {
        for (int i = 0; i < runner->rnn.rnn_p.in_state_size; i++) {
            in_state[i] += noise_deviation * gauss_dev();
        }
    }
    update_rnn_runner(runner);
}


void compute_lyapunov_main (
        const struct analysis_parameters *ap,
        struct rnn_runner *runner)
{
    rnn_add_target(&runner->rnn, ap->mem_size, NULL, NULL);
    int delay_length = rnn_delay_length_from_runner(runner);
    struct rnn_state *rnn_s = runner->rnn.rnn_s + runner->rnn.series_num - 1;
    struct rnn_lyapunov_info rl_info;
    init_rnn_lyapunov_info(&rl_info, rnn_s, delay_length, 0);

    int spectrum_size;
    if (ap->lyapunov_spectrum_size < 0 ||
            ap->lyapunov_spectrum_size > rl_info.dimension) {
        spectrum_size = rl_info.dimension;
    } else {
        spectrum_size = ap->lyapunov_spectrum_size;
    }
    double lyapunov[spectrum_size], tmp[spectrum_size];
    for (int i = 0; i < ap->sample_num; i++) {
        set_init_state_of_rnn_runner(runner, -1);
        for (long n = 0; n < ap->truncate_length; n++) {
            update_rnn_runner_with_noise(runner, ap->noise_deviation);
        }
        for (int j = 0; j < spectrum_size; j++) {
            lyapunov[j] = 0;
        }
        for (long n = 0; n < ap->length; n++) {
            int m = (int)(n % rnn_s->length);
            update_rnn_runner_with_noise(runner, ap->noise_deviation);
            memcopy_rnn_state(rnn_s, rnn_state_from_runner(runner), m, 0);
            if ((m+1) >= rnn_s->length) {
                rnn_lyapunov_spectrum(&rl_info, tmp, spectrum_size);
                for (int j = 0; j < spectrum_size; j++) {
                    lyapunov[j] += tmp[j] * rnn_s->length;
                }
            }
        }
        if ((ap->length % rnn_s->length) > 0) {
            int len = rnn_s->length;
            rnn_s->length = ap->length % rnn_s->length;
            rnn_lyapunov_spectrum(&rl_info, tmp, spectrum_size);
            for (int j = 0; j < spectrum_size; j++) {
                lyapunov[j] += tmp[j] * rnn_s->length;
            }
            rnn_s->length = len;
        }
        for (int j = 0; j < spectrum_size; j++) {
            printf("%f%c", lyapunov[j] / ap->length,
                    (j + 1 < spectrum_size) ? '\t' : '\n');
        }
    }
    free_rnn_lyapunov_info(&rl_info);
}

