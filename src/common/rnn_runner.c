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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "mt19937ar.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "utils.h"
#include "rnn_runner.h"


/******************************************************************************/
/********** Initialization and Free *******************************************/
/******************************************************************************/


int _new_rnn_runner (struct rnn_runner **runner)
{
    (*runner) = malloc(sizeof(struct rnn_runner));
    if ((*runner) == NULL) {
        return 1;
    }
    return 0;
}

void _delete_rnn_runner (struct rnn_runner *runner)
{
    free(runner);
}




void init_rnn_runner (
        struct rnn_runner *runner,
        FILE *fp)
{
    int delay_length;

    FREAD(&delay_length, 1, fp);
    fread_recurrent_neural_network(&runner->rnn, fp);
    rnn_add_target(&runner->rnn, delay_length, NULL, NULL);
    runner->id = runner->rnn.series_num - 1;
}


void free_rnn_runner (struct rnn_runner *runner)
{
    free_recurrent_neural_network(&runner->rnn);
}


static void copy_init_state (
        struct rnn_state *dst,
        struct rnn_state *src)
{
    for (int n = 0; n < dst->length; n++) {
        if (n < src->length) {
            memmove(dst->in_state[n], src->in_state[n], sizeof(double) *
                    dst->rnn_p->in_state_size);
        } else {
            for (int i = 0; i < dst->rnn_p->in_state_size; i++) {
                dst->in_state[n][i] = (2*genrand_real3()-1);
            }
        }
    }
    memmove(dst->init_c_state, src->init_c_state, sizeof(double) *
            dst->rnn_p->c_state_size);
    memmove(dst->init_c_inter_state, src->init_c_inter_state, sizeof(double) *
            dst->rnn_p->c_state_size);
}

static void random_init_state (struct rnn_state *rnn_s)
{
    for (int n = 0; n < rnn_s->length; n++) {
        for (int i = 0; i < rnn_s->rnn_p->in_state_size; i++) {
            rnn_s->in_state[n][i] = (2*genrand_real3()-1);
        }
    }
    for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++) {
        rnn_s->init_c_state[i] = (2*genrand_real3()-1);
        rnn_s->init_c_inter_state[i] = atanh(rnn_s->init_c_state[i]);
    }
}


void set_init_state_of_rnn_runner (
        struct rnn_runner *runner,
        int series_id)
{
    if (series_id >= 0 && series_id < runner->id) {
        copy_init_state(runner->rnn.rnn_s + runner->id, runner->rnn.rnn_s +
                series_id);
    } else {
        random_init_state(runner->rnn.rnn_s + runner->id);
    }
}


/******************************************************************************/
/********** Computation of forward dynamics ***********************************/
/******************************************************************************/


static void rnn_fmap (struct rnn_state *rnn_s)
{
    const struct rnn_parameters *rnn_p = rnn_s->rnn_p;

    assert(rnn_p->in_state_size <= rnn_p->out_state_size);

    rnn_forward_map(rnn_p, rnn_s->in_state[0], rnn_s->init_c_inter_state,
            rnn_s->init_c_state, rnn_s->c_inputsum[0], rnn_s->c_inter_state[0],
            rnn_s->c_state[0], rnn_s->o_inter_state[0], rnn_s->out_state[0]);

    for (int n = 1; n < rnn_s->length; n++) {
        memmove(rnn_s->in_state[n-1], rnn_s->in_state[n], sizeof(double) *
                rnn_p->in_state_size);
    }
    memmove(rnn_s->in_state[rnn_s->length-1], rnn_s->out_state[0],
            sizeof(double) * rnn_p->in_state_size);
    memmove(rnn_s->init_c_state, rnn_s->c_state[0], sizeof(double) *
            rnn_p->c_state_size);
    memmove(rnn_s->init_c_inter_state, rnn_s->c_inter_state[0], sizeof(double) *
            rnn_p->c_state_size);
}


void update_rnn_runner (struct rnn_runner *runner)
{
    rnn_fmap(runner->rnn.rnn_s + runner->id);
}



/******************************************************************************/
/********** Interface *********************************************************/
/******************************************************************************/

int rnn_in_state_size_from_runner (struct rnn_runner *runner)
{
    return runner->rnn.rnn_p.in_state_size;
}

int rnn_c_state_size_from_runner (struct rnn_runner *runner)
{
    return runner->rnn.rnn_p.c_state_size;
}

int rnn_out_state_size_from_runner (struct rnn_runner *runner)
{
    return runner->rnn.rnn_p.out_state_size;
}

int rnn_delay_length_from_runner (struct rnn_runner *runner)
{
    return runner->rnn.rnn_s[runner->id].length;
}

int rnn_target_num_from_runner (struct rnn_runner *runner)
{
    return runner->id;
}

double* rnn_in_state_from_runner (struct rnn_runner *runner)
{
    return runner->rnn.rnn_s[runner->id].in_state[0];
}

double* rnn_c_state_from_runner (struct rnn_runner *runner)
{
    return runner->rnn.rnn_s[runner->id].init_c_state;
}

double* rnn_c_inter_state_from_runner (struct rnn_runner *runner)
{
    return runner->rnn.rnn_s[runner->id].init_c_inter_state;
}

double* rnn_out_state_from_runner (struct rnn_runner *runner)
{
    return runner->rnn.rnn_s[runner->id].out_state[0];
}

struct rnn_state* rnn_state_from_runner (struct rnn_runner *runner)
{
    return runner->rnn.rnn_s + runner->id;
}

