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

#define TEST_CODE
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "minunit.h"
#include "my_assert.h"
#include "utils.h"
#include "rnn_runner.h"


/* assert functions */

void assert_equal_rnn_p (
        struct rnn_parameters *rnn1_p,
        struct rnn_parameters *rnn2_p);

void assert_equal_rnn_s (
        struct rnn_state *rnn1_s,
        struct rnn_state *rnn2_s);




/* test functions */

static void test_new_rnn_runner (void)
{
    struct rnn_runner *runner;
    int stat = _new_rnn_runner(&runner);
    assert_equal_int(0, stat);
    mu_assert(runner != NULL);
    _delete_rnn_runner(runner);
}

typedef struct test_rnn_runner_data {
    struct rnn_runner runner;
    int in_state_size;
    int c_state_size;
    int out_state_size;
    int delay_length;
    rnn_output_t output_type;
    int target_num;
} test_rnn_runner_data;



void test_rnn_state_setup (
        struct recurrent_neural_network *rnn,
        int target_num,
        int *target_length);

static void test_init_rnn_runner (
        struct test_rnn_runner_data *t_data,
        int in_state_size,
        int c_state_size,
        int out_state_size,
        int delay_length,
        rnn_output_t output_type,
        int target_num,
        int *target_length)
{
    struct recurrent_neural_network rnn;
    FILE *fp;

    init_recurrent_neural_network(&rnn, in_state_size, c_state_size,
            out_state_size);
    rnn.rnn_p.output_type = output_type;
    test_rnn_state_setup(&rnn, target_num, target_length);

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("Cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    if (fwrite(&delay_length, sizeof(int), 1, fp) != 1) {
        print_error_msg();
        exit(EXIT_FAILURE);
    }
    fwrite_recurrent_neural_network(&rnn, fp);
    fseek(fp, 0L, SEEK_SET);
    init_rnn_runner(&t_data->runner, fp);
    fclose(fp);

    t_data->in_state_size = in_state_size;
    t_data->c_state_size = c_state_size;
    t_data->out_state_size = out_state_size;
    t_data->delay_length = delay_length;
    t_data->output_type = output_type;
    t_data->target_num = target_num;

    assert_equal_int(in_state_size,
            rnn_in_state_size_from_runner(&t_data->runner));
    assert_equal_int(c_state_size,
            rnn_c_state_size_from_runner(&t_data->runner));
    assert_equal_int(out_state_size,
            rnn_out_state_size_from_runner(&t_data->runner));
    assert_equal_int(delay_length,
            rnn_delay_length_from_runner(&t_data->runner));
    assert_equal_int((int)output_type,
            rnn_output_type_from_runner(&t_data->runner));
    assert_equal_int(target_num, rnn_target_num_from_runner(&t_data->runner));
    assert_equal_rnn_p(&rnn.rnn_p, &t_data->runner.rnn.rnn_p);
    for (int i = 0; i < target_num; i++) {
        assert_equal_rnn_s(rnn.rnn_s + i, t_data->runner.rnn.rnn_s + i);
    }
    free_recurrent_neural_network(&rnn);
}


static void test_set_init_state_of_rnn_runner (
        struct test_rnn_runner_data *t_data)
{
    struct rnn_runner *runner = &t_data->runner;
    for (int i = 0; i < t_data->target_num; i++) {
        set_init_state_of_rnn_runner(runner, i);
        int length = rnn_delay_length_from_runner(runner);
        if (length > runner->rnn.rnn_s[i].length) {
            length = runner->rnn.rnn_s[i].length;
        }
        const struct rnn_state *dst = rnn_state_from_runner(runner);
        const struct rnn_state *src = runner->rnn.rnn_s + i;
        assert_equal_vector_sequence(src->in_state, src->rnn_p->in_state_size,
                length, dst->in_state, dst->rnn_p->in_state_size, length);
        assert_equal_memory(src->init_c_state, src->rnn_p->c_state_size *
                sizeof(double), dst->init_c_state, dst->rnn_p->c_state_size *
                sizeof(double));
        assert_equal_memory(src->init_c_inter_state, src->rnn_p->c_state_size *
                sizeof(double), dst->init_c_inter_state,
                dst->rnn_p->c_state_size * sizeof(double));
    }
}

static void test_update_rnn_runner (struct test_rnn_runner_data *t_data)
{
    struct rnn_runner *runner = &t_data->runner;
    const int c_mem_size = t_data->c_state_size * sizeof(double);
    const int out_mem_size = t_data->out_state_size * sizeof(double);
    for (int i = 0; i < t_data->target_num; i++) {
        set_init_state_of_rnn_runner(runner, i);
        struct rnn_state *rnn_s = runner->rnn.rnn_s + i;
        rnn_forward_dynamics_in_closed_loop(rnn_s, t_data->delay_length);
        for (int n = 0; n < rnn_s->length; n++) {
            update_rnn_runner(runner);
            assert_equal_memory(rnn_s->out_state[n], out_mem_size,
                    rnn_out_state_from_runner(runner), out_mem_size);
            assert_equal_memory(rnn_s->c_state[n], c_mem_size,
                    rnn_c_state_from_runner(runner), c_mem_size);
            assert_equal_memory(rnn_s->c_inter_state[n], c_mem_size,
                    rnn_c_inter_state_from_runner(runner), c_mem_size);
        }
    }
}


static void test_rnn_in_state_size_from_runner (
        struct test_rnn_runner_data *t_data)
{
    assert_equal_int(t_data->in_state_size,
            rnn_in_state_size_from_runner(&t_data->runner));
}

static void test_rnn_c_state_size_from_runner (
        struct test_rnn_runner_data *t_data)
{
    assert_equal_int(t_data->c_state_size,
            rnn_c_state_size_from_runner(&t_data->runner));
}

static void test_rnn_out_state_size_from_runner (
        struct test_rnn_runner_data *t_data)
{
    assert_equal_int(t_data->out_state_size,
            rnn_out_state_size_from_runner(&t_data->runner));
}

static void test_rnn_delay_length_from_runner (
        struct test_rnn_runner_data *t_data)
{
    assert_equal_int(t_data->delay_length,
            rnn_delay_length_from_runner(&t_data->runner));
}

static void test_rnn_output_type_from_runner (
        struct test_rnn_runner_data *t_data)
{
    assert_equal_int((int)t_data->output_type,
            rnn_output_type_from_runner(&t_data->runner));
}

static void test_rnn_target_num_from_runner (
        struct test_rnn_runner_data *t_data)
{
    assert_equal_int(t_data->target_num,
            rnn_target_num_from_runner(&t_data->runner));
}

static void test_rnn_in_state_from_runner (struct test_rnn_runner_data *t_data)
{
    const struct rnn_state *rnn_s = rnn_state_from_runner(&t_data->runner);
    assert_equal_pointer(rnn_s->in_state[0],
            rnn_in_state_from_runner(&t_data->runner));
}

static void test_rnn_c_state_from_runner (struct test_rnn_runner_data *t_data)
{
    const struct rnn_state *rnn_s = rnn_state_from_runner(&t_data->runner);
    assert_equal_pointer(rnn_s->init_c_state,
            rnn_c_state_from_runner(&t_data->runner));
}

static void test_rnn_c_inter_state_from_runner (
        struct test_rnn_runner_data *t_data)
{
    const struct rnn_state *rnn_s = rnn_state_from_runner(&t_data->runner);
    assert_equal_pointer(rnn_s->init_c_inter_state,
            rnn_c_inter_state_from_runner(&t_data->runner));
}

static void test_rnn_out_state_from_runner (struct test_rnn_runner_data *t_data)
{
    const struct rnn_state *rnn_s = rnn_state_from_runner(&t_data->runner);
    assert_equal_pointer(rnn_s->out_state[0],
            rnn_out_state_from_runner(&t_data->runner));
}

static void test_rnn_state_from_runner (struct test_rnn_runner_data *t_data)
{
    const struct rnn_state *rnn_s = t_data->runner.rnn.rnn_s +
        t_data->runner.id;
    assert_equal_pointer(rnn_s, rnn_state_from_runner(&t_data->runner));
}



void test_rnn_runner (void)
{
    struct test_rnn_runner_data t_data[4];

    init_genrand(391919L);

    mu_run_test(test_new_rnn_runner);

    mu_run_test_with_args(test_init_rnn_runner, t_data, 1, 10, 1, 1,
            STANDARD_TYPE, 2, (int[]){50,100});
    mu_run_test_with_args(test_init_rnn_runner, t_data+1, 3, 13, 3, 3,
            SOFTMAX_TYPE, 3, (int[]){30,30,20});
    mu_run_test_with_args(test_init_rnn_runner, t_data+2, 0, 7, 2, 5,
            STANDARD_TYPE, 2, (int[]){100,50});
    mu_run_test_with_args(test_init_rnn_runner, t_data+3, 4, 10, 4, 10,
            SOFTMAX_TYPE, 3, (int[]){5,50,10});

    for (int i = 0; i < 4; i++) {
        mu_run_test_with_args(test_set_init_state_of_rnn_runner, t_data + i);
        mu_run_test_with_args(test_update_rnn_runner, t_data + i);
        mu_run_test_with_args(test_rnn_in_state_size_from_runner, t_data + i);
        mu_run_test_with_args(test_rnn_c_state_size_from_runner, t_data + i);
        mu_run_test_with_args(test_rnn_out_state_size_from_runner, t_data + i);
        mu_run_test_with_args(test_rnn_delay_length_from_runner, t_data + i);
        mu_run_test_with_args(test_rnn_output_type_from_runner, t_data + i);
        mu_run_test_with_args(test_rnn_target_num_from_runner, t_data + i);
        mu_run_test_with_args(test_rnn_in_state_from_runner, t_data + i);
        mu_run_test_with_args(test_rnn_c_state_from_runner, t_data + i);
        mu_run_test_with_args(test_rnn_c_inter_state_from_runner, t_data + i);
        mu_run_test_with_args(test_rnn_out_state_from_runner, t_data + i);
        mu_run_test_with_args(test_rnn_state_from_runner, t_data + i);

        free_rnn_runner(&t_data[i].runner);
    }
}


