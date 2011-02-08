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
#include "solver.h"


typedef struct test_solver_info {
    double **data;
    double **vector;
    double *spectrum;
    int length;
    int dim;
} test_solver_info;

static void init_test_solver_info (
        struct test_solver_info *ts_info,
        int length,
        int dim)
{
    ts_info->length = length;
    ts_info->dim = dim;

    MALLOC2(ts_info->data, length, dim);
    MALLOC(ts_info->vector, dim);
    for (int i = 0; i < dim; i++) {
        MALLOC(ts_info->vector[i], dim);
    }
    MALLOC(ts_info->spectrum, dim);
}

static void free_test_solver_info (struct test_solver_info *ts_info)
{
    FREE2(ts_info->data);
    for (int i = 0; i < ts_info->dim; i++) {
        FREE(ts_info->vector[i]);
    }
    FREE(ts_info->vector);
    FREE(ts_info->spectrum);
}



/* assert functions */


/* test functions */

static void test_gram_schmidt_orthogonalization (
        struct test_solver_info *ts_info)
{
    if (ts_info->dim < 2) return;

    ts_info->vector[0][0] = 1;
    ts_info->vector[0][1] = 1;
    ts_info->vector[1][0] = 0;
    ts_info->vector[1][1] = 1;

    gram_schmidt_orthogonalization(ts_info->vector, 2, 2);

    assert_equal_double(1.0, ts_info->vector[0][0], 1e-10);
    assert_equal_double(1.0, ts_info->vector[0][1], 1e-10);
    assert_equal_double(-0.5, ts_info->vector[1][0], 1e-10);
    assert_equal_double(0.5, ts_info->vector[1][1], 1e-10);

    if (ts_info->dim < 4) return;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            ts_info->vector[i][j] = 0;
        }
        ts_info->vector[i][i] = i+1;
    }
    gram_schmidt_orthogonalization(ts_info->vector, 4, 4);
    assert_equal_double(4.0, ts_info->vector[0][3], 1e-10);
    assert_equal_double(3.0, ts_info->vector[1][2], 1e-10);
    assert_equal_double(2.0, ts_info->vector[2][1], 1e-10);
    assert_equal_double(1.0, ts_info->vector[3][0], 1e-10);
}

static double logistic_map (double x, double a)
{
    return a * x * (1-x);
}

static double** dlogistic_map(const double* vector, int n, int t,
        double** matrix, void *obj)
{
    if (n < 1 || t < 0) return NULL;

    double *p = (double*)obj;
    double a = *p;
    matrix[0][0] = a - 2 * a * vector[0];
    return matrix;
}

typedef struct henon_info {
    double a;
    double b;
} henon_info;

static void henon_map (
        const struct henon_info *h_info,
        const double *x,
        double *y)
{
    y[0] = 1 - h_info->a * (x[0] * x[0]) + x[1];
    y[1] = h_info->b * x[0];
}


static double** dhenon_map (const double* vector, int n, int t,
        double** matrix, void *obj)
{
    if (n < 2 || t < 0) return NULL;

    struct henon_info *h_info = (struct henon_info*)obj;
    matrix[0][0] = -2.0 * h_info->a * vector[0];
    matrix[0][1] = 1.0;
    matrix[1][0] = h_info->b;
    matrix[1][1] = 0.0;
    return matrix;
}

static void test_lyapunov_spectrum (struct test_solver_info *ts_info)
{
    if (ts_info->length <= 0 || ts_info->dim < 1) return;

    double a;
    a = 4.0;
    ts_info->data[0][0] = 0.223;
    for (int n = 1; n < ts_info->length; n++) {
        ts_info->data[n][0] = logistic_map(ts_info->data[n-1][0], a);
    }
    lyapunov_spectrum((const double* const*)ts_info->data, ts_info->length,
            1, 1, 5, dlogistic_map, &a, ts_info->spectrum, NULL, NULL);

    mu_assert(ts_info->spectrum[0] > 0);

    a = 3.5;
    ts_info->data[0][0] = 0.341;
    for (int n = 1; n < ts_info->length; n++) {
        ts_info->data[n][0] = logistic_map(ts_info->data[n-1][0], a);
    }
    lyapunov_spectrum((const double* const*)ts_info->data, ts_info->length,
            1, 1, 5, dlogistic_map, &a, ts_info->spectrum, NULL, NULL);

    mu_assert(ts_info->spectrum[0] <= 0);


    if (ts_info->dim < 2) return;

    struct henon_info h_info;
    h_info.a = 1.4;
    h_info.b = 0.3;
    ts_info->data[0][0] = 0.4;
    ts_info->data[0][1] = 0.15;
    for (int n = 1; n < ts_info->length; n++) {
        henon_map(&h_info, ts_info->data[n-1], ts_info->data[n]);
    }
    lyapunov_spectrum((const double* const*)ts_info->data, ts_info->length,
            2, 2, 5, dhenon_map, &h_info, ts_info->spectrum, NULL, NULL);

    mu_assert(ts_info->spectrum[0] > 0);
    mu_assert(ts_info->spectrum[1] <= ts_info->spectrum[0]);

    h_info.a = 0.9;
    h_info.b = 0.3;
    ts_info->data[0][0] = 0.4;
    ts_info->data[0][1] = 0.15;
    for (int n = 1; n < ts_info->length; n++) {
        henon_map(&h_info, ts_info->data[n-1], ts_info->data[n]);
    }
    lyapunov_spectrum((const double* const*)ts_info->data, ts_info->length,
            2, 2, 5, dhenon_map, &h_info, ts_info->spectrum, NULL, NULL);

    mu_assert(ts_info->spectrum[0] <= 0);
    mu_assert(ts_info->spectrum[1] <= ts_info->spectrum[0]);
}


static void test_lyapunov_exponent (struct test_solver_info *ts_info)
{
    if (ts_info->length <= 0 || ts_info->dim < 1) return;

    double ly_exp;
    double a;
    a = 4.0;
    ts_info->data[0][0] = 0.223;
    for (int n = 1; n < ts_info->length; n++) {
        ts_info->data[n][0] = logistic_map(ts_info->data[n-1][0], a);
    }
    ly_exp = lyapunov_exponent_sss((const double* const*)ts_info->data,
            ts_info->length, 1, 10, 1000);
    mu_assert(ly_exp > 0);
    ly_exp = lyapunov_exponent_wolf((const double* const*)ts_info->data,
            ts_info->length, 1, 0.1);
    mu_assert(ly_exp > 0);

    a = 3.5;
    ts_info->data[0][0] = 0.341;
    for (int n = 1; n < ts_info->length; n++) {
        ts_info->data[n][0] = logistic_map(ts_info->data[n-1][0], a);
    }
    ly_exp = lyapunov_exponent_sss((const double* const*)ts_info->data,
            ts_info->length, 1, 10, 1000);
    mu_assert(ly_exp <= 0);
    ly_exp = lyapunov_exponent_wolf((const double* const*)ts_info->data,
            ts_info->length, 1, 0.1);
    mu_assert(ly_exp <= 0);


    if (ts_info->dim < 2) return;

    struct henon_info h_info;
    h_info.a = 1.4;
    h_info.b = 0.3;
    ts_info->data[0][0] = 0.4;
    ts_info->data[0][1] = 0.15;
    for (int n = 1; n < ts_info->length; n++) {
        henon_map(&h_info, ts_info->data[n-1], ts_info->data[n]);
    }
    ly_exp = lyapunov_exponent_sss((const double* const*)ts_info->data,
            ts_info->length, 2, 10, 1000);
    mu_assert(ly_exp > 0);
    ly_exp = lyapunov_exponent_wolf((const double* const*)ts_info->data,
            ts_info->length, 2, 0.1);
    mu_assert(ly_exp > 0);

    h_info.a = 0.9;
    h_info.b = 0.3;
    ts_info->data[0][0] = 0.4;
    ts_info->data[0][1] = 0.15;
    for (int n = 1; n < ts_info->length; n++) {
        henon_map(&h_info, ts_info->data[n-1], ts_info->data[n]);
    }
    ly_exp = lyapunov_exponent_sss((const double* const*)ts_info->data,
            ts_info->length, 2, 10, 1000);
    mu_assert(ly_exp <= 0);
    ly_exp = lyapunov_exponent_wolf((const double* const*)ts_info->data,
            ts_info->length, 2, 0.1);
    mu_assert(ly_exp <= 0);
}



static void test_box_counter (struct test_solver_info *ts_info)
{
    int box_num, *box_count;
    MALLOC(box_count, ts_info->length);

    for (int n = 0; n < ts_info->length; n++) {
        for (int i = 0; i < ts_info->dim; i++) {
            ts_info->data[n][i] = n;
        }
    }
    box_num = box_counter((const double* const*)ts_info->data,
            ts_info->length, ts_info->dim, 1.0, box_count);
    assert_equal_int(ts_info->length, box_num);

    for (int n = 0; n < ts_info->length; n++) {
        for (int i = 0; i < ts_info->dim; i++) {
            ts_info->data[n][i] = 0;
        }
    }
    box_num = box_counter((const double* const*)ts_info->data,
            ts_info->length, ts_info->dim, 1.0, box_count);
    assert_equal_int(1, box_num);

    for (int n = 0; n < ts_info->length; n++) {
        for (int i = 0; i < ts_info->dim; i++) {
            ts_info->data[n][i] = (n % 10);
        }
    }
    box_num = box_counter((const double* const*)ts_info->data,
            ts_info->length, ts_info->dim, 1.0, box_count);
    assert_equal_int(10, box_num);

    FREE(box_count);
}


static void test_generalized_dimension (struct test_solver_info *ts_info)
{
    if (ts_info->length <= 0 || ts_info->dim < 1) return;

    int box_num, *box_count;
    MALLOC(box_count, ts_info->length);

    init_genrand(5747L);
    for (int n = 0; n < ts_info->length; n++) {
        ts_info->data[n][0] = genrand_real1();
    }

    int dnum = 10;
    double dimension[dnum];
    for (int q = 0; q < dnum; q++) {
        dimension[q] = generalized_dimension(
                (const double* const*)ts_info->data, ts_info->length,
                1, 1e-2, q, &box_num);
    }
    for (int q = 1; q < dnum; q++) {
        mu_assert(dimension[q-1] >= dimension[q]);
    }

    FREE(box_count);
}


void test_solver (void)
{
    struct test_solver_info ts_info;
    init_test_solver_info(&ts_info, 1000, 4);

    mu_run_test_with_args(test_gram_schmidt_orthogonalization, &ts_info);
    mu_run_test_with_args(test_lyapunov_spectrum, &ts_info);
    mu_run_test_with_args(test_lyapunov_exponent, &ts_info);
    mu_run_test_with_args(test_box_counter, &ts_info);
    mu_run_test_with_args(test_generalized_dimension, &ts_info);

    free_test_solver_info(&ts_info);
}


