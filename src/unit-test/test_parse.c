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
#include <stddef.h>
#include <string.h>
#include <math.h>

#include "minunit.h"
#include "my_assert.h"
#include "utils.h"
#include "parse.h"


/* assert functions */


/* test functions */


static void test_str_to_connection (void)
{
    char *str = NULL;
    int **has_connection;
    double **connectivity;

    MALLOC(str, BUFSIZ);
    MALLOC2(has_connection, 10, 10);
    MALLOC2(connectivity, 10, 10);

    int has_connection2[10][2] = {{1,0},{1,0},{1,0},{1,0},{1,0},{0,1},{0,1},
        {0,1}, {0,1},{0,1}};
    strncpy(str, "1t1-5,2t6-10", BUFSIZ);
    str_to_connection(str, 2, 10, has_connection, connectivity);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 2; j++) {
            assert_equal_int(has_connection2[i][j], has_connection[i][j]);
            assert_equal_double(1.0, connectivity[i][j], 1e-12);
        }
    }
    int has_connection3[3][5] = {{1,1,0,0,0},{0,1,0,1,1},{0,0,1,0,0}};
    double connectivity3[3][5] = {{1,1,1,1,1},{1,1,1,0.2,0.2},{1,1,0.4,1,1}};
    strncpy(str, "1-2t1-1,3t3:0.4,2t1-2,4-5t2:0.2", BUFSIZ);
    str_to_connection(str, 5, 3, has_connection, connectivity);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            assert_equal_int(has_connection3[i][j], has_connection[i][j]);
            assert_equal_double(connectivity3[i][j], connectivity[i][j], 1e-12);
        }
    }
    int has_connection4[6][4] = {{0,0,1,1}, {1,1,1,1}, {0,0,1,1}, {1,1,1,1},
        {0,0,0,1}, {0,0,0,1}};
    double connectivity4[6][4] = {{1,1,1,0.75}, {0.1,0.1,1,0.75}, {1,1,1,0.75},
        {1,1,1,0.75}, {1,1,1,0.75}, {1,1,1,0.75}};
    strncpy(str, "-t4,1-2t2:0.1,3t-4,4t-:0.75", BUFSIZ);
    str_to_connection(str, 4, 6, has_connection, connectivity);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 4; j++) {
            assert_equal_int(has_connection4[i][j], has_connection[i][j]);
            assert_equal_double(connectivity4[i][j], connectivity[i][j], 1e-12);
        }
    }
    strncpy(str, "1t1-5,2-10", BUFSIZ); // syntax error case
    str_to_connection(str, 2, 10, has_connection, connectivity);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 2; j++) {
            assert_equal_int(1, has_connection[i][j]);
            assert_equal_double(1, connectivity[i][j], 1e-12);
        }
    }
    FREE2(has_connection);
    FREE2(connectivity);
    FREE(str);
}


static void test_str_to_const_init_c (void)
{
    char *str;
    MALLOC(str, BUFSIZ);
    int const_init_c[10] = {0,};
    int const_init_c2[10] = {0,0,1,1,0,0,1,1,1,0};
    strncpy(str, "3-4,7-9", BUFSIZ);
    str_to_const_init_c(str, 10, const_init_c);
    for (int i = 0; i < 10; i++) {
        assert_equal_int(const_init_c2[i], const_init_c[i]);
    }
    int const_init_c3[10] = {1,1,0,0,0,0,0,1,1,1};
    strncpy(str, "-2,8-", BUFSIZ);
    str_to_const_init_c(str, 10, const_init_c);
    for (int i = 0; i < 10; i++) {
        assert_equal_int(const_init_c3[i], const_init_c[i]);
    }
    FREE(str);
}

static void test_str_to_softmax_group (void)
{
    char *str;

    MALLOC(str, BUFSIZ);

    int softmax_group_num = 0;
    int softmax_group_id[10] = {0,};
    int softmax_group_id2[10] = {0,0,1,1,2,2,1,2,2,2};
    strncpy(str, "3-4&7,1-2", BUFSIZ);
    str_to_softmax_group(str, 10, &softmax_group_num, softmax_group_id);
    assert_equal_int(3, softmax_group_num);
    for (int i = 0; i < 10; i++) {
        assert_equal_int(softmax_group_id2[i], softmax_group_id[i]);
    }
    int softmax_group_id3[10] = {0,1,2,1,3,3,2,1,1,2};
    strncpy(str, "2&4&8-9,1,5-6", BUFSIZ);
    str_to_softmax_group(str, 10, &softmax_group_num, softmax_group_id);
    assert_equal_int(4, softmax_group_num);
    for (int i = 0; i < 10; i++) {
        assert_equal_int(softmax_group_id3[i], softmax_group_id[i]);
    }
    FREE(str);
}


static void test_str_to_init_tau (void)
{
    char *str = NULL;
    MALLOC(str, BUFSIZ);
    double tau[10] = {0,};
    strncpy(str, "10", BUFSIZ);
    str_to_init_tau(str, 6, tau);
    for (int i = 0; i < 6; i++) {
        assert_equal_double(10.0, tau[i], 1e-10);
    }
    double tau2[10] = {10,2,2,2,2,10,4,4,4,4};
    strncpy(str, "10,2.0:2-5,4:7-", BUFSIZ);
    str_to_init_tau(str, 10, tau);
    for (int i = 0; i < 10; i++) {
        assert_equal_double(tau2[i], tau[i], 1e-10);
    }
    strncpy(str, "6:-4,inf:5-", BUFSIZ);
    str_to_init_tau(str, 8, tau);
    for (int i = 0; i < 8; i++) {
        if (i < 4) {
            assert_equal_double(6.0, tau[i], 1e-10);
        } else {
            mu_assert(isinf(tau[i]));
        }
    }
    FREE(str);
}


void test_parse (void)
{
    mu_run_test(test_str_to_connection);
    mu_run_test(test_str_to_const_init_c);
    mu_run_test(test_str_to_softmax_group);
    mu_run_test(test_str_to_init_tau);
}


