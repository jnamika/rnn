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
#include "mt19937ar.h"

#include "minunit.h"
#include "my_assert.h"
#include "utils.h"
#include "entropy.h"


static void gen_Morse_sequence (
        int *sequence,
        int length)
{
    if (length <= 0) return;

    sequence[0] = 0;
    for (int n = 1, b = 1; n < length; n++) {
        if (n >= 2 * b) {
            b *= 2;
        }
        sequence[n] = sequence[n-b] ? 0 : 1;
    }
}

/* assert functions */


/* test functions */


static void test_block_entropy (void)
{
    int length, max_block_length;
    double entropy;
    struct block_frequency bf;
    int *sequence;

    length = 10000;
    max_block_length = 5;

    MALLOC(sequence, length);

    for (int n = 0; n < length; n++) {
        sequence[n] = 0;
    }
    for (int n = 1; n <= max_block_length; n++) {
        init_block_frequency(&bf, sequence, length, n);
        entropy = block_entropy(&bf) / n;
        assert_equal_double(0, entropy, 1e-3);
        free_block_frequency(&bf);
    }

    for (int n = 0; n < length; n++) {
        sequence[n] = n % 2;
    }
    for (int n = 1; n <= max_block_length; n++) {
        init_block_frequency(&bf, sequence, length, n);
        entropy = block_entropy(&bf) / n;
        assert_equal_double(1.0/n, entropy, 1e-3);
        free_block_frequency(&bf);
    }

    for (int n = 0; n < length; n++) {
        sequence[n] = n % 4;
    }
    for (int n = 1; n <= max_block_length; n++) {
        init_block_frequency(&bf, sequence, length, n);
        entropy = block_entropy(&bf) / n;
        assert_equal_double(2.0/n, entropy, 1e-3);
        free_block_frequency(&bf);
    }

    gen_Morse_sequence(sequence, length);
    for (int n = 1; n <= max_block_length ; n++) {
        init_block_frequency(&bf, sequence, length, n);
        entropy = block_entropy(&bf) / n;
        mu_assert(entropy >= 0.5);
        free_block_frequency(&bf);
    }
    free(sequence);
}

static void test_kullback_leibler_divergence (void)
{
    int length, max_block_length;
    struct block_frequency bf_x, bf_y;
    int *x, *y;
    double kl_div;

    length = 1024;
    max_block_length = 5;

    MALLOC(x, length);
    MALLOC(y, length);

    gen_Morse_sequence(x, length);
    for (int n = 0; n < length; n++) {
        y[n] = x[(n+300)%length];
    }
    for (int n = 1; n < max_block_length; n++) {
        init_block_frequency(&bf_x, x, length, n);
        init_block_frequency(&bf_y, y, length, n);
        kl_div = kullback_leibler_divergence(&bf_x, &bf_y);
        assert_equal_double(0, kl_div, 1e-3);
        free_block_frequency(&bf_x);
        free_block_frequency(&bf_y);
    }
    init_genrand(61107L);
    for (int i = 0; i < 10; i++) {
        for (int n = 0; n < length; n++) {
            x[n] = genrand_int32() % 2;
            y[n] = genrand_int32() % 2;
        }
        for (int n = 1; n < max_block_length; n++) {
            init_block_frequency(&bf_x, x, length, n);
            init_block_frequency(&bf_y, y, length, n);
            kl_div = kullback_leibler_divergence(&bf_x, &bf_y);
            mu_assert(kl_div >= 0);
            free_block_frequency(&bf_x);
            free_block_frequency(&bf_y);

            init_block_frequency(&bf_x, x, length-100, n);
            init_block_frequency(&bf_y, y, length, n);
            kl_div = kullback_leibler_divergence(&bf_x, &bf_y);
            mu_assert(kl_div >= 0);
            free_block_frequency(&bf_x);
            free_block_frequency(&bf_y);

            init_block_frequency(&bf_x, x, length, n);
            init_block_frequency(&bf_y, y, length-100, n);
            kl_div = kullback_leibler_divergence(&bf_x, &bf_y);
            mu_assert(kl_div >= 0);
            free_block_frequency(&bf_x);
            free_block_frequency(&bf_y);
        }
    }

    for (int n = 0; n < length; n++) {
        x[n] = 0;
        y[n] = 1;
    }
    for (int n = 1; n < max_block_length; n++) {
        init_block_frequency(&bf_x, x, length, n);
        init_block_frequency(&bf_y, y, length, n);
        kl_div = kullback_leibler_divergence(&bf_x, &bf_y);
        mu_assert(kl_div > 0);
        free_block_frequency(&bf_x);
        free_block_frequency(&bf_y);
    }
    free(x);
    free(y);
}

static void test_generation_rate (void)
{
    int length, max_block_length;
    struct block_frequency bf_x, bf_y;
    int *x, *y;
    double gen_rate;

    length = 1024;
    max_block_length = 5;

    MALLOC(x, length);
    MALLOC(y, length);

    gen_Morse_sequence(x, length);
    for (int n = 0; n < length; n++) {
        y[n] = x[(n+300)%length];
    }
    for (int n = 1; n < max_block_length; n++) {
        init_block_frequency(&bf_x, x, length, n);
        init_block_frequency(&bf_y, y, length, n);
        gen_rate = generation_rate(&bf_x, &bf_y);
        assert_equal_double(1, gen_rate, 1e-3);
        free_block_frequency(&bf_x);
        free_block_frequency(&bf_y);
    }
    for (int n = 0; n < length; n++) {
        x[n] = 0;
        y[n] = 1;
    }
    for (int n = 1; n < max_block_length; n++) {
        init_block_frequency(&bf_x, x, length, n);
        init_block_frequency(&bf_y, y, length, n);
        gen_rate = generation_rate(&bf_x, &bf_y);
        assert_equal_double(0, gen_rate, 1e-3);
        free_block_frequency(&bf_x);
        free_block_frequency(&bf_y);
    }
    free(x);
    free(y);
}

void test_entropy (void)
{
    mu_run_test(test_block_entropy);
    mu_run_test(test_kullback_leibler_divergence);
    mu_run_test(test_generation_rate);
}


