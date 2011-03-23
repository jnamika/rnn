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
    int dimension, length, max_block_length;
    double entropy;
    struct block_frequency bf;
    int **sequence;

    dimension = 3;
    length = 10000;
    max_block_length = 5;

    MALLOC2(sequence, length, dimension);
    for (int n = 0; n < length; n++) {
        for (int i = 0; i < dimension; i++) {
            sequence[n][i] = 0;
        }
    }
    for (int n = 1; n <= max_block_length; n++) {
        init_block_frequency(&bf, (const int* const*)sequence, dimension,
                length, n);
        entropy = block_entropy(&bf) / n;
        assert_equal_double(0, entropy, 1e-3);
        free_block_frequency(&bf);
    }

    for (int n = 0; n < length; n++) {
        for (int i = 0; i < dimension; i++) {
            sequence[n][i] = (n + i) % 2;
        }
    }
    for (int n = 1; n <= max_block_length; n++) {
        init_block_frequency(&bf, (const int* const*)sequence, dimension,
                length, n);
        entropy = block_entropy(&bf) / n;
        assert_equal_double(1.0/n, entropy, 1e-3);
        free_block_frequency(&bf);
    }

    for (int n = 0; n < length; n++) {
        for (int i = 0; i < dimension; i++) {
            sequence[n][i] = (n + i) % 4;
        }
    }
    for (int n = 1; n <= max_block_length; n++) {
        init_block_frequency(&bf, (const int* const*)sequence, dimension,
                length, n);
        entropy = block_entropy(&bf) / n;
        assert_equal_double(2.0/n, entropy, 1e-3);
        free_block_frequency(&bf);
    }

    int *tmp;
    MALLOC(tmp, length);
    gen_Morse_sequence(tmp, length);
    for (int n = 0; n < length; n++) {
        for (int i = 0; i < dimension; i++) {
            sequence[n][i] = tmp[n];
        }
    }
    FREE(tmp);
    for (int n = 1; n <= max_block_length ; n++) {
        init_block_frequency(&bf, (const int* const*)sequence, dimension,
                length, n);
        entropy = block_entropy(&bf) / n;
        mu_assert(entropy >= 0.5);
        free_block_frequency(&bf);
    }
    FREE2(sequence);
}

static void test_kullback_leibler_divergence (void)
{
    int dimension, length, max_block_length;
    struct block_frequency bf_x, bf_y;
    int **x, **y;
    double kl_div;

    dimension = 2;
    length = 1024;
    max_block_length = 5;

    MALLOC2(x, length, dimension);
    MALLOC2(y, length, dimension);

    int *tmp;
    MALLOC(tmp, length);
    gen_Morse_sequence(tmp, length);
    for (int n = 0; n < length; n++) {
        for (int i = 0; i < dimension; i++) {
            x[n][i] = tmp[(n+i) % length];
            y[n][i] = tmp[(n+i+300) % length];
        }
    }
    FREE(tmp);
    for (int n = 1; n < max_block_length; n++) {
        init_block_frequency(&bf_x, (const int* const*)x, dimension, length, n);
        init_block_frequency(&bf_y, (const int* const*)y, dimension, length, n);
        kl_div = kullback_leibler_divergence(&bf_x, &bf_y);
        assert_equal_double(0, kl_div, 1e-3);
        free_block_frequency(&bf_x);
        free_block_frequency(&bf_y);
    }
    init_genrand(61107L);
    for (int i = 0; i < 10; i++) {
        for (int n = 0; n < length; n++) {
            for (int j = 0; j < dimension; j++) {
                x[n][j] = xor128() % 2;
                y[n][j] = xor128() % 2;
            }
        }
        for (int n = 1; n < max_block_length; n++) {
            init_block_frequency(&bf_x, (const int* const*)x, dimension, length,
                    n);
            init_block_frequency(&bf_y, (const int* const*)y, dimension, length,
                    n);
            kl_div = kullback_leibler_divergence(&bf_x, &bf_y);
            mu_assert(kl_div >= 0);
            free_block_frequency(&bf_x);
            free_block_frequency(&bf_y);

            init_block_frequency(&bf_x, (const int* const*)x, dimension,
                    length-100, n);
            init_block_frequency(&bf_y, (const int* const*)y, dimension, length,
                    n);
            kl_div = kullback_leibler_divergence(&bf_x, &bf_y);
            mu_assert(kl_div >= 0);
            free_block_frequency(&bf_x);
            free_block_frequency(&bf_y);

            init_block_frequency(&bf_x, (const int* const*)x, dimension, length,
                    n);
            init_block_frequency(&bf_y, (const int* const*)y, dimension,
                    length-100, n);
            kl_div = kullback_leibler_divergence(&bf_x, &bf_y);
            mu_assert(kl_div >= 0);
            free_block_frequency(&bf_x);
            free_block_frequency(&bf_y);
        }
    }

    for (int n = 0; n < length; n++) {
        for (int i = 0; i < dimension; i++) {
            x[n][i] = i;
            y[n][i] = i + 1;
        }
    }
    for (int n = 1; n < max_block_length; n++) {
        init_block_frequency(&bf_x, (const int* const*)x, dimension, length, n);
        init_block_frequency(&bf_y, (const int* const*)y, dimension, length, n);
        kl_div = kullback_leibler_divergence(&bf_x, &bf_y);
        mu_assert(kl_div > 0);
        free_block_frequency(&bf_x);
        free_block_frequency(&bf_y);
    }
    FREE2(x);
    FREE2(y);
}

static void test_generation_rate (void)
{
    int dimension, length, max_block_length;
    struct block_frequency bf_x, bf_y;
    int **x, **y;
    double gen_rate;

    dimension = 4;
    length = 1024;
    max_block_length = 5;

    MALLOC2(x, length, dimension);
    MALLOC2(y, length, dimension);

    int *tmp;
    MALLOC(tmp, length);
    gen_Morse_sequence(tmp, length);
    for (int n = 0; n < length; n++) {
        for (int i = 0; i < dimension; i++) {
            x[n][i] = tmp[(n+i) % length];
            y[n][i] = tmp[(n+i+300) % length];
        }
    }
    FREE(tmp);
    for (int n = 1; n < max_block_length; n++) {
        init_block_frequency(&bf_x, (const int* const*)x, dimension, length, n);
        init_block_frequency(&bf_y, (const int* const*)y, dimension, length, n);
        gen_rate = generation_rate(&bf_x, &bf_y);
        assert_equal_double(1, gen_rate, 1e-3);
        free_block_frequency(&bf_x);
        free_block_frequency(&bf_y);
    }
    for (int n = 0; n < length; n++) {
        for (int i = 0; i < dimension; i++) {
            x[n][i] = i;
            y[n][i] = i + 1;
        }
    }
    for (int n = 1; n < max_block_length; n++) {
        init_block_frequency(&bf_x, (const int* const*)x, dimension, length, n);
        init_block_frequency(&bf_y, (const int* const*)y, dimension, length, n);
        gen_rate = generation_rate(&bf_x, &bf_y);
        assert_equal_double(0, gen_rate, 1e-3);
        free_block_frequency(&bf_x);
        free_block_frequency(&bf_y);
    }
    FREE2(x);
    FREE2(y);
}

void test_entropy (void)
{
    mu_run_test(test_block_entropy);
    mu_run_test(test_kullback_leibler_divergence);
    mu_run_test(test_generation_rate);
}


