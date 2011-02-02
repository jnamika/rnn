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
#include "entropy.h"


#ifndef ADD_MEMORY_SIZE
#define ADD_MEMORY_SIZE 100
#endif

#ifndef NO_EXIST_FREQUENCY
#define NO_EXIST_FREQUENCY 0.001
#endif



void init_block_frequency (
        struct block_frequency *bf,
        const int *sequence,
        int length,
        int block_length)
{
    bf->size = 0;
    bf->mem_size = 0;
    bf->count = NULL;
    bf->index = NULL;
    bf->sequence = sequence;
    bf->length = length;
    bf->block_length = block_length;

    const int sum = length - block_length + 1;
    for (int n = 0; n < sum; n++) {
        int has_item = 0;
        for (int i = 0; i < bf->size; i++) {
            int is_equal = 1;
            for (int j = 0; j < block_length; j++) {
                if (sequence[n+j] != sequence[bf->index[i]+j]) {
                    is_equal = 0;
                    break;
                }
            }
            if (is_equal) {
                has_item = 1;
                bf->count[i]++;
                break;
            }
        }
        if (!has_item) {
            bf->size++;
            if (bf->size >= bf->mem_size) {
                bf->mem_size += ADD_MEMORY_SIZE;
                REALLOC(bf->count, bf->mem_size);
                REALLOC(bf->index, bf->mem_size);
            }
            bf->count[bf->size-1] = 1;
            bf->index[bf->size-1] = n;
        }
    }
}

void free_block_frequency (struct block_frequency *bf)
{
    free(bf->count);
    free(bf->index);
}



double block_entropy (const struct block_frequency *bf)
{
    const int len = bf->length - bf->block_length + 1;
    const double rate = 1.0 / (double)len;
    double block_entropy = 0;
    for (int n = 0; n < bf->size; n++) {
        double p = bf->count[n] * rate;
        block_entropy -= p * log(p);
    }
    return block_entropy/log(2);
}


double kullback_leibler_divergence (
        const struct block_frequency *bf_x,
        const struct block_frequency *bf_y)
{
    assert(bf_x->block_length == bf_y->block_length);

    const int len_x = bf_x->length - bf_x->block_length + 1;
    const int len_y = bf_y->length - bf_y->block_length + 1;
    const double r = (len_x == len_y) ? 0 : log(len_y/(double)len_x);
    const int *x = bf_x->sequence;
    const int *y = bf_y->sequence;
    double kl_div = 0;
    for (int m = 0; m < bf_x->size; m++) {
        double p = 1.0 * bf_x->count[m];
        double q = NO_EXIST_FREQUENCY;
        for (int n = 0; n < bf_y->size; n++) {
            int is_equal = 1;
            for (int i = 0; i < bf_x->block_length; i++) {
                if (x[bf_x->index[m]+i] != y[bf_y->index[n]+i]) {
                    is_equal = 0;
                    break;
                }
            }
            if (is_equal) {
                q = 1.0 * bf_y->count[n];
                break;
            }
        }
        kl_div += p * (log(p/q) + r);
    }
    for (int m = 0; m < bf_y->size; m++) {
        double q = 1.0 * bf_y->count[m];
        double p = NO_EXIST_FREQUENCY;
        int has_item = 0;
        for (int n = 0; n < bf_x->size; n++) {
            int is_equal = 1;
            for (int i = 0; i < bf_y->block_length; i++) {
                if (y[bf_y->index[m]+i] != x[bf_x->index[n]+i]) {
                    is_equal = 0;
                    break;
                }
            }
            if (is_equal) {
                has_item = 1;
                break;
            }
        }
        if (!has_item) {
            kl_div += p * (log(p/q) + r);
        }
    }
    kl_div /= (double)len_x;
    return kl_div;
}


double generation_rate (
        const struct block_frequency *bf_x,
        const struct block_frequency *bf_y)
{
    assert(bf_x->block_length == bf_y->block_length);

    const int *x = bf_x->sequence;
    const int *y = bf_y->sequence;
    int k, m;
    k = m = 0;
    while (m < bf_x->size) {
        int has_item = 0;
        for (int n = 0; n < bf_y->size; n++) {
            int is_equal = 1;
            for (int i = 0; i < bf_x->block_length; i++) {
                if (x[bf_x->index[m]+i] != y[bf_y->index[n]+i]) {
                    is_equal = 0;
                    break;
                }
            }
            if (is_equal) {
                has_item = 1;
                break;
            }
        }
        if (has_item) {
            k++;
        }
        m++;
    }
    return k/(double)m;
}


