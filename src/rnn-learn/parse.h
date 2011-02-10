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

#ifndef PARSE_H
#define PARSE_H

void str_to_connection (
        const char *str,
        int in_size,
        int out_size,
        int **has_connection,
        double **connectivity);

void str_to_const_init_c (
        const char *str,
        int c_state_size,
        int *const_init_c);

void str_to_softmax_group (
        const char *str,
        int out_state_size,
        int *softmax_group_num,
        int *softmax_group_id);

void str_to_init_tau (
        const char *str,
        int c_state_size,
        double *tau);

#endif

