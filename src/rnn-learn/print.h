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

#ifndef PRINT_H
#define PRINT_H

#include "main.h"
#include "rnn.h"

typedef struct output_files {
    int array_size;
    FILE **fp_wstate_array;
    FILE **fp_wclosed_state_array;
    FILE *fp_wweight;
    FILE *fp_wthreshold;
    FILE *fp_wtau;
    FILE *fp_wsigma;
    FILE *fp_winit;
    FILE *fp_wadapt_lr;
    FILE *fp_werror;
    FILE *fp_wclosed_error;
    FILE *fp_wlyapunov;
    FILE *fp_wentropy;
} output_files;


void init_output_files (
        const struct general_parameters *gp,
        const struct recurrent_neural_network *rnn,
        struct output_files *fp_list,
        const char* mode);

void free_output_files (struct output_files *fp_list);


void print_training_main_begin (
        const struct general_parameters *gp,
        const struct recurrent_neural_network *rnn,
        struct output_files *fp_list);

void print_training_main_loop (
        long epoch,
        const struct general_parameters *gp,
        struct recurrent_neural_network *rnn,
        struct output_files *fp_list);

#endif

