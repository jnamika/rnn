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
#include <float.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "utils.h"
#include "parse.h"


static void str_to_intpair (
        char *str,
        int *begin,
        int *end,
        int min,
        int max)
{
    if (strlen(str) == 0) {
        *begin = *end = 0;
        return;
    }
    char *p;
    if ((p = strpbrk(str, "-")) != NULL) {
        *p = '\0';
        p++;
    }
    *begin = (strlen(str) > 0) ? atoi(str)-1 : min;
    *begin = (*begin < min) ? min : *begin;
    if (p != NULL) {
        *end = (strlen(p) > 0) ? atoi(p) : max;
    } else {
        *end = *begin + 1;
    }
    *end = (*end > max) ? max: *end;
}


void str_to_connection (
        const char *str,
        int in_size,
        int out_size,
        int **has_connection,
        double ** connectivity)
{
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++) {
            has_connection[i][j] = 0;
            connectivity[i][j] = 1.0;
        }
    }
    const int length = strlen(str);
    if (length > 0) {
        char *buf;
        MALLOC(buf, length + 1);
        strcpy(buf, str);
        char *p = strtok(buf, ",");
        while (p != NULL) {
            double c = 1.0;
            char *q;
            if ((q = strpbrk(p, ":")) != NULL) {
                *q = '\0';
                q++;
                c = atof(q);
            }
            if ((q = strpbrk(p, "t")) == NULL) {
                print_error_msg("warning: syntax error in `%s'", str);
                FREE(buf);
                goto error;
            }
            *q = '\0';
            q++;
            int in_begin, in_end, out_begin, out_end;
            str_to_intpair(p, &in_begin, &in_end, 0, in_size);
            str_to_intpair(q, &out_begin, &out_end, 0, out_size);
            for (int i = out_begin; i < out_end; i++) {
                for (int j = in_begin; j < in_end; j++) {
                    has_connection[i][j] = 1;
                    connectivity[i][j] = c;
                }
            }
            p = strtok(NULL, ",");
        }
        FREE(buf);
    }
    return;
error:
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++) {
            has_connection[i][j] = 1;
        }
    }
}


void str_to_const_init_c (
        const char *str,
        int c_state_size,
        int *const_init_c)
{
    for (int i = 0; i < c_state_size; i++) {
        const_init_c[i] = 0;
    }
    const int length = strlen(str);
    if (length > 0) {
        char *buf;
        MALLOC(buf, length + 1);
        strcpy(buf, str);
        char *p = strtok(buf, ",");
        while (p != NULL) {
            int begin, end;
            str_to_intpair(p, &begin, &end, 0, c_state_size);
            for (int i = begin; i < end; i++) {
                const_init_c[i] = 1;
            }
            p = strtok(NULL, ",");
        }
        FREE(buf);
    }
}



void str_to_softmax_group (
        const char *str,
        int out_state_size,
        int *softmax_group_num,
        int *softmax_group_id)
{
    for (int i = 0; i < out_state_size; i++) {
        softmax_group_id[i] = 0;
    }
    *softmax_group_num = 1;

    const int length = strlen(str);
    if (length > 0) {
        char *buf;
        MALLOC(buf, length + 1);
        strcpy(buf, str);
        char *p = strtok(buf, ",");
        for (int c = 1; p != NULL; c++) {
            do {
                char *q;
                if ((q = strpbrk(p, "&")) != NULL) {
                    *q = '\0';
                    q++;
                }
                int begin, end;
                str_to_intpair(p, &begin, &end, 0, out_state_size);
                for (int i = begin; i < end; i++) {
                    softmax_group_id[i] = c;
                }
                p = q;
            } while (p != NULL);
            p = strtok(NULL, ",");
        }
        int c = 0;
        for (int i = 0; i < out_state_size; i++) {
            if (softmax_group_id[i] > c) {
                int swap_c = softmax_group_id[i];
                for (int j = i; j < out_state_size; j++) {
                    if (softmax_group_id[j] == swap_c) {
                        softmax_group_id[j] = c;
                    } else if (softmax_group_id[j] == c) {
                        softmax_group_id[j] = swap_c;
                    }
                }
                c++;
            } else if (softmax_group_id[i] == c) {
                c++;
            }
        }
        *softmax_group_num = c;
        FREE(buf);
    }
}



void str_to_init_tau (
        const char *str,
        int c_state_size,
        double *tau)
{
    const int length = strlen(str);
    if (length > 0) {
        char *buf;
        MALLOC(buf, length + 1);
        strcpy(buf, str);
        char *p = strtok(buf, ",");
        while (p != NULL) {
            int begin, end;
            char *q;
            if ((q = strpbrk(p, ":")) != NULL) {
                *q = '\0';
                q++;
                str_to_intpair(q, &begin, &end, 0, c_state_size);
            } else {
                begin = 0;
                end = c_state_size;
            }
            double t = atof(p);
            if (t < 1.0) {
                t = 1.0;
            }
            for (int i = begin; i < end; i++) {
                tau[i] = t;
            }
            p = strtok(NULL, ",");
        }
        FREE(buf);
    }
}

