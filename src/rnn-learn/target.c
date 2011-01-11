/*
    Copyright (c) 2011, Jun Namikawa <jnamika@gmail.com>

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
#include <sys/types.h>
#include <math.h>
#include <string.h>
#include <errno.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "utils.h"
#include "target.h"


#ifndef INIT_MEMSIZE
#define INIT_MEMSIZE 1024
#endif


static int num_of_items_in_str (
        const char *str,
        const char *separator)
{
    char *tmp;
    MALLOC(tmp, strlen(str) + 1);
    strcpy(tmp, str);
    int n = 0;
    char *p = strtok(tmp, separator);
    while (p != NULL) {
        n++;
        p = strtok(NULL, separator);
    }
    free(tmp);
    return n;
}

static double* string_to_double (
        const char *str,
        double *x)
{
    char *endptr;
    *x = strtod(str, &endptr);
    errno = 0;
    if (endptr == str) {
        print_error_msg("no digits were found");
        return NULL;
    } else if ((errno == ERANGE && (*x == HUGE_VAL || *x == -HUGE_VAL)) ||
            (errno != 0 && *x == 0)) {
        print_error_msg();
        errno = 0;
        return NULL;
    }
    return x;
}

static int str2vector (
        char *str,
        const char *separator,
        double **vector)
{
    int num = num_of_items_in_str(str, separator);
    MALLOC(*vector, num);
    int n = 0;
    char *p = strtok(str, separator);
    while (p != NULL) {
        if (string_to_double(p, *vector + n) == NULL) {
            print_error_msg("error at column %d", n + 1);
            goto error;
        }
        n++;
        p = strtok(NULL, separator);
    }
    return num;
error:
    free(*vector);
    return -1;
}


void init_target_reader (struct target_reader *t_reader)
{
    t_reader->dimension = -1;
    t_reader->num = 0;
    t_reader->t_list = NULL;
}

void free_target_reader (struct target_reader *t_reader)
{
    for (int i = 0; i < t_reader->num; i++) {
        free(t_reader->t_list[i].target[0]);
        free(t_reader->t_list[i].target);
    }
    free(t_reader->t_list);
}

static void set_target (
        struct target_t *t,
        const double* const* vec_series,
        int length,
        int dimension)
{
    t->length = length;
    MALLOC(t->target, t->length);
    MALLOC(t->target[0], dimension * t->length);
    for (int n = 0; n < t->length; n++) {
        t->target[n] = t->target[0] + (n * dimension);
        memcpy(t->target[n], vec_series[n], sizeof(double) * dimension);
    }
}

int read_target_from_file (
        struct target_reader *t_reader,
        const char *separator,
        FILE *fp)
{
    int stat = 0;
    char *str = NULL;
    size_t str_size = 0;
    int line, length, max_length;
    int mem_length = INIT_MEMSIZE;
    double **vec_series = NULL;
    MALLOC(vec_series, mem_length);

    line = length = max_length = 0;
    while (getline(&str, &str_size, fp) != -1) {
        char *p = strchr(str, '\n');
        if (p != NULL) { *p = '\0'; }
        p = strchr(str, '#');
        if (p != NULL) { *p = '\0'; }
        line++;
        if (str[0] != '\0') {
            if (mem_length <= length) {
                mem_length *= 2;
                REALLOC(vec_series, mem_length);
            }
            int vec_size = str2vector(str, separator, vec_series + length);
            if (vec_size < 0) {
                print_error_msg("error at line %d", line);
                goto error;
            }
            length++;
            if (max_length < length) {
                max_length = length;
            }
            if (t_reader->dimension < 0) {
                t_reader->dimension = vec_size;
            } else if (t_reader->dimension != vec_size) {
                print_error_msg("wrong number of data items at line %d", line);
                goto error;
            }
        } else if (length > 0) {
            t_reader->num++;
            REALLOC(t_reader->t_list, t_reader->num);
            set_target(t_reader->t_list + t_reader->num - 1,
                    (const double* const*)vec_series, length,
                    t_reader->dimension);
            length = 0;
        }
    }
    if (length > 0) {
        t_reader->num++;
        REALLOC(t_reader->t_list, t_reader->num);
        set_target(t_reader->t_list + t_reader->num - 1,
                (const double* const*)vec_series, length, t_reader->dimension);
    }
    stat = 1;
error:
    free(str);
    for (int n = 0; n < max_length; n++) {
        free(vec_series[n]);
    }
    free(vec_series);
    return (stat ? 0 : -1);
}


