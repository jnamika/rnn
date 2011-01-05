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
#include <stdarg.h>
#include <math.h>
#include <string.h>

#include "minunit.h"

static unsigned int mu_test_count = 0;
static unsigned int mu_assertion_count = 0;
static unsigned int mu_failure_count = 0;
static char str_buf[BUFSIZ];


void mu_summary (void)
{
    fprintf(stderr, "%u test(s), %u assertion(s), %u failure(s)\n",
            mu_test_count, mu_assertion_count, mu_failure_count);
    if (mu_assertion_count > 0) {
        fprintf(stderr, "%d%% passed\n", (int)floor(
                    (100 * (mu_assertion_count - mu_failure_count)) /
                    ((double)mu_assertion_count)));
    }
}

void mu_reset_test (void)
{
    mu_test_count = 0;
    mu_assertion_count = 0;
    mu_failure_count = 0;
}




void ___mu_preprocess_test (void)
{
    str_buf[0] = '\0';
}

void ___mu_postprocess_test (void)
{
    mu_test_count++;
    fprintf(stderr, "\n%s\n", str_buf);
}

static void print_failure_msg (
        const char* file,
        int line,
        const char *fmt,
        va_list argp)
{
    int len, add_len;
    static char str[BUFSIZ];

    snprintf(str, BUFSIZ, "%u) Failure: %s:%d\n", mu_failure_count, file, line);
    len = strlen(str);
    vsnprintf(str + len, BUFSIZ -len, fmt, argp);

    len = strlen(str_buf);
    add_len = strlen(str);
    if (BUFSIZ - len < add_len) {
        fprintf(stderr, "\n%s\n", str_buf);
        len = 0;
    }
    snprintf(str_buf + len, BUFSIZ - len, "%s", str);
}

void ___mu_assert (
        int cond,
        const char *filename,
        int line,
        const char *fmt, ...)
{
    va_list argp;

    mu_assertion_count++;
    if (!cond) {
        fprintf(stderr, "F");
        mu_failure_count++;
        va_start(argp, fmt);
        print_failure_msg(filename, line, fmt, argp);
        va_end(argp);
    } else {
        fprintf(stderr, ".");
    }
}

