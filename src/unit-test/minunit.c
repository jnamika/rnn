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

#include <stdio.h>
#include <stdarg.h>
#include "minunit.h"

static unsigned int mu_test_count = 0;
static unsigned int mu_assertion_count = 0;
static unsigned int mu_failure_count = 0;


void mu_summary (void)
{
    printf("%u test(s), %u assertion(s), %u failure(s)\n", mu_test_count,
            mu_assertion_count, mu_failure_count);
    if (mu_assertion_count > 0) {
        printf("%u%% passed\n", (100 * (mu_assertion_count - mu_failure_count))
                / mu_assertion_count);
    }
}

void mu_reset_test (void)
{
    mu_test_count = 0;
    mu_assertion_count = 0;
    mu_failure_count = 0;
}




void _mu_test_preprocess (const char *funcname)
{
    printf("test: %s()\n", funcname); \
}

void _mu_test_postprocess (void)
{
    mu_test_count++;
    printf("\n");
}

void _mu_assert (
        int cond,
        const char *filename,
        int line,
        const char *fmt, ...)
{
    va_list argp;
    mu_assertion_count++;
    if (!cond) {
        printf("F");
        mu_failure_count++;
        va_start(argp, fmt);
        fprintf(stderr, "%u) Failure:%s:%d: ", mu_failure_count, filename,
                line);
        vfprintf(stderr, fmt, argp);
        fprintf(stderr, "\n");
        va_end(argp);
    } else {
        printf(".");
    }
}

