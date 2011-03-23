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
#include <stddef.h>
#include <string.h>

#include "minunit.h"
#include "my_assert.h"
#include "utils.h"


/* assert functions */


/* test functions */


static void test_getline (void)
{
    int line;
    char *str1, *str2;
    size_t size;
    FILE *fp;

    MALLOC(str1, 10 * BUFSIZ);
    memset(str1, 'a', 10 * BUFSIZ);
    str1[10 * BUFSIZ - 1] = '\0';
    str2 = NULL;
    size = 0;

    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "%s", str1);
    fflush(fp);
    fseek(fp, 0L, SEEK_SET);
    line = 0;
    while (getline(&str2, &size, fp) != -1) {
        mu_assert(strcmp(str1, str2) == 0);
        line++;
    }
    assert_equal_int(1, line);
    fseek(fp, 0L, SEEK_SET);
    memset(str1, 'b', 10 * BUFSIZ);
    str1[10 * BUFSIZ - 1] = '\0';
    str1[10 * BUFSIZ - 2] = '\n';
    fprintf(fp, "%s%s", str1, str1);
    fflush(fp);
    fseek(fp, 0L, SEEK_SET);
    line = 0;
    while (getline(&str2, &size, fp) != -1) {
        mu_assert(strcmp(str1, str2) == 0);
        line++;
    }
    assert_equal_int(2, line);
    fclose(fp);

    FREE(str1);
    FREE(str2);
}


void test_utils (void)
{
    mu_run_test(test_getline);
}


