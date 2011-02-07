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
#include <string.h>
#include <math.h>

#include "minunit.h"
#include "my_assert.h"
#include "utils.h"
#include "target.h"


/* assert functions */


/* test functions */

static void test_read_target_from_file1 (void)
{
    struct target_reader t_reader;
    const char *separator = "\t";

    FILE *fp;
    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "hoge\n");
    fseek(fp, 0L, SEEK_SET);
    init_target_reader(&t_reader);
    mu_assert(read_target_from_file(&t_reader, separator, fp) == -1);
    free_target_reader(&t_reader);

    fclose(fp);
}

static void test_read_target_from_file2 (void)
{
    struct target_reader t_reader;
    const char *separator = "\t";

    FILE *fp;
    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }

    double **src = NULL;
    const int length = 50;
    const int dim = 10;
    MALLOC2(src, length, dim);

    fprintf(fp, "# comment\n");
    for (int n = 0; n < length; n++) {
        for (int i = 0; i < dim; i++) {
            src[n][i] = 0.1*(n+i);
            fprintf(fp, "%f%c", src[n][i], (i<dim-1)?separator[0]:'\n');
        }
    }

    fseek(fp, 0L, SEEK_SET);
    init_target_reader(&t_reader);
    mu_assert(read_target_from_file(&t_reader, separator, fp) != -1);
    assert_equal_int(dim, t_reader.dimension);
    assert_equal_int(1, t_reader.num);
    assert_equal_int(length, t_reader.t_list[0].length);
    for (int n = 0; n < length; n++) {
        for (int i = 0; i < dim; i++) {
            assert_equal_double(src[n][i], t_reader.t_list[0].target[n][i],
                    1e-6);
        }
    }
    free_target_reader(&t_reader);
    FREE2(src);

    fclose(fp);
}

static void test_read_target_from_file3 (void)
{
    struct target_reader t_reader;
    const char *separator = "\t";

    FILE *fp;
    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }

    for (int n = 0; n < 100; n++) {
        for (int i = 0; i < 5; i++) {
            if (n == 80 && i == 3) {
                fprintf(fp, "hogehoge");
            } else {
                fprintf(fp, "%f", 0.1*(n+i));
            }
            fprintf(fp, "%c", (i<9)?'\t':'\n');
        }
    }
    fseek(fp, 0L, SEEK_SET);
    init_target_reader(&t_reader);
    mu_assert(read_target_from_file(&t_reader, separator, fp) == -1);
    free_target_reader(&t_reader);

    fclose(fp);
}


static void test_read_target_from_file4 (void)
{
    struct target_reader t_reader;
    const char *separator = "\t";

    FILE *fp;
    fp = tmpfile();
    if (fp == NULL) {
        print_error_msg("cannot open tmpfile");
        exit(EXIT_FAILURE);
    }

    const int length[] = {50, 100, 75};
    const int dim = 8;
    for (int k = 0; k < 3; k++) {
        for (int n = 0; n < length[k]; n++) {
            for (int i = 0; i < dim; i++) {
                fprintf(fp, "%f%c", 0.01 * (n * k * i),
                        (i<dim-1)?separator[0]:'\n');
            }
        }
        fprintf(fp, "\n");
    }

    fseek(fp, 0L, SEEK_SET);
    init_target_reader(&t_reader);
    mu_assert(read_target_from_file(&t_reader, separator, fp) != -1);
    assert_equal_int(dim, t_reader.dimension);
    assert_equal_int(3, t_reader.num);
    for (int k = 0; k < 3; k++) {
        assert_equal_int(length[k], t_reader.t_list[k].length);
        for (int n = 0; n < length[k]; n++) {
            for (int i = 0; i < dim; i++) {
                assert_equal_double(0.01 * (n * k * i),
                        t_reader.t_list[k].target[n][i], 1e-6);
            }
        }
    }
    fseek(fp, 0L, SEEK_SET);
    mu_assert(read_target_from_file(&t_reader, separator, fp) != -1);
    assert_equal_int(dim, t_reader.dimension);
    assert_equal_int(6, t_reader.num);
    for (int k = 0; k < 6; k++) {
        assert_equal_int(length[k % 3], t_reader.t_list[k].length);
        for (int n = 0; n < length[k % 3]; n++) {
            for (int i = 0; i < dim; i++) {
                assert_equal_double(0.01 * (n * (k % 3) * i),
                        t_reader.t_list[k].target[n][i], 1e-6);
            }
        }
    }
    free_target_reader(&t_reader);

    fclose(fp);
}

static void test_read_target_from_file (void)
{
    test_read_target_from_file1();
    test_read_target_from_file2();
    test_read_target_from_file3();
    test_read_target_from_file4();
}


void test_target (void)
{
    mu_run_test(test_read_target_from_file);
}


