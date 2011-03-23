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
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <setjmp.h>
#ifdef ENABLE_MTRACE
#include <mcheck.h>
#endif

#include "minunit.h"
#include "test_utils.h"
#include "test_rnn.h"
#include "test_entropy.h"
#include "test_solver.h"
#include "test_rnn_lyapunov.h"
#include "test_target.h"
#include "test_parse.h"
#include "test_rnn_runner.h"
#include "utils.h"


static void my_shutdown (void)
{
    mu_summary();
}

int main (void)
{
#ifdef ENABLE_MTRACE
    mtrace();
#endif
    atexit(my_shutdown);
    opterr = 0;

    test_utils();
    test_rnn();
    test_entropy();
    test_solver();
    test_rnn_lyapunov();
    test_target();
    test_parse();
    test_rnn_runner();

#ifdef ENABLE_MTRACE
    muntrace();
#endif
    return EXIT_SUCCESS;
}

