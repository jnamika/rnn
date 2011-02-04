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

#ifndef CONFIG_H
#define CONFIG_H

/*
 * This file is used in target source codes to test.
 * DO NOT include this file in testing source codes because
 * this overwrites some functions.
 */

#define MAX_ITERATION_IN_ADAPTIVE_LR 10000
//#define MAX_PERF_INC (1.0-1e-10)
#define MAX_PERF_INC (1.0+1e-10)


#undef print_error_msg
#define print_error_msg(...)


/*
 * The followings are defined in order to overwrite exit(3) and assert(3).
 * This code is evil!
 */
#include <setjmp.h>
extern jmp_buf _g_jbuf; // defined in main.c
#undef exit
#undef assert
#define exit(X) (longjmp(_g_jbuf, 1))
#define assert(X) do { if (!(X)) { longjmp(_g_jbuf, 1); }} while(0)

#endif

