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

#ifndef MY_ASSERT_H
#define MY_ASSERT_H


#include <setjmp.h>
#include "minunit.h"


/* assert functions */

#define assert_equal_int(x,y) \
    mu_assert_with_msg((x)==(y), " expected: %d\n  but was: %d\n", \
            (int)(x), (int)(y))
#define assert_equal_double(x,y,p) \
    mu_assert_with_msg(((x)<=(y)+(p))&&((x)>=(y)-(p)), \
            " expected: %f\n  but was: %f\n where precision: %f\n", \
            (double)(x), (double)(y), (double)(p))
#define assert_equal_pointer(x,y) \
    mu_assert_with_msg((x)==(y), " expected: %p\n  but was: %p\n", \
            (const void*)(x), (const void*)(y))
#define assert_equal_string(x,y) \
    mu_assert_with_msg(!strcmp(x,y), " expected: %s\n  but was: %s\n", \
            (const char*)(x), (const char*)(y))
#define assert_equal_memory(x,x_size,y,y_size) \
    mu_assert_with_msg(((x_size)==(y_size))&&!memcmp(x,y,x_size), \
        "byte strings %s and %s differ\n", #x, #y)


#define assert_equal_vector_sequence(x,x_size,x_length,y,y_size,y_length) \
    do { \
        int iseq=1; \
        if(((x_size)!=(y_size))||((x_length)!=(y_length))){iseq=0;}\
        else{for(int n=0;n<(x_length);n++){\
            if(memcmp((x)[n],(y)[n],(x_size))){iseq=0;break;}}} \
        mu_assert_with_msg(iseq, \
                " vector sequences %s and %s differ\n", #x, #y); \
    } while(0)






extern jmp_buf ___g_jbuf; // defined in main.c

/* In the target source codes, exit() is overwrited as longjmp(___g_jbuf, 1)
 * (see config.h) */

#define assert_exit_call(func,...) \
    do { \
        volatile int is_exit_called = 0; jmp_buf tmp_jbuf; \
        memcpy(&tmp_jbuf, &___g_jbuf, sizeof(jmp_buf)); \
        if (setjmp(___g_jbuf) == 0) { func(__VA_ARGS__); \
        } else { is_exit_called = 1; } \
        memcpy(&___g_jbuf, &tmp_jbuf, sizeof(jmp_buf)); \
        mu_assert_with_msg(is_exit_called, " exit() was not called\n"); \
    } while(0)

#define assert_exit_nocall(func,...) \
    do { \
        volatile int is_exit_called = 1; jmp_buf tmp_jbuf; \
        memcpy(&tmp_jbuf, &___g_jbuf, sizeof(jmp_buf)); \
        if (setjmp(___g_jbuf) == 0) { func(__VA_ARGS__); \
        } else { is_exit_called = 0; } \
        memcpy(&___g_jbuf, &tmp_jbuf, sizeof(jmp_buf)); \
        mu_assert_with_msg(is_exit_called, " exit() was called\n"); \
    } while(0)


#endif

