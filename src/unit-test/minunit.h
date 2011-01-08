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

#ifndef MINUNIT_H
#define MINUNIT_H

/*
 * MinUnit: a minimal unit testing framework for C99
 * The original version is at http://www.jera.com/techinfo/jtns/jtn002.html
 */


/* public assert functions */
// Assertion `1 == 0' failed
#define mu_assert_I(cond) ___mu_assert(cond, __FILE__, __LINE__, \
        "assertion `" #cond "' failed")
#define mu_assert(cond) mu_assert_I(cond)
#define mu_assert_with_msg(cond,...) \
    ___mu_assert(cond, __FILE__, __LINE__, "" __VA_ARGS__)
#define mu_fail(...) ___mu_assert(0, __FILE__, __LINE__, "" __VA_ARGS__)



#define mu_run_test(func) do { \
    ___mu_test_preprocess(#func); \
    func(); \
    ___mu_test_postprocess();} while(0)

#define mu_run_test_with_args(func,...) do { \
    ___mu_test_preprocess(#func); \
    func(__VA_ARGS__); \
    ___mu_test_postprocess();} while(0)



void mu_summary(void);
void mu_reset_test(void);


/*
 * These functions are not officially "public".
 * They exist here because they need to be for proper operation of minunit.
 * Please use the aforementioned macros instead of them.
 */

void ___mu_assert (int cond, const char *filename, int line,
        const char *fmt, ...);
void ___mu_test_preprocess (const char* funcname);
void ___mu_test_postprocess (void);

#endif

