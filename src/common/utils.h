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

#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#ifndef print_error_msg
#define print_error_msg(...) do { \
    fprintf(stderr, "%s:%d:%s", __FILE__, __LINE__, __func__); \
    fprintf(stderr, ": "__VA_ARGS__); fprintf(stderr, "\n"); \
    if (errno != 0) { perror(NULL); }} while(0)
#endif


#ifndef MALLOC
#define MALLOC(x,n) do { \
    (x) = malloc(sizeof(*(x)) * (n)); \
    if ((n) != 0 && (x) == NULL) { \
        print_error_msg("`malloc' failed"); \
        exit(EXIT_FAILURE); \
    }} while(0)
#endif

#ifndef REALLOC
#define REALLOC(x,n) do { \
    (x) = realloc((x), sizeof(*(x)) * (n)); \
    if ((n) != 0 && (x) == NULL) { \
        print_error_msg("`realloc' failed"); \
        exit(EXIT_FAILURE); \
    }} while(0)
#endif


#ifndef FWRITE
#define FWRITE(x,n,fp) do { \
    if (fwrite((x), sizeof(*(x)), (n), (fp)) != (size_t)(n)) { \
        print_error_msg("`fwrite' failed"); \
        exit(EXIT_FAILURE); \
    }} while(0)
#endif

#ifndef FREAD
#define FREAD(x,n,fp) do { \
    if (fread((x), sizeof(*(x)), (n), (fp)) != (size_t)(n)) { \
        print_error_msg("`fread' failed"); \
        exit(EXIT_FAILURE); \
    }} while(0)
#endif


#include <stdint.h>

uint32_t xor128(void);
void init_xor128(uint32_t s);
void init_genrand(unsigned long s);
double genrand_real1(void);
double genrand_real2(void);
double genrand_real3(void);


#include <sys/types.h>

ssize_t getline (char **lineptr, size_t *n, FILE *fp);


#endif

