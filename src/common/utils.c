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
#include <stdint.h>
#include <sys/types.h>
#include <math.h>
#include <string.h>

#include "utils.h"



static uint32_t x = 123456789;
static uint32_t y = 362436069;
static uint32_t z = 521288629;
static uint32_t w = 88675123;

uint32_t xor128(void)
{
    uint32_t t;
    t = x ^ (x << 11);
    x = y; y = z; z = w;
    return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
}

void init_xor128(uint32_t s)
{
    x = s;
    y = 1812433253 * (x^(x>>30)) + 1;
    z = 1812433253 * (y^(y>>30)) + 2;
    w = 1812433253 * (z^(z>>30)) + 3;
}

void init_genrand(unsigned long s)
{
    init_xor128((uint32_t)(s & UINT32_MAX));
}

/* generates a random number on [0,1]-interval */
double genrand_real1(void)
{
    return xor128() * (1.0 / UINT32_MAX);
}

/* generates a random number on [0,1)-interval */
double genrand_real2(void)
{
    return xor128() * (1.0 / (UINT32_MAX + 1.0));
}

/* generates a random number on (0,1)-interval */
double genrand_real3(void)
{
    return (xor128() + 0.5) * (1.0 / (UINT32_MAX + 1.0));
}

#ifndef _GNU_SOURCE

/*
 * reads an entire line from stream, storing the address of the buffer
 * containing the text into *lineptr.
 */
ssize_t getline (
        char **lineptr,
        size_t *n,
        FILE *fp)
{
    if (*lineptr == NULL || *n < BUFSIZ) {
        *n = BUFSIZ;
        REALLOC(*lineptr, *n);
    }
    *lineptr[0] = '\0';
    char *buf = *lineptr;
    int is_eof = 0;
    size_t length = 0;
    for(;;) {
        is_eof = (fgets(buf, BUFSIZ, fp) == NULL);
        char *p = strchr(buf, '\n');
        if (p != NULL || is_eof) {
            length += strlen(buf);
            break;
        }
        length += BUFSIZ - 1;
        if (*n < length + BUFSIZ) {
            *n *= 2;
            REALLOC(*lineptr, *n);
        }
        buf = *lineptr + length;
    }
    return (*lineptr[0] == '\0' && is_eof) ? -1 : (ssize_t)length;
}

#endif // _GNU_SOURCE


