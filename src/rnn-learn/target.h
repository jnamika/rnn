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

#ifndef TARGET_H
#define TARGET_H



typedef struct target_reader {
    int dimension;
    int num;
    struct target_t {
        int length;
        double **target;
    } *t_list;
} target_reader;

void init_target_reader (struct target_reader *t_reader);

int read_target_from_file (
        struct target_reader *t_reader,
        const char *separator,
        FILE *fp);

void free_target_reader (struct target_reader *t_reader);


#endif

