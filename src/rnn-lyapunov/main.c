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
#include <unistd.h>
#include <time.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef ENABLE_MTRACE
#include <mcheck.h>
#endif
#include "parameter.h"
#include "utils.h"
#include "main.h"
#include "rnn_runner.h"
#include "lyapunov.h"


#define TO_STRING_I(s) #s
#define TO_STRING(s) TO_STRING_I(s)

static void display_help (void)
{
    puts("rnn-lyapunov  - a program to analyze the Lyapunov exponents of "
            "recurrent neural networks");
    puts("");
    puts("Usage: rnn-lyapunov [-s seed] [-n length] [-c samples] "
            "[-m memory-size] [-t truncate-length] "
            "[-l number-of-Lyapunov-exponents] [-d noise-deviation] "
            "rnn-file");
    puts("Usage: rnn-lyapunov [-v] [-h]");
    puts("");
    puts("Available options are:");
    puts("-s seed");
    puts("    `seed' is the seed for the initialization of random number "
            "generator, which specifies a starting point for the random number "
            "sequence, and provides for restarting at the same point. If this "
            "option is omitted, the current system time is used.");
    puts("-n length");
    puts("    Iterations of the dynamics. Default is " TO_STRING(LENGTH) ".");
    puts("-c samples");
    puts("    Number of samples. Default is " TO_STRING(SAMPLE_NUM) ".");
    puts("-m memory-size");
    puts("    Size of memory for recording time series. Default is "
            TO_STRING(MEM_SIZE) ".");
    puts("-t truncate-length");
    puts("    `truncate-length' is the time steps before evaluating Lyapunov "
            "exponents in order to discard transient effects. Default is "
            TO_STRING(TRUNCATE_LENGTH) ".");
    puts("-l number-of-Lyapunov-exponents");
    puts("    Number of Lyapunov exponents which are evaluated. If this option "
            "is set with a negative value, then the whole spectrum of Lyapunov "
            "exponents is computed. Default is "
            TO_STRING(LYAPUNOV_SPECTRUM_SIZE) ".");
    puts("-d noise-deviation");
    puts("    Standard deviation of the normal distribution for adding noise "
            "to dynamics. Default is " TO_STRING(NOISE_DEVIATION) ".");
    puts("");
    puts("Program execution:");
    puts("rnn-lyapunov computes Lyapunov exponents of a recurrent neural "
            "network given by the rnn-file (ex: rnn.dat) which is generated "
            "by rnn-learn.");
}

static void display_version (void)
{
    printf("rnn-lyapunov version %s\n", TO_STRING(VERSION));
}

static void init_parameters (struct analysis_parameters *ap)
{
    // 0 < seed < 4294967296
    ap->seed = (((unsigned long)(time(NULL) * getpid())) % 4294967295) + 1;
    ap->length = LENGTH;
    ap->sample_num = SAMPLE_NUM;
    ap->mem_size = MEM_SIZE;
    ap->truncate_length = TRUNCATE_LENGTH;
    ap->lyapunov_spectrum_size = LYAPUNOV_SPECTRUM_SIZE;
    ap->noise_deviation = NOISE_DEVIATION;
}

static void read_options (
        int argc,
        char *argv[],
        struct analysis_parameters *ap)
{
    int opt;
    while ((opt = getopt(argc, argv, "s:n:c:m:t:l:d:vh")) != -1) {
        switch (opt) {
            case 's':
                ap->seed = strtoul(optarg, NULL, 0);
                break;
            case 'n':
                ap->length = atol(optarg);
                break;
            case 'c':
                ap->sample_num = atoi(optarg);
                break;
            case 'm':
                ap->mem_size = atoi(optarg);
                break;
            case 't':
                ap->truncate_length = atol(optarg);
                break;
            case 'l':
                ap->lyapunov_spectrum_size = atoi(optarg);
                break;
            case 'd':
                ap->noise_deviation = atof(optarg);
                break;
            case 'v':
                display_version();
                exit(EXIT_SUCCESS);
            case 'h':
                display_help();
                exit(EXIT_SUCCESS);
            default: /* '?' */
                fprintf(stderr, "Try `rnn-lyapunov -h' for more "
                        "information.\n");
                exit(EXIT_SUCCESS);
        }
    }
}

static void check_parameters (const struct analysis_parameters *ap)
{
    if (ap->seed <= 0) {
        print_error_msg("seed for random number generator not in valid "
                "range: x >= 1 (integer)");
        exit(EXIT_FAILURE);
    }
    if (ap->mem_size <= 0) {
        print_error_msg("`memory size' not in valid range: x > 0 (integer)");
        exit(EXIT_FAILURE);
    }
    if (ap->truncate_length < 0) {
        print_error_msg("`truncate length' not in valid range: "
                "x >= 0 (integer)");
        exit(EXIT_FAILURE);
    }
    if (ap->noise_deviation < 0) {
        print_error_msg("`noise deviation' not in valid range: "
                "x >= 0 (float)");
        exit(EXIT_FAILURE);
    }
}

int main (int argc, char *argv[])
{
#ifdef ENABLE_MTRACE
    mtrace();
#endif
    struct analysis_parameters ap;
    init_parameters(&ap);

    read_options(argc, argv, &ap);

    check_parameters(&ap);

    if (optind >= argc) {
        print_error_msg("Usage: rnn-lyapunov [option] rnn-file\n"
                "Try `rnn-lyapunov -h' for more information.");
        exit(EXIT_SUCCESS);
    }

    init_genrand(ap.seed);

    struct rnn_runner runner;
    FILE *fp;
    if ((fp = fopen(argv[optind], "r")) == NULL) {
        print_error_msg("cannot open %s", argv[optind]);
        exit(EXIT_FAILURE);
    }
    init_rnn_runner(&runner, fp);
    fclose(fp);

    compute_lyapunov_main(&ap, &runner);

    free_rnn_runner(&runner);

#ifdef ENABLE_MTRACE
    muntrace();
#endif
    return EXIT_SUCCESS;
}

