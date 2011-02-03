/*
    Copyright (c) 2010-2011, Jun Namikawa <jnamika@gmail.com>

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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "utils.h"
#include "print.h"
#include "entropy.h"
#include "rnn_lyapunov.h"



static void fopen_array (
        FILE **fp_array,
        int size,
        const char *template_filename,
        const char *mode)
{
    char str[7], *filename, *p;
    int length = strlen(template_filename);
    MALLOC(filename, length + 1);
    strcpy(filename, template_filename);
    p = strstr(filename, "XXXXXX");
    if (p == NULL) {
        REALLOC(filename, length + 8);
        filename[length] = '.';
        filename[length + 7] = '\0';
        p = filename + length + 1;
    }
    for (int i = 0; i < size; i++) {
        snprintf(str, sizeof(str), "%.6d", i);
        memmove(p, str, 6);
        fp_array[i] = fopen(filename, mode);
        if (fp_array[i] == NULL) {
            print_error_msg();
            goto error;
        }
    }
    free(filename);
    return;
error:
    exit(EXIT_FAILURE);
}


void init_output_files (
        const struct general_parameters *gp,
        const struct recurrent_neural_network *rnn,
        struct output_files *fp_list,
        const char *mode)
{
    fp_list->array_size = rnn->series_num;
    if (strlen(gp->iop.state_filename) > 0) {
        MALLOC(fp_list->fp_wstate_array, fp_list->array_size);
        fopen_array(fp_list->fp_wstate_array, fp_list->array_size,
                gp->iop.state_filename, mode);
    } else {
        fp_list->fp_wstate_array = NULL;
    }

    if (strlen(gp->iop.closed_state_filename) > 0) {
        MALLOC(fp_list->fp_wclosed_state_array, fp_list->array_size);
        fopen_array(fp_list->fp_wclosed_state_array, fp_list->array_size,
                gp->iop.closed_state_filename, mode);
    } else {
        fp_list->fp_wclosed_state_array = NULL;
    }

    if (strlen(gp->iop.weight_filename) > 0) {
        fp_list->fp_wweight = fopen(gp->iop.weight_filename, mode);
        if (fp_list->fp_wweight == NULL) goto error;
    } else {
        fp_list->fp_wweight = NULL;
    }
    if (strlen(gp->iop.threshold_filename) > 0) {
        fp_list->fp_wthreshold = fopen(gp->iop.threshold_filename, mode);
        if (fp_list->fp_wthreshold == NULL) goto error;
    } else {
        fp_list->fp_wthreshold = NULL;
    }
    if (strlen(gp->iop.tau_filename) > 0) {
        fp_list->fp_wtau = fopen(gp->iop.tau_filename, mode);
        if (fp_list->fp_wtau == NULL) goto error;
    } else {
        fp_list->fp_wtau = NULL;
    }
    if (strlen(gp->iop.sigma_filename) > 0) {
        fp_list->fp_wsigma = fopen(gp->iop.sigma_filename, mode);
        if (fp_list->fp_wsigma == NULL) goto error;
    } else {
        fp_list->fp_wsigma = NULL;
    }
    if (strlen(gp->iop.init_filename) > 0) {
        fp_list->fp_winit = fopen(gp->iop.init_filename, mode);
        if (fp_list->fp_winit == NULL) goto error;
    } else {
        fp_list->fp_winit = NULL;
    }
    if (strlen(gp->iop.adapt_lr_filename) > 0 && gp->mp.use_adaptive_lr) {
        fp_list->fp_wadapt_lr = fopen(gp->iop.adapt_lr_filename, mode);
        if (fp_list->fp_wadapt_lr == NULL) goto error;
    } else {
        fp_list->fp_wadapt_lr = NULL;
    }
    if (strlen(gp->iop.error_filename) > 0) {
        fp_list->fp_werror = fopen(gp->iop.error_filename, mode);
        if (fp_list->fp_werror == NULL) goto error;
    } else {
        fp_list->fp_werror = NULL;
    }
    if (strlen(gp->iop.closed_error_filename) > 0) {
        fp_list->fp_wclosed_error = fopen(gp->iop.closed_error_filename, mode);
        if (fp_list->fp_wclosed_error == NULL) goto error;
    } else {
        fp_list->fp_wclosed_error = NULL;
    }
    if (strlen(gp->iop.lyapunov_filename) > 0) {
        fp_list->fp_wlyapunov = fopen(gp->iop.lyapunov_filename, mode);
        if (fp_list->fp_wlyapunov == NULL) goto error;
    } else {
        fp_list->fp_wlyapunov = NULL;
    }
    if (strlen(gp->iop.entropy_filename) > 0) {
        fp_list->fp_wentropy = fopen(gp->iop.entropy_filename, mode);
        if (fp_list->fp_wentropy == NULL) goto error;
    } else {
        fp_list->fp_wentropy = NULL;
    }
    if (strlen(gp->iop.period_filename) > 0) {
        fp_list->fp_wperiod = fopen(gp->iop.period_filename, mode);
        if (fp_list->fp_wperiod == NULL) goto error;
    } else {
        fp_list->fp_wperiod = NULL;
    }
    return;
error:
    print_error_msg();
    exit(EXIT_FAILURE);
}

void free_output_files (struct output_files *fp_list)
{
    if (fp_list->fp_wstate_array) {
        for (int i = 0; i < fp_list->array_size; i++) {
            fclose(fp_list->fp_wstate_array[i]);
        }
        free(fp_list->fp_wstate_array);
    }
    if (fp_list->fp_wclosed_state_array) {
        for (int i = 0; i < fp_list->array_size; i++) {
            fclose(fp_list->fp_wclosed_state_array[i]);
        }
        free(fp_list->fp_wclosed_state_array);
    }
    if (fp_list->fp_wweight) {
        fclose(fp_list->fp_wweight);
    }
    if (fp_list->fp_wthreshold) {
        fclose(fp_list->fp_wthreshold);
    }
    if (fp_list->fp_wtau) {
        fclose(fp_list->fp_wtau);
    }
    if (fp_list->fp_wsigma) {
        fclose(fp_list->fp_wsigma);
    }
    if (fp_list->fp_winit) {
        fclose(fp_list->fp_winit);
    }
    if (fp_list->fp_wadapt_lr) {
        fclose(fp_list->fp_wadapt_lr);
    }
    if (fp_list->fp_werror) {
        fclose(fp_list->fp_werror);
    }
    if (fp_list->fp_wclosed_error) {
        fclose(fp_list->fp_wclosed_error);
    }
    if (fp_list->fp_wlyapunov) {
        fclose(fp_list->fp_wlyapunov);
    }
    if (fp_list->fp_wentropy) {
        fclose(fp_list->fp_wentropy);
    }
    if (fp_list->fp_wperiod) {
        fclose(fp_list->fp_wperiod);
    }
}

static void print_general_parameters (
        FILE *fp,
        const struct general_parameters *gp)
{
    fprintf(fp, "# seed = %lu\n", gp->mp.seed);
    if (gp->mp.use_adaptive_lr) {
        fprintf(fp, "# use_adaptive_lr\n");
    }
    fprintf(fp, "# rho = %f\n", gp->mp.rho);
    fprintf(fp, "# momentum = %f\n", gp->mp.momentum);
    fprintf(fp, "# delay_length = %d\n", gp->mp.delay_length);
    fprintf(fp, "# lambda = %f\n", gp->mp.lambda);
    fprintf(fp, "# alpha = %f\n", gp->mp.alpha);

    fprintf(fp, "# truncate_length = %d\n", gp->ap.truncate_length);
    fprintf(fp, "# block_length = %d\n", gp->ap.block_length);
    fprintf(fp, "# divide_num = %d\n", gp->ap.divide_num);
    fprintf(fp, "# lyapunov_spectrum_size = %d\n",
            gp->ap.lyapunov_spectrum_size);
    fprintf(fp, "# threshold_period = %g\n", gp->ap.threshold_period);
}



static void print_rnn_parameters (
        FILE *fp,
        const struct recurrent_neural_network *rnn)
{
    fprintf(fp, "# in_state_size = %d\n", rnn->rnn_p.in_state_size);
    fprintf(fp, "# c_state_size = %d\n", rnn->rnn_p.c_state_size);
    fprintf(fp, "# out_state_size = %d\n", rnn->rnn_p.out_state_size);
    if (rnn->rnn_p.output_type == STANDARD_TYPE) {
        fprintf(fp, "# output_type = STANDARD_TYPE\n");
    } else if (rnn->rnn_p.output_type == SOFTMAX_TYPE) {
        fprintf(fp, "# output_type = SOFTMAX_TYPE\n");
        for (int c = 0; c < rnn->rnn_p.softmax_group_num; c++) {
            fprintf(fp, "# group%d = ", c);
            for (int i = 0; i < rnn->rnn_p.out_state_size; i++) {
                if (rnn->rnn_p.softmax_group_id[i] == c) {
                    fprintf(fp, "%d,", i);
                }
            }
            fprintf(fp, "\n");
        }
    }
    if (rnn->rnn_p.fixed_weight) {
        fprintf(fp, "# fixed_weight\n");
    }
    if (rnn->rnn_p.fixed_threshold) {
        fprintf(fp, "# fixed_threshold\n");
    }
    if (rnn->rnn_p.fixed_tau) {
        fprintf(fp, "# fixed_tau\n");
    }
    if (rnn->rnn_p.fixed_init_c_state) {
        fprintf(fp, "# fixed_init_c_state\n");
    }
    if (rnn->rnn_p.fixed_sigma) {
        fprintf(fp, "# fixed_sigma\n");
    }

    fprintf(fp, "# target_num = %d\n", rnn->series_num);
    for (int i = 0; i < rnn->series_num; i++) {
        fprintf(fp, "# target %d\tlength = %d\n", i, rnn->rnn_s[i].length);
    }
    fprintf(fp, "# prior_strength = %f\n", rnn->rnn_p.prior_strength);

    const struct rnn_parameters *rnn_p = &rnn->rnn_p;
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        fprintf(fp, "# const_init_c[%d] = %d\n", i, rnn_p->const_init_c[i]);
    }

    for (int i = 0; i < rnn_p->c_state_size; i++) {
        fprintf(fp, "# connection_weight_ci[%d] = ", i);
        int I = 0;
        while (rnn_p->connection_ci[i][I].begin != -1) {
            int begin = rnn_p->connection_ci[i][I].begin;
            int end = rnn_p->connection_ci[i][I].end;
            fprintf(fp, "(%d,%d)", begin, end);
            I++;
        }
        fprintf(fp, "\n");
    }
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        fprintf(fp, "# connection_weight_cc[%d] = ", i);
        int I = 0;
        while (rnn_p->connection_cc[i][I].begin != -1) {
            int begin = rnn_p->connection_cc[i][I].begin;
            int end = rnn_p->connection_cc[i][I].end;
            fprintf(fp, "(%d,%d)", begin, end);
            I++;
        }
        fprintf(fp, "\n");
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        fprintf(fp, "# connection_weight_oc[%d] = ", i);
        int I = 0;
        while (rnn_p->connection_oc[i][I].begin != -1) {
            int begin = rnn_p->connection_oc[i][I].begin;
            int end = rnn_p->connection_oc[i][I].end;
            fprintf(fp, "(%d,%d)", begin, end);
            I++;
        }
        fprintf(fp, "\n");
    }
}


static void print_rnn_weight (
        FILE *fp,
        long epoch,
        const struct rnn_parameters *rnn_p)
{
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        for (int j = 0; j < rnn_p->in_state_size; j++) {
            fprintf(fp, "\t%f", rnn_p->weight_ci[i][j]);
        }
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            fprintf(fp, "\t%f", rnn_p->weight_cc[i][j]);
        }
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        for (int j = 0; j < rnn_p->c_state_size; j++) {
            fprintf(fp, "\t%f", rnn_p->weight_oc[i][j]);
        }
    }
    fprintf(fp, "\n");
}


static void print_rnn_threshold (
        FILE *fp,
        long epoch,
        const struct rnn_parameters *rnn_p)
{
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        fprintf(fp, "\t%f", rnn_p->threshold_c[i]);
    }
    for (int i = 0; i < rnn_p->out_state_size; i++) {
        fprintf(fp, "\t%f", rnn_p->threshold_o[i]);
    }
    fprintf(fp, "\n");
}


static void print_rnn_tau (
        FILE *fp,
        long epoch,
        const struct rnn_parameters *rnn_p)
{
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < rnn_p->c_state_size; i++) {
        fprintf(fp, "\t%g", rnn_p->tau[i]);
    }
    fprintf(fp, "\n");
}

static void print_rnn_sigma (
        FILE *fp,
        long epoch,
        const struct rnn_parameters *rnn_p)
{
    fprintf(fp, "%ld\t%f\t%f\n", epoch, rnn_p->sigma, rnn_p->variance);
}

static void print_rnn_init (
        FILE *fp,
        long epoch,
        const struct recurrent_neural_network *rnn)
{
    fprintf(fp, "# epoch = %ld\n", epoch);
    for (int i = 0; i < rnn->series_num; i++) {
        fprintf(fp, "%d", i);
        for (int j = 0; j < rnn->rnn_p.c_state_size; j++) {
            fprintf(fp, "\t%f", rnn->rnn_s[i].init_c_inter_state[j]);
        }
        fprintf(fp, "\n");
    }
}

static void print_adapt_lr (
        FILE *fp,
        long epoch,
        double adapt_lr)
{
    fprintf(fp, "%ld\t%f\n", epoch, adapt_lr);
}


static void print_rnn_error (
        FILE *fp,
        long epoch,
        const struct recurrent_neural_network *rnn)
{
    double error[rnn->series_num];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rnn->series_num; i++) {
        error[i] = rnn_get_error(rnn->rnn_s + i);
        error[i] /= rnn->rnn_s[i].length * rnn->rnn_p.out_state_size;
    }
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < rnn->series_num; i++) {
        fprintf(fp, "\t%g", error[i]);
    }
    fprintf(fp, "\n");
}


static void print_rnn_state (
        FILE *fp,
        const struct rnn_state *rnn_s)
{
    for (int n = 0; n < rnn_s->length; n++) {
        fprintf(fp, "%d", n);
        for (int i = 0; i < rnn_s->rnn_p->out_state_size; i++) {
            fprintf(fp, "\t%f", rnn_s->teach_state[n][i]);
            fprintf(fp, "\t%f", rnn_s->out_state[n][i]);
        }
        for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++) {
            //fprintf(fp, "\t%f", rnn_s->c_state[n][i]);
            fprintf(fp, "\t%f", rnn_s->c_inter_state[n][i]);
        }
        fprintf(fp, "\n");
    }
}


static void print_rnn_state_forall (
        FILE **fp_array,
        long epoch,
        const struct recurrent_neural_network *rnn)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rnn->series_num; i++) {
        fprintf(fp_array[i], "# epoch = %ld\n", epoch);
        fprintf(fp_array[i], "# target:%d\n", i);
        print_rnn_state(fp_array[i], rnn->rnn_s + i);
    }
}




static void compute_lyapunov_spectrum_of_rnn_state (
        const struct rnn_state *rnn_s,
        int spectrum_size,
        int delay_length,
        int truncate_length,
        double *spectrum)
{
    if (rnn_s->length > truncate_length) {
        struct rnn_lyapunov_info rl_info;
        init_rnn_lyapunov_info(&rl_info, rnn_s, delay_length, truncate_length);
        spectrum = rnn_lyapunov_spectrum(&rl_info, spectrum, spectrum_size);
        if (spectrum == NULL) {
            print_error_msg();
            exit(EXIT_FAILURE);
        }
        free_rnn_lyapunov_info(&rl_info);
    } else {
        for (int i = 0; i < spectrum_size; i++) {
            spectrum[i] = 0;
        }
    }
}


static void print_lyapunov_spectrum_of_rnn (
        FILE *fp,
        long epoch,
        const struct recurrent_neural_network *rnn,
        int spectrum_size,
        int delay_length,
        int truncate_length)
{
    int max_num;
    // decides spectrum_size which is the number to evaluate Lyapunov exponents
    max_num = (rnn->rnn_p.in_state_size * delay_length) +
        rnn->rnn_p.c_state_size;
    if (max_num < spectrum_size || spectrum_size < 0) {
        spectrum_size = max_num;
    }

    if (spectrum_size <= 0) return;

    double **spectrum = NULL;
    MALLOC(spectrum, rnn->series_num);
    MALLOC(spectrum[0], rnn->series_num * spectrum_size);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rnn->series_num; i++) {
        spectrum[i] = spectrum[0] + i * spectrum_size;
        compute_lyapunov_spectrum_of_rnn_state(rnn->rnn_s + i, spectrum_size,
                delay_length, truncate_length, spectrum[i]);
    }
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < rnn->series_num; i++) {
        for (int j = 0; j < spectrum_size; j++) {
            fprintf(fp, "\t%f", spectrum[i][j]);
        }
    }
    fprintf(fp, "\n");
    free(spectrum[0]);
    free(spectrum);
}


/* assigns an index to the vector with respect to indexed hypercubes in the
 * R^dimension space */
static int vector2symbol (
        const double *vector,
        int dimension,
        double min,
        double max,
        int divide_num)
{
    int e, symbol;
    double mesh_size, split;

    mesh_size = (max - min)/divide_num;

    symbol = 0;
    e = 1;
    for (int i = 0; i < dimension; i++) {
        split = min;
        for (int j = 0; j < divide_num; j++) {
            split += mesh_size;
            if (vector[i] <= split || j == divide_num-1) {
                symbol += e * j;
                break;
            }
        }
        e *= divide_num;
    }
    return symbol;
}


static void compute_kl_divergence_of_rnn_compression_state (
        const struct rnn_state *rnn_s,
        int truncate_length,
        int block_length,
        int divide_num,
        double *kl_div,
        double *entropy_t,
        double *entropy_o,
        double *gen_rate)
{
    if (rnn_s->length > truncate_length) {
        double min, max;
        int *sequence_t, *sequence_o;
        struct block_frequency bf_t, bf_o;
        const int length = rnn_s->length - truncate_length;
        MALLOC(sequence_t, length);
        MALLOC(sequence_o, length);
        if (rnn_s->rnn_p->output_type == STANDARD_TYPE) {
            min = -1.0; max = 1.0;
        } else {
            min = 0.0; max = 1.0;
        }
        for (int n = 0; n < length; n++) {
            int N = n + truncate_length;
            sequence_t[n] = vector2symbol(rnn_s->teach_state[N],
                    rnn_s->rnn_p->out_state_size, min, max, divide_num);
            sequence_o[n] = vector2symbol(rnn_s->out_state[N],
                    rnn_s->rnn_p->out_state_size, min, max, divide_num);
        }
        init_block_frequency(&bf_t, sequence_t, length, block_length);
        init_block_frequency(&bf_o, sequence_o, length, block_length);
        *kl_div = kullback_leibler_divergence(&bf_t, &bf_o);
        *entropy_t = block_entropy(&bf_t) / block_length;
        *entropy_o = block_entropy(&bf_o) / block_length;
        *gen_rate = generation_rate(&bf_t, &bf_o);
        free_block_frequency(&bf_t);
        free_block_frequency(&bf_o);
        free(sequence_t);
        free(sequence_o);
    } else {
        *kl_div = 0;
        *entropy_t = 0;
        *entropy_o = 0;
        *gen_rate = 0;
    }
}


static void print_kl_divergence_of_rnn (
        FILE *fp,
        long epoch,
        const struct recurrent_neural_network *rnn,
        int truncate_length,
        int block_length,
        int divide_num)
{
    double kl_div[rnn->series_num];
    double entropy_t[rnn->series_num];
    double entropy_o[rnn->series_num];
    double gen_rate[rnn->series_num];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rnn->series_num; i++) {
        compute_kl_divergence_of_rnn_compression_state(rnn->rnn_s + i,
                truncate_length, block_length, divide_num, kl_div + i,
                entropy_t + i, entropy_o + i, gen_rate + i);
    }
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < rnn->series_num; i++) {
        fprintf(fp, "\t%g\t%g\t%g\t%g", kl_div[i], gen_rate[i], entropy_t[i],
                entropy_o[i]);
    }
    fprintf(fp, "\n");
}



static int get_period_of_rnn_state (
        const struct rnn_state *rnn_s,
        double threshold)
{
    int period = 1;
    for (int n = rnn_s->length - 2; n >= 0; n--, period++) {
        double d = 0;
        for (int i = 0; i < rnn_s->rnn_p->c_state_size; i++) {
            double x = rnn_s->c_state[rnn_s->length-1][i] -
                rnn_s->c_state[n][i];
            d += x * x;
        }
        for (int i = 0; i < rnn_s->rnn_p->out_state_size; i++) {
            double x = rnn_s->out_state[rnn_s->length-1][i] -
                rnn_s->out_state[n][i];
            d += x * x;
        }
        if (d <= threshold) {
            break;
        }
    }
    return period;
}

static void print_period_of_rnn (
        FILE *fp,
        long epoch,
        const struct recurrent_neural_network *rnn,
        double threshold)
{
    int period[rnn->series_num];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rnn->series_num; i++) {
        period[i] = get_period_of_rnn_state(rnn->rnn_s + i, threshold);
    }
    fprintf(fp, "%ld", epoch);
    for (int i = 0; i < rnn->series_num; i++) {
        fprintf(fp, "\t%d", period[i]);
    }
    fprintf(fp, "\n");
}

static int enable_print (
        long epoch,
        const struct print_interval *pi)
{
    long interval;
    if (pi->use_logscale_interval) {
        interval = 1;
        while (epoch >= 10 * interval) {
            interval *= 10;
        }
        if (interval > pi->interval) {
            interval = pi->interval;
        }
    } else {
        interval = pi->interval;
    }
    return ((epoch % interval) == 0 && epoch >= pi->init && epoch <= pi->end);
}


static void print_parameters_with_epoch (
        long epoch,
        const struct general_parameters *gp,
        const struct recurrent_neural_network *rnn,
        struct output_files *fp_list)
{
    if (fp_list->fp_wweight &&
            enable_print(epoch, &gp->iop.interval_for_weight_file)) {
        print_rnn_weight(fp_list->fp_wweight, epoch, &rnn->rnn_p);
    }

    if (fp_list->fp_wthreshold &&
            enable_print(epoch, &gp->iop.interval_for_threshold_file)) {
        print_rnn_threshold(fp_list->fp_wthreshold, epoch, &rnn->rnn_p);
    }

    if (fp_list->fp_wtau &&
            enable_print(epoch, &gp->iop.interval_for_tau_file)) {
        print_rnn_tau(fp_list->fp_wtau, epoch, &rnn->rnn_p);
    }

    if (fp_list->fp_wsigma &&
            enable_print(epoch, &gp->iop.interval_for_sigma_file)) {
        print_rnn_sigma(fp_list->fp_wsigma, epoch, &rnn->rnn_p);
        fflush(fp_list->fp_wsigma);
    }

    if (fp_list->fp_winit &&
            enable_print(epoch, &gp->iop.interval_for_init_file)) {
        print_rnn_init(fp_list->fp_winit, epoch, rnn);
    }

    if (fp_list->fp_wadapt_lr &&
            enable_print(epoch, &gp->iop.interval_for_adapt_lr_file)) {
        print_adapt_lr(fp_list->fp_wadapt_lr, epoch, gp->inp.adapt_lr);
        fflush(fp_list->fp_wadapt_lr);
    }
}


static void print_open_loop_data_with_epoch (
        long epoch,
        const struct general_parameters *gp,
        struct recurrent_neural_network *rnn,
        struct output_files *fp_list)
{
    int compute_forward_dynamics = 0;

    if (fp_list->fp_werror &&
            enable_print(epoch, &gp->iop.interval_for_error_file)) {
        if (!compute_forward_dynamics) {
            rnn_forward_dynamics_forall(rnn);
            compute_forward_dynamics = 1;
        }
        print_rnn_error(fp_list->fp_werror, epoch, rnn);
        fflush(fp_list->fp_werror);
    }

    if (fp_list->fp_wstate_array &&
            enable_print(epoch, &gp->iop.interval_for_state_file)) {
        if (!compute_forward_dynamics) {
            rnn_forward_dynamics_forall(rnn);
            compute_forward_dynamics = 1;
        }
        print_rnn_state_forall(fp_list->fp_wstate_array, epoch, rnn);
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wstate_array[i], "\n");
        }
    }
}

static void print_closed_loop_data_with_epoch (
        long epoch,
        const struct general_parameters *gp,
        struct recurrent_neural_network *rnn,
        struct output_files *fp_list)
{
    int compute_forward_dynamics = 0;

    if (fp_list->fp_wclosed_error &&
            enable_print(epoch, &gp->iop.interval_for_closed_error_file)) {
        if (!compute_forward_dynamics) {
            rnn_forward_dynamics_in_closed_loop_forall(rnn,
                    gp->mp.delay_length);
            compute_forward_dynamics = 1;
        }
        print_rnn_error(fp_list->fp_wclosed_error, epoch, rnn);
        fflush(fp_list->fp_wclosed_error);
    }

    if (fp_list->fp_wclosed_state_array &&
            enable_print(epoch, &gp->iop.interval_for_closed_state_file)) {
        if (!compute_forward_dynamics) {
            rnn_forward_dynamics_in_closed_loop_forall(rnn,
                    gp->mp.delay_length);
            compute_forward_dynamics = 1;
        }
        print_rnn_state_forall(fp_list->fp_wclosed_state_array, epoch, rnn);
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wclosed_state_array[i], "\n");
        }
    }

    if (fp_list->fp_wlyapunov &&
            enable_print(epoch, &gp->iop.interval_for_lyapunov_file)) {
        if (!compute_forward_dynamics) {
            rnn_forward_dynamics_in_closed_loop_forall(rnn,
                    gp->mp.delay_length);
            compute_forward_dynamics = 1;
        }
        print_lyapunov_spectrum_of_rnn(fp_list->fp_wlyapunov, epoch, rnn,
                gp->ap.lyapunov_spectrum_size, gp->mp.delay_length,
                gp->ap.truncate_length);
        fflush(fp_list->fp_wlyapunov);
    }

    if (fp_list->fp_wentropy &&
            enable_print(epoch, &gp->iop.interval_for_entropy_file)) {
        if (!compute_forward_dynamics) {
            rnn_forward_dynamics_in_closed_loop_forall(rnn,
                    gp->mp.delay_length);
            compute_forward_dynamics = 1;
        }
        print_kl_divergence_of_rnn(fp_list->fp_wentropy, epoch, rnn,
                gp->ap.truncate_length, gp->ap.block_length,
                gp->ap.divide_num);
        fflush(fp_list->fp_wentropy);
    }

    if (fp_list->fp_wperiod &&
            enable_print(epoch, &gp->iop.interval_for_period_file)) {
        if (!compute_forward_dynamics) {
            rnn_forward_dynamics_in_closed_loop_forall(rnn,
                    gp->mp.delay_length);
            compute_forward_dynamics = 1;
        }
        print_period_of_rnn(fp_list->fp_wperiod, epoch, rnn,
                gp->ap.threshold_period);
        fflush(fp_list->fp_wperiod);
    }
}


void print_training_main_begin (
        const struct general_parameters *gp,
        const struct recurrent_neural_network *rnn,
        struct output_files *fp_list)
{
    if (fp_list->fp_wstate_array) {
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wstate_array[i], "# STATE FILE\n");
            print_general_parameters(fp_list->fp_wstate_array[i], gp);
            print_rnn_parameters(fp_list->fp_wstate_array[i], rnn);
        }
    }
    if (fp_list->fp_wclosed_state_array) {
        for (int i = 0; i < fp_list->array_size; i++) {
            fprintf(fp_list->fp_wclosed_state_array[i],  "# STATE FILE\n");
            print_general_parameters(fp_list->fp_wclosed_state_array[i], gp);
            print_rnn_parameters(fp_list->fp_wclosed_state_array[i], rnn);
        }
    }
    if (fp_list->fp_wweight) {
        fprintf(fp_list->fp_wweight, "# WEIGHT FILE\n");
        print_general_parameters(fp_list->fp_wweight, gp);
        print_rnn_parameters(fp_list->fp_wweight, rnn);
    }
    if (fp_list->fp_wthreshold) {
        fprintf(fp_list->fp_wthreshold, "# THRESHOLD FILE\n");
        print_general_parameters(fp_list->fp_wthreshold, gp);
        print_rnn_parameters(fp_list->fp_wthreshold, rnn);
    }
    if (fp_list->fp_wtau) {
        fprintf(fp_list->fp_wtau, "# TAU FILE\n");
        print_general_parameters(fp_list->fp_wtau, gp);
        print_rnn_parameters(fp_list->fp_wtau, rnn);
    }
    if (fp_list->fp_wsigma) {
        fprintf(fp_list->fp_wsigma, "# SIGMA FILE\n");
        print_general_parameters(fp_list->fp_wsigma, gp);
        print_rnn_parameters(fp_list->fp_wsigma, rnn);
    }
    if (fp_list->fp_winit) {
        fprintf(fp_list->fp_winit, "# INIT FILE\n");
        print_general_parameters(fp_list->fp_winit, gp);
        print_rnn_parameters(fp_list->fp_winit, rnn);
    }
    if (fp_list->fp_wadapt_lr) {
        fprintf(fp_list->fp_wadapt_lr, "# ADAPT_LR FILE\n");
        print_general_parameters(fp_list->fp_wadapt_lr, gp);
        print_rnn_parameters(fp_list->fp_wadapt_lr, rnn);
    }
    if (fp_list->fp_werror) {
        fprintf(fp_list->fp_werror, "# ERROR FILE\n");
        print_general_parameters(fp_list->fp_werror, gp);
        print_rnn_parameters(fp_list->fp_werror, rnn);
    }
    if (fp_list->fp_wclosed_error) {
        fprintf(fp_list->fp_wclosed_error, "# ERROR FILE\n");
        print_general_parameters(fp_list->fp_wclosed_error, gp);
        print_rnn_parameters(fp_list->fp_wclosed_error, rnn);
    }
    if (fp_list->fp_wlyapunov) {
        fprintf(fp_list->fp_wlyapunov, "# LYAPUNOV FILE\n");
        print_general_parameters(fp_list->fp_wlyapunov, gp);
        print_rnn_parameters(fp_list->fp_wlyapunov, rnn);
    }
    if (fp_list->fp_wentropy) {
        fprintf(fp_list->fp_wentropy, "# ENTROPY FILE\n");
        print_general_parameters(fp_list->fp_wentropy, gp);
        print_rnn_parameters(fp_list->fp_wentropy, rnn);
    }
    if (fp_list->fp_wperiod) {
        fprintf(fp_list->fp_wperiod, "# PERIOD FILE\n");
        print_general_parameters(fp_list->fp_wperiod, gp);
        print_rnn_parameters(fp_list->fp_wperiod, rnn);
    }
}

void print_training_main_loop (
        long epoch,
        const struct general_parameters *gp,
        struct recurrent_neural_network *rnn,
        struct output_files *fp_list)
{
    print_parameters_with_epoch(epoch, gp, rnn, fp_list);
    print_open_loop_data_with_epoch(epoch, gp, rnn, fp_list);
    print_closed_loop_data_with_epoch(epoch, gp, rnn, fp_list);
}


