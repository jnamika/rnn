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

#ifndef RNN_H
#define RNN_H


typedef enum rnn_output_t {
    STANDARD_TYPE,
    SOFTMAX_TYPE
} rnn_output_t;


typedef struct rnn_parameters {
    int in_state_size;
    int c_state_size;
    int out_state_size;

    /*
     * If output_type == STANDARD_TYPE, tanh() is used as a neural firing
     * function in the output layer.
     * If output_type == SOFTMAX_TYPE, softmax function is used in the output
     * layer.
     */
    enum rnn_output_t output_type;


    /*
     * If fixed_[weight|threshold|tau|init_c_state|sigma] != 0,
     * then it does not change by learning.
     */
    int fixed_weight;
    int fixed_threshold;
    int fixed_tau;
    int fixed_init_c_state;
    int fixed_sigma;

    int *const_init_c;

    /*
     * softmax_group_num means the number of groups for output neurons.
     * If softmax_group_id[i] == c, output neuron i belongs to the group c,
     * and a neural activation of neuron i is computed by the softmax function
     * over the group c if output_type == SOFTMAX_TYPE.
     * Note that softmax_group_id has to satisfy 0 <= softmax_group_id[i] and
     * softmax_group_id[i] < softmax_group_num.
     */
    int softmax_group_num;
    int *softmax_group_id;

    double sigma;
    double variance;
    double **weight_ci;
    double **weight_cc;
    double **weight_oc;
    double *threshold_c;
    double *threshold_o;
    double *tau;
    double *eta;

    double delta_sigma;
    double **delta_weight_ci;
    double **delta_weight_cc;
    double **delta_weight_oc;
    double *delta_threshold_c;
    double *delta_threshold_o;
    double *delta_tau;

    double prior_strength;
    double prior_sigma;
    double **prior_weight_ci;
    double **prior_weight_cc;
    double **prior_weight_oc;
    double *prior_threshold_c;
    double *prior_threshold_o;
    double *prior_tau;

    struct connection_domain {
        int begin;
        int end;
    } **connection_ci;
    struct connection_domain **connection_cc;
    struct connection_domain **connection_oc;

#ifdef ENABLE_ADAPTIVE_LEARNING_RATE
    double tmp_sigma;
    double tmp_variance;
    double *tmp_weight_ci;
    double *tmp_weight_cc;
    double *tmp_weight_oc;
    double *tmp_threshold_c;
    double *tmp_threshold_o;
    double *tmp_tau;
    double *tmp_eta;
#endif
} rnn_parameters;

typedef struct rnn_state {
    struct rnn_parameters *rnn_p;
    int length;

    double *init_c_inter_state;
    double *init_c_state;
    double *delta_init_c_inter_state;

    double **in_state;
    double **c_state;
    double **out_state;
    double **teach_state;

    double **c_inputsum;
    double **c_inter_state;
    double **o_inter_state;

    double **likelihood;
    double **delta_likelihood;
    double **delta_c_inter;
    double **delta_o_inter;

    double delta_s;
    double **delta_w_ci;
    double **delta_w_cc;
    double **delta_w_oc;
    double *delta_t_c;
    double *delta_t_o;
    double *delta_tau;
    double *delta_i;

#ifdef ENABLE_ADAPTIVE_LEARNING_RATE
    double *tmp_init_c_inter_state;
    double *tmp_init_c_state;
#endif
} rnn_state;


typedef struct recurrent_neural_network {
    int series_num;
    struct rnn_state *rnn_s;
    struct rnn_parameters rnn_p;
} recurrent_neural_network;


void init_rnn_parameters (
        struct rnn_parameters *rnn_p,
        int in_state_size,
        int c_state_size,
        int out_state_size);

void init_rnn_state (
        struct rnn_state *rnn_s,
        struct rnn_parameters *rnn_p,
        int length,
        double **input,
        double **target);

void init_recurrent_neural_network (
        struct recurrent_neural_network *rnn,
        int in_state_size,
        int c_state_size,
        int out_state_size);

void rnn_add_target (
        struct recurrent_neural_network *rnn,
        int length,
        double **input,
        double **target);

void rnn_clean_target (struct recurrent_neural_network *rnn);

void rnn_parameters_alloc (struct rnn_parameters *rnn_p);
void rnn_state_alloc (struct rnn_state *rnn_s);

void free_rnn_parameters (struct rnn_parameters *rnn_p);
void free_rnn_state (struct rnn_state *rnn_s);
void free_recurrent_neural_network (struct recurrent_neural_network *rnn);



void fwrite_rnn_parameters (
        const struct rnn_parameters *rnn_p,
        FILE *fp);

void fread_rnn_parameters (
        struct rnn_parameters *rnn_p,
        FILE *fp);

void fwrite_rnn_state (
        const struct rnn_state *rnn_s,
        FILE *fp);

void fread_rnn_state (
        struct rnn_state *rnn_s,
        FILE *fp);

void fwrite_recurrent_neural_network (
        const struct recurrent_neural_network *rnn,
        FILE *fp);

void fread_recurrent_neural_network (
        struct recurrent_neural_network *rnn,
        FILE *fp);




void rnn_reset_delta_parameters (struct rnn_parameters *rnn_p);

void rnn_reset_prior_distribution (struct rnn_parameters *rnn_p);


void rnn_get_connection (
        int size,
        const struct connection_domain *connection,
        int *has_connection);

void rnn_set_connection (
        int size,
        struct connection_domain *connection,
        const int *has_connection);

void rnn_add_connection (
        int size,
        struct connection_domain *connection,
        int begin,
        int end);

void rnn_delete_connection (
        int size,
        struct connection_domain *connection,
        int begin,
        int end);

void rnn_reset_weight_by_connection (struct rnn_parameters *rnn_p);

void rnn_set_uniform_tau (
        struct rnn_parameters *rnn_p,
        double tau);

void rnn_set_tau (
        struct rnn_parameters *rnn_p,
        const double *tau);

void rnn_set_sigma (
        struct rnn_parameters *rnn_p,
        double sigma);

int rnn_get_total_length (const struct recurrent_neural_network *rnn);

double rnn_get_error (const struct rnn_state *rnn_s);
double rnn_get_total_error (const struct recurrent_neural_network *rnn);

double rnn_get_likelihood (const struct rnn_state *rnn_s);
double rnn_get_total_likelihood (const struct recurrent_neural_network *rnn);


void rnn_forward_context_map (
        const struct rnn_parameters *rnn_p,
        const double *in_state,
        const double *prev_c_inter_state,
        const double *prev_c_state,
        double *c_inputsum,
        double *c_inter_state,
        double *c_state);

void rnn_forward_output_map (
        const struct rnn_parameters *rnn_p,
        const double *c_state,
        double *o_inter_state,
        double *out_state);

void rnn_forward_map (
        const struct rnn_parameters *rnn_p,
        const double *in_state,
        const double *prev_c_inter_state,
        const double *prev_c_state,
        double *c_inputsum,
        double *c_inter_state,
        double *c_state,
        double *o_inter_state,
        double *out_state);

void rnn_forward_dynamics (struct rnn_state *rnn_s);

void rnn_forward_dynamics_in_closed_loop (
        struct rnn_state *rnn_s,
        int delay_length);

void rnn_forward_dynamics_forall (struct recurrent_neural_network *rnn);

void rnn_forward_dynamics_in_closed_loop_forall (
        struct recurrent_neural_network *rnn,
        int delay_length);


void rnn_backward_output_map (
        const struct rnn_parameters *rnn_p,
        const double *delta_likelihood,
        const double *out_state,
        double *delta_o_inter);

void rnn_backward_context_map (
        const struct rnn_parameters *rnn_p,
        const double *delta_o_inter,
        const double *next_delta_c_inter,
        const double *c_state,
        double *delta_c_inter);

void rnn_set_likelihood (struct rnn_state *rnn_s);

void rnn_backward_map (
        const struct rnn_parameters *rnn_p,
        const double *delta_likelihood,
        const double *next_delta_c_inter,
        const double *c_state,
        const double *out_state,
        double *delta_c_inter,
        double *delta_o_inter);

void rnn_backward_dynamics (struct rnn_state *rnn_s);

void rnn_forward_backward_dynamics (struct rnn_state *rnn_s);

void rnn_forward_backward_dynamics_forall (
        struct recurrent_neural_network *rnn);

void rnn_set_delta_w (struct rnn_state *rnn_s);
void rnn_set_delta_t (struct rnn_state *rnn_s);
void rnn_set_delta_tau (struct rnn_state *rnn_s);
void rnn_set_delta_s (struct rnn_state *rnn_s);
void rnn_set_delta_i (struct rnn_state *rnn_s);
void rnn_set_delta_parameters (struct rnn_state *rnn_s);


void rnn_update_delta_weight (
        struct recurrent_neural_network *rnn,
        double momentum);

void rnn_update_delta_threshold (
        struct recurrent_neural_network *rnn,
        double momentum);

void rnn_update_delta_tau (
        struct recurrent_neural_network *rnn,
        double momentum);

void rnn_update_delta_sigma (
        struct recurrent_neural_network *rnn,
        double momentum);

void rnn_update_delta_init_c_inter_state (
        struct rnn_state *rnn_s,
        double momentum);

void rnn_update_weight (
        struct rnn_parameters *rnn_p,
        double rho);

void rnn_update_threshold (
        struct rnn_parameters *rnn_p,
        double rho);

void rnn_update_tau (
        struct rnn_parameters *rnn_p,
        double rho);

void rnn_update_sigma (
        struct rnn_parameters *rnn_p,
        double rho);

void rnn_update_init_c_inter_state (
        struct rnn_state *rnn_s,
        double rho);

void rnn_update_delta_parameters (
        struct recurrent_neural_network *rnn,
        double momentum);

void rnn_update_parameters (
        struct recurrent_neural_network *rnn,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma);

void rnn_learn (
        struct recurrent_neural_network *rnn,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma,
        double momentum);

void rnn_learn_s (
        struct recurrent_neural_network *rnn,
        double rho,
        double momentum);

#ifdef ENABLE_ADAPTIVE_LEARNING_RATE

void rnn_backup_learning_parameters (struct recurrent_neural_network *rnn);
void rnn_restore_learning_parameters (struct recurrent_neural_network *rnn);

double rnn_update_parameters_with_adapt_lr (
        struct recurrent_neural_network *rnn,
        double adapt_lr,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma);

double rnn_learn_with_adapt_lr (
        struct recurrent_neural_network *rnn,
        double adapt_lr,
        double rho_weight,
        double rho_tau,
        double rho_init,
        double rho_sigma,
        double momentum);

double rnn_learn_s_with_adapt_lr (
        struct recurrent_neural_network *rnn,
        double adapt_lr,
        double rho,
        double momentum);

#endif

double** rnn_jacobian_matrix (
        double** matrix,
        const struct rnn_parameters *rnn_p,
        const double *prev_c_state,
        const double *c_state,
        const double *out_state);


void rnn_update_prior_strength (
        struct recurrent_neural_network *rnn,
        double lambda,
        double alpha);

#endif

