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

#ifndef SOLVER_H
#define SOLVER_H

/*
 * A function pointer to compute Jacobian matrix
 *
 *   @parameter  vector : n-dimensional vector
 *   @parameter    n    : dimension
 *   @parameter    t    : current time step
 *   @parameter  matrix : Jacobian matrix
 *   @parameter   obj   : object
 *
 *   @return            : Jacobian matrix (returns NULL on failure)
 */
typedef double** (*Jacobian_matrix)(const double* vector, int n, int t,
        double** matrix, void *obj);


void gram_schmidt_orthogonalization (double** vector, int m, int n);
double* lyapunov_spectrum (const double* const *data, int t, int m, int n,
        int T, Jacobian_matrix pfunc, void *obj, double* spectrum,
        double **matrix, double ***vector);
double lyapunov_exponent_sss (const double* const *vector, int t, int n,
        int T, int N);
double lyapunov_exponent_wolf (const double* const *vector, int t, int n,
        double epsilon);

int box_counter (const double* const *data, int t, int n, double epsilon,
        int* box_count);
double generalized_dimension (const double* const *data, int t, int n,
        double epsilon, double q, int* box_num);
double capacity_dimension (const double* const *data, int t, int n,
        double epsilon, int* box_num);
double information_dimension (const double* const *data, int t, int n,
        double epsilon, int* box_num);
double correlation_dimension (const double* const *data, int t, int n,
        double epsilon, int* box_num);
int get_embedding_data (const double* data, int t, int n,
        double** embedding_data);

#endif

