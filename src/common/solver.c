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
#include <string.h>
#include <math.h>
#include "mt19937ar.h"

#include "solver.h"
#include "utils.h"


static int compar (const void* x, const void* y);
static void product (const double* const *a, const double* b, double* ab, int m,
        int n);
static double get_length (const double* vector, int n);
static void swap_maxlen_to_head (double** vector, int m, int n);
static double scalar_product (const double* vector1, const double* vector2,
        int n);
static void resize (double* vector, int n, double length);
static double get_distance (const double* vector1, const double* vector2,
        int n);
static int index_of_nearest_point (const double* const *data, int I, int t,
        int n);
static int index_of_nearest_point_in_epsilon_neighborhood (
        const double* const *data, int I, int t, int n, double epsilon);
static int check_in_box (const double* vector, const double* median_point,
        int n, double epsilon);


/*
 * returns the Lyapunov spectrum from n-dimensional time series by using the
 * Jacobian matrix
 *
 *   @parameter  data     : n-dimensional time series
 *   @parameter  t        : length of time series
 *   @parameter  m        : number of Lyapunov exponents
 *   @parameter  n        : dimension of time series
 *   @parameter  T        : interval to compute Gram-Schmidt orthogonalization
 *   @parameter  pfunc    : function pointer to compute Jacobian matrix
 *   @parameter  obj      : object used in pfunc
 *   @parameter  spectrum : Lyapunov spectrum (result)
 *   @parameter  matrix   : Jacobian matrix (temporal data cache)
 *   @parameter  vector   : orthogonal vector (temporal data cache)
 *
 *   @return              : Lyapunov spectrum
 */
double* lyapunov_spectrum (const double* const *data, int t, int m, int n,
        int T, Jacobian_matrix pfunc, void *obj, double* spectrum,
        double **matrix, double ***vector)
{
    if (m < 1 || n < 1) return spectrum;

    double tmp_vec[n];
    int flag_matrix_alloc, flag_vector_alloc;

    if (matrix == NULL) {
        MALLOC(matrix, n);
        MALLOC(matrix[0], n * n);
        for (int i = 0; i < n; i++) {
            matrix[i] = matrix[0] + i * n;
        }
        flag_matrix_alloc = 1;
    } else {
        flag_matrix_alloc = 0;
    }
    if (vector == NULL) {
        MALLOC(vector, m);
        MALLOC(vector[0], m * m);
        MALLOC(vector[0][0], m * m * n);
        for (int i = 0; i < m; i++) {
            vector[i] = vector[0] + i * m;
            vector[i][0] = vector[0][0] + i * m * n;
            for (int j = 0; j < m; j++) {
                vector[i][j] = vector[i][0] + j * n;
            }
        }
        flag_vector_alloc = 1;
    } else {
        flag_vector_alloc = 0;
    }

    for (int i = 0; i < m; i++) {
        spectrum[i] = 0; /* initializes i'th Lyapunov exponent */

        /* vector initialization */
        for (int j = 0; j <= i; j++) {
            for (int k = 0; k < n; k++) {
                vector[i][j][k] = (k%(i+1) == j) ? 1 : 0;
            }
        }
        gram_schmidt_orthogonalization(vector[i], i+1, n);
        for (int j = 0; j <= i; j++) {
            resize(vector[i][j], n, 1); /* resizes vector length to 1 */
        }
    }

    /* computing the Lyapunov spectrum */
    for (int i = 0; i < t; i++) {
        /* computing Jacobian matrix */
        if (pfunc(data[i], n, i, matrix, obj) == NULL) return NULL;
        for (int j = 0; j < m; j++) {
            for (int k = 0; k <= j; k++) {
                product((const double* const*)matrix, vector[j][k],
                        tmp_vec, n, n);
                memcpy(vector[j][k], tmp_vec, n * sizeof(double));
            }
        }
        if (( (i+1) % T ) == 0 || i+1 == t) {
            for (int j = 0; j < m; j++) {
                gram_schmidt_orthogonalization(vector[j], j+1, n);
                for (int k = 0; k <= j; k++) {
                    // computing expansion rate
                    spectrum[j] += log(get_length(vector[j][k], n));
                    resize(vector[j][k], n, 1); /* resizes vector length to 1 */
                }
            }
        }
    }
    for (int i = 0; i < m; i++) {
        spectrum[i] /= t; /* time average */
        for (int j = 0; j < i; j++) {
            spectrum[i] -= spectrum[j];
        }
    }
    qsort(spectrum, m, sizeof(double), compar);

    if (flag_matrix_alloc) {
        free(matrix[0]);
        free(matrix);
    }
    if (flag_vector_alloc) {
        free(vector[0][0]);
        free(vector[0]);
        free(vector);
    }
    return spectrum;
}




/*
 * transforms vectors by using Gram-Schmidt orthogonalization algorithm
 *
 *   @parameter  vector : list of vectors
 *   @parameter  m      : number of vectors
 *   @parameter  n      : dimension of vectors (m <= n)
 */
void gram_schmidt_orthogonalization (double** vector, int m, int n)
{
    if (m <= 1) return;

    for (int i = 0; i < m; i++) {
        swap_maxlen_to_head(vector+i, m-i, n);
        for (int j = i+1; j < m; j++) {
            double alpha = scalar_product(vector[i], vector[j], n);
            double length = get_length(vector[i], n);
            /* check length > 0 (if length=0, then alpha=0) */
            if (isnormal(length)) {
                alpha /= length * length;
            }
            for (int k = 0; k < n; k++) {
                vector[j][k] -= alpha * vector[i][k];
            }
        }
    }
}




/*
 * returns the maximum Lyapunov exponent from n-dimensional time series
 * this algorithm was proposed by M. Sato, S. Sano and Y. Sawada
 * (Phys. Rev. Lett 55, 1082, 1985)
 *
 *   @parameter  data   : n-dimensional time series
 *   @parameter  t      : length of time series
 *   @parameter  n      : dimension of time series
 *   @parameter  T      : sampling interval
 *   @parameter  N      : number of sample
 *
 *   @return            : maximum Lyapunov exponent
 */
double lyapunov_exponent_sss (const double* const *data, int t, int n, int T,
        int N)
{
    if (t <= T) return 0;

    double lyap_exp = 0;
    for (int i = 0; i < N; i++) {
        /* selects a sample from data randomly */
        int I = genrand_int32() % (t-T);
        int J = index_of_nearest_point((const double* const*)data, I, t-T, n);
        double dist_0 = get_distance(data[I], data[J], n);
        double dist_T = get_distance(data[I+T], data[J+T], n);
        if (isnormal(dist_0)) { /* check dist_0 > 0 */
            lyap_exp += log(dist_T/dist_0);
        }
    }
    lyap_exp /= N*T;
    return lyap_exp;
}





/*
 * returns the maximum Lyapunov exponent from n-dimensional time series
 * this algorithm was proposed by A. Wolf et al. (Physica D 16, 285-317, 1985)
 *
 *   @parameter  data    : n-dimensional time series
 *   @parameter  t       : length of time series
 *   @parameter  n       : dimension of time series
 *   @parameter  epsilon : supremum of distance
 *
 *   @return            : maximum Lyapunov exponent
 */
double lyapunov_exponent_wolf (const double* const *data, int t, int n,
        double epsilon)
{
    int I, J, T;
    double lyap_exp;

    lyap_exp = 0;
    I = 0;
    while ((J = index_of_nearest_point_in_epsilon_neighborhood(
                    (const double* const*)data, I, t-1, n, epsilon)) != -1) {
        for (T = 1; I+T < t-1 && J+T < t-1; T++) {
            if ( get_distance(data[I+T], data[J+T], n) > epsilon ) break;
        }
        double dist_0 = get_distance(data[I], data[J], n);
        double dist_T = get_distance(data[I+T], data[J+T], n);
        if (isnormal(dist_0)) { /* check dist_0 > 0 */
            lyap_exp += log(dist_T/dist_0);
        }
        I = I + T;
        if (I >= t - 1) break;
    }
    if (I != 0) lyap_exp /= I;
    return lyap_exp;
}



/*
 * This function computes the number of hypercubes containing of data
 *
 *   @parameter  data      : n-dimensional time series
 *   @parameter  t         : length of time series
 *   @parameter  n         : dimension of time series
 *   @parameter  epsilon   : size of a hypercube
 *   @parameter  box_count : number of data in a hypercube (result)
 *
 *   @return               : number of hypercubes containing of data
 */
int box_counter (const double* const *data, int t, int n, double epsilon,
        int* box_count)
{
    const double* median_points[t];
    int num = 0;
    for (int i = 0; i < t; i++) {
        int flag = 0;
        for (int j = 0; j < num; j++) {
            if (check_in_box(data[i], median_points[j], n, epsilon)) {
                box_count[j]++;
                flag = 1;
                break;
            }
        }
        if (flag == 0) {
            median_points[num] = data[i];
            box_count[num] = 1;
            num++;
        }
    }
    return num;
}


/*
 * This function computes the generalized dimension
 *
 *   @parameter  data     : n-dimensional time series
 *   @parameter  t        : length of time series
 *   @parameter  n        : dimension of time series
 *   @parameter  epsilon  : size of a hypercube
 *   @parameter  q        : weight of a hypercube
 *   @parameter  box_num  : number of hypercubes containing of data (result)
 *
 *   @return              : generalized dimension
 */
double generalized_dimension (const double* const *data, int t, int n,
        double epsilon, double q, int* box_num)
{
    double dim;
    int box_count[t];
    const int num = box_counter(data, t, n, epsilon, box_count);
    /* for the case of information dimension (q==1) */
    if (fpclassify(q - 1.0) == FP_ZERO) {
        dim = 0;
        for (int i = 0; i < num; i++) {
            double mu = box_count[i]/(double)t;
            dim += mu * log(mu);
        }
        dim /= log(epsilon);
    } else {         /* otherwise (q!=1) */
        dim = 0;
        for (int i = 0; i < num; i++) {
            double mu = box_count[i]/(double)t;
            dim += pow(mu, q);
        }
        dim = log(dim) / (log(epsilon) * (q-1));
    }
    if (box_num != NULL) *box_num = num;
    return dim;
}



/*
 * This function computes the capacity dimension
 * (generalized dimension with q==0)
 */
double capacity_dimension (const double* const *data, int t, int n,
        double epsilon, int* box_num)
{
    return generalized_dimension(data, t, n, epsilon, 0, box_num);
}

/*
 * This function computes the information dimension
 * (generalized dimension with q==1)
 */
double information_dimension (const double* const *data, int t, int n,
        double epsilon, int* box_num)
{
    return generalized_dimension(data, t, n, epsilon, 1, box_num);
}

/*
 * This function computes the correlation dimension
 * (generalized dimension with q==2)
 */
double correlation_dimension (const double* const *data, int t, int n,
        double epsilon, int* box_num)
{
    return generalized_dimension(data, t, n, epsilon, 2, box_num);
}


/*
 * embeds 1-dimensional data into another Euclidean space by using the method of
 * delays
 *
 *   @parameter  data           : 1-dimensional time series
 *   @parameter  t              : length of time series
 *   @parameter  n              : embedding dimension
 *   @parameter  embedding_data : time series embedded in n-dimension space
 *                                (result)
 *                                this needs a cache of sizeof(double[t-n+1][n])
 *                                in order to store the data
 *
 *   @return                    : length of embedding_data
 */
int get_embedding_data (const double* data, int t, int n,
        double** embedding_data)
{
    const int emb_num = t - n + 1;
    for (int i = 0; i < emb_num; i++) {
        for (int j = 0; j < n; j++) {
            embedding_data[i][j] = data[i+j];
        }
    }
    return emb_num;
}






static int compar (const void* x, const void* y)
{
    double a = *((double*)x);
    double b = *((double*)y);
    if (a > b) {
        return -1;
    } else if (a < b){
        return 1;
    } else {
        return 0;
    }
}

static void product (const double* const *a, const double* b, double* ab,
        int m, int n)
{
    for (int i = 0; i < m; i++) {
        ab[i] = 0;
        for (int j = 0; j < n; j++) {
            ab[i] += a[i][j] * b[j];
        }
    }
}

static double get_length (const double* vector, int n)
{
    double len = 0;
    for (int i = 0; i < n; i++) {
        len += vector[i] * vector[i];
    }
    return sqrt(len);
}


static void swap_maxlen_to_head (double** vector, int m, int n)
{
    double length[m], tmp[n];
    for (int i = 0; i < m; i++) {
        length[i] = get_length(vector[i], n);
    }
    int max_id = 0;
    for (int i = 1; i < m; i++) {
        if (length[max_id] < length[i]) {
            max_id = i;
        }
    }
    if (max_id != 0) {
        memcpy(tmp, vector[0], sizeof(double) * n);
        memcpy(vector[0], vector[max_id], sizeof(double) * n);
        memcpy(vector[max_id], tmp, sizeof(double) * n);
    }
}


static double scalar_product (const double* vector1, const double* vector2,
        int n)
{
    double prod = 0;
    for (int i = 0; i < n; i++) {
        prod += vector1[i] * vector2[i];
    }
    return prod;
}

static void resize (double* vector, int n, double length)
{
    double rate = get_length(vector, n)/length;
    for (int i = 0; i < n; i++) {
        vector[i] /= rate;
    }
}

static double get_distance (const double* vector1, const double* vector2,
        int n)
{
    double dist = 0;
    for (int i = 0; i < n; i++) {
        double d = vector1[i] - vector2[i];
        dist += d * d;
    }
    return sqrt(dist);
}

static int index_of_nearest_point (const double* const *data, int I, int t,
        int n)
{
    int J = (I+1) % t;
    double minimum = get_distance(data[I], data[J], n);
    for (int i = 0; i < t; i++) {
        if (i == I) continue;
        double distance = get_distance(data[I], data[i], n);
        if (distance < minimum) {
            minimum = distance;
            J = i;
        }
    }
    return J;
}

static int index_of_nearest_point_in_epsilon_neighborhood (
        const double* const *data, int I, int t, int n, double epsilon)
{
    int J = index_of_nearest_point(data, I, t, n);
    if (get_distance(data[I], data[J], n) > epsilon) return -1;
    return J;
}

static int check_in_box (const double* vector, const double* median_point,
        int n, double epsilon)
{
    for (int i = 0; i < n; i++) {
        if (vector[i] < median_point[i]-(epsilon/2) ||
                median_point[i]+(epsilon/2) < vector[i]) {
            return 0;
        }
    }
    return 1;
}


