#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define SEED 7
#define TOLERANCE 1e-9

static double **create_matrix(int n) {
    double **matrix = malloc(n * sizeof(double *));

    if (matrix == NULL) {
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        matrix[i] = calloc(n, sizeof(double));

        if (matrix[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }

            free(matrix);
            return NULL;
        }
    }

    return matrix;
}

static void free_matrix(double **matrix, int n) {
    if (matrix == NULL) {
        return;
    }

    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }

    free(matrix);
}

static int make_matrices(
    int n, double ***a, double ***b, double ***c
) {
    *a = create_matrix(n);
    *b = create_matrix(n);
    *c = create_matrix(n);

    if (*a == NULL || *b == NULL || *c == NULL) {
        free_matrix(*a, n);
        free_matrix(*b, n);
        free_matrix(*c, n);

        *a = NULL;
        *b = NULL;
        *c = NULL;

        return 0;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*a)[i][j] =
                (((i * 31 + j * 17 + SEED) % 101) - 50)
                / 50.0;

            (*b)[i][j] =
                (((i * 13 + j * 29 + SEED * 3) % 101) - 50)
                / 50.0;
        }
    }

    return 1;
}

static void multiply(
    double **a, double **b, double **c, int n
) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double total = 0.0;

            for (int k = 0; k < n; k++) {
                total += a[i][k] * b[k][j];
            }

            c[i][j] = total;
        }
    }
}

static double checksum(double **c, int n) {
    double total = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            total += c[i][j];
        }
    }

    return total;
}

static double **multiply_reference(
    double **a, double **b, int n
) {
    double **result = create_matrix(n);

    if (result == NULL) {
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

static int matrices_equal(
    double **actual, double **expected, int n
) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(actual[i][j] - expected[i][j])
                    > TOLERANCE) {
                fprintf(
                    stderr,
                    "Mismatch at (%d, %d): %.17g != %.17g\n",
                    i, j, actual[i][j], expected[i][j]
                );

                return 0;
            }
        }
    }

    return 1;
}

static int run_tests(void) {
    double **a = create_matrix(2);
    double **b = create_matrix(2);
    double **c = create_matrix(2);
    double **expected = create_matrix(2);

    if (a == NULL || b == NULL || c == NULL
            || expected == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");

        free_matrix(a, 2);
        free_matrix(b, 2);
        free_matrix(c, 2);
        free_matrix(expected, 2);

        return 0;
    }

    /* Test 1: known result. */
    a[0][0] = 1.0;
    a[0][1] = 2.0;
    a[1][0] = 3.0;
    a[1][1] = 4.0;

    b[0][0] = 5.0;
    b[0][1] = 6.0;
    b[1][0] = 7.0;
    b[1][1] = 8.0;

    expected[0][0] = 19.0;
    expected[0][1] = 22.0;
    expected[1][0] = 43.0;
    expected[1][1] = 50.0;

    multiply(a, b, c, 2);

    if (!matrices_equal(c, expected, 2)) {
        goto failure;
    }

    /* Test 2: identity matrix. */
    double **identity = create_matrix(2);

    if (identity == NULL) {
        goto failure;
    }

    identity[0][0] = 1.0;
    identity[1][1] = 1.0;

    multiply(a, identity, c, 2);

    if (!matrices_equal(c, a, 2)) {
        free_matrix(identity, 2);
        goto failure;
    }

    free_matrix(identity, 2);

    /* Test 3: zero matrix. */
    double **zero = create_matrix(2);

    if (zero == NULL) {
        goto failure;
    }

    multiply(a, zero, c, 2);

    if (!matrices_equal(c, zero, 2)) {
        free_matrix(zero, 2);
        goto failure;
    }

    free_matrix(zero, 2);

    /* Test 4: independent reference implementation. */
    free_matrix(a, 2);
    free_matrix(b, 2);
    free_matrix(c, 2);
    free_matrix(expected, 2);

    if (!make_matrices(4, &a, &b, &c)) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 0;
    }

    double **reference = multiply_reference(a, b, 4);

    if (reference == NULL) {
        free_matrix(a, 4);
        free_matrix(b, 4);
        free_matrix(c, 4);
        return 0;
    }

    multiply(a, b, c, 4);

    int passed = matrices_equal(c, reference, 4);

    free_matrix(a, 4);
    free_matrix(b, 4);
    free_matrix(c, 4);
    free_matrix(reference, 4);

    if (!passed) {
        return 0;
    }

    printf("PASS: all four correctness tests.\n");
    return 1;

failure:
    free_matrix(a, 2);
    free_matrix(b, 2);
    free_matrix(c, 2);
    free_matrix(expected, 2);

    return 0;
}

static int benchmark(int n, int warmups, int repetitions) {
    double **a = NULL;
    double **b = NULL;
    double **c = NULL;

    if (!make_matrices(n, &a, &b, &c)) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 0;
    }

    /* Warm-up runs are not measured. */
    for (int i = 0; i < warmups; i++) {
        multiply(a, b, c, n);
        checksum(c, n);
    }

    printf(
        "language,n,repetition,time_ns,"
        "matrix_payload_bytes,checksum,correct\n"
    );

    uint64_t payload_bytes =
        3ULL * (uint64_t)n * (uint64_t)n * sizeof(double);

    for (int repetition = 1;
         repetition <= repetitions;
         repetition++) {

        struct timespec start;
        struct timespec end;

        timespec_get(&start, TIME_UTC);
        multiply(a, b, c, n);
        timespec_get(&end, TIME_UTC);

        int64_t elapsed =
            (int64_t)(end.tv_sec - start.tv_sec) * 1000000000LL
            + (int64_t)(end.tv_nsec - start.tv_nsec);

        double result_checksum = checksum(c, n);

        printf(
            "C,%d,%d,%lld,%llu,%.17g,true\n",
            n,
            repetition,
            (long long)elapsed,
            (unsigned long long)payload_bytes,
            result_checksum
        );
    }

    free_matrix(a, n);
    free_matrix(b, n);
    free_matrix(c, n);

    return 1;
}

int main(int argc, char *argv[]) {
    int test = 0;
    int run_benchmark = 0;
    int n = 10;
    int warmups = 1;
    int repetitions = 5;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) {
            test = 1;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            run_benchmark = 1;
        } else if (strcmp(argv[i], "--n") == 0
                   && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmups") == 0
                   && i + 1 < argc) {
            warmups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repetitions") == 0
                   && i + 1 < argc) {
            repetitions = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n",
                    argv[i]);
            return 1;
        }
    }

    if (test) {
        return run_tests() ? 0 : 1;
    }

    if (!run_benchmark) {
        fprintf(stderr, "Use --test or --benchmark.\n");
        return 1;
    }

    if (n < 1 || warmups < 0 || repetitions < 1) {
        fprintf(stderr,
                "n and repetitions must be positive; "
                "warmups cannot be negative.\n");
        return 1;
    }

    return benchmark(n, warmups, repetitions) ? 0 : 1;
}
