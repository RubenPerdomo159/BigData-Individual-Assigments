#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_GROUPS 10000
#define MAX_LINE 1024

typedef struct {
    char language[16];
    int n;
    double *times;
    int count;
    int capacity;
    unsigned long long payload_bytes;
} Group;

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static int compare_groups(const void *a, const void *b) {
    const Group *x = a;
    const Group *y = b;

    if (x->n != y->n) {
        return (x->n > y->n) - (x->n < y->n);
    }

    return strcmp(x->language, y->language);
}

static double quantile(
    const double sorted[], int count, double fraction
) {
    if (count == 1) {
        return sorted[0];
    }

    double position = (count - 1) * fraction;
    int lower = (int)position;
    int upper = lower + 1;
    double weight = position - lower;

    return sorted[lower] * (1.0 - weight)
         + sorted[upper] * weight;
}

static int add_time(Group *group, double seconds) {
    if (group->count == group->capacity) {
        int new_capacity = group->capacity == 0
            ? 8 : group->capacity * 2;

        double *new_times = realloc(
            group->times,
            (size_t)new_capacity * sizeof(double)
        );

        if (new_times == NULL) {
            return 0;
        }

        group->times = new_times;
        group->capacity = new_capacity;
    }

    group->times[group->count++] = seconds;
    return 1;
}

int main(void) {
    FILE *input = fopen("data/raw_measurements.csv", "r");

    if (input == NULL) {
        perror("Could not open data/raw_measurements.csv");
        fprintf(stderr, "Run run_benchmarks first.\n");
        return 1;
    }

    char line[MAX_LINE];

    if (fgets(line, sizeof(line), input) == NULL) {
        fprintf(stderr, "CSV file is empty.\n");
        fclose(input);
        return 1;
    }

    Group *groups = calloc(MAX_GROUPS, sizeof(Group));

    if (groups == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        fclose(input);
        return 1;
    }

    int group_count = 0;
    int line_number = 1;
    int success = 1;

    while (fgets(line, sizeof(line), input) != NULL) {
        line_number++;

        char language[16];
        char correct[16];
        int n, repetition;
        long long time_ns;
        unsigned long long payload;

        int fields = sscanf(
            line,
            "%15[^,],%d,%d,%lld,%llu,%*lf,%15[^,\r\n]",
            language, &n, &repetition,
            &time_ns, &payload, correct
        );

        if (fields != 6 || strcmp(correct, "true") != 0
                || n < 1 || time_ns < 0) {
            fprintf(stderr, "Invalid CSV row at line %d.\n",
                    line_number);
            success = 0;
            break;
        }

        int index = 0;

        while (index < group_count) {
            if (groups[index].n == n
                    && strcmp(groups[index].language, language) == 0) {
                break;
            }
            index++;
        }

        if (index == group_count) {
            if (group_count >= MAX_GROUPS) {
                fprintf(stderr, "Too many groups.\n");
                success = 0;
                break;
            }

            strcpy(groups[index].language, language);
            groups[index].n = n;
            groups[index].payload_bytes = payload;
            group_count++;
        }

        if (!add_time(&groups[index], time_ns / 1e9)) {
            fprintf(stderr, "Memory allocation failed.\n");
            success = 0;
            break;
        }

        groups[index].payload_bytes = payload;
    }

    fclose(input);

    if (!success || group_count == 0) {
        if (group_count == 0 && success) {
            fprintf(stderr, "No measurements found.\n");
        }

        for (int i = 0; i < group_count; i++) {
            free(groups[i].times);
        }

        free(groups);
        return 1;
    }

    qsort(groups, group_count, sizeof(Group), compare_groups);

    FILE *output = fopen("data/summary.csv", "w");

    if (output == NULL) {
        perror("Could not create data/summary.csv");

        for (int i = 0; i < group_count; i++) {
            free(groups[i].times);
        }

        free(groups);
        return 1;
    }

    fprintf(
        output,
        "language,n,repetitions,median_seconds,"
        "q1_seconds,q3_seconds,iqr_seconds,matrix_payload_bytes\n"
    );

    printf("Summary of measured execution times:\n\n");

    for (int i = 0; i < group_count; i++) {
        Group *group = &groups[i];

        qsort(group->times, group->count,
              sizeof(double), compare_double);

        double q1 = quantile(group->times, group->count, 0.25);
        double median = quantile(group->times, group->count, 0.5);
        double q3 = quantile(group->times, group->count, 0.75);
        double iqr = q3 - q1;

        fprintf(
            output,
            "%s,%d,%d,%.12g,%.12g,%.12g,%.12g,%llu\n",
            group->language, group->n, group->count,
            median, q1, q3, iqr, group->payload_bytes
        );

        printf(
            "%s, n=%d: median=%.6f s, IQR=%.6f s, repetitions=%d\n",
            group->language, group->n, median, iqr, group->count
        );
    }

    fclose(output);

    for (int i = 0; i < group_count; i++) {
        free(groups[i].times);
    }

    free(groups);

    printf("\nSummary saved to data/summary.csv\n");
    return 0;
}
