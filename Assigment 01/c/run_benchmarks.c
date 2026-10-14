#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

#ifdef _WIN32
    #include <direct.h>
    #define MAKE_DIR(path) _mkdir(path)
    #define OPEN_PIPE(command) _popen(command, "r")
    #define CLOSE_PIPE(pipe) _pclose(pipe)
    #define EXECUTABLE ".\\matrix_benchmark.exe"
#else
    #define MAKE_DIR(path) mkdir(path, 0777)
    #define OPEN_PIPE(command) popen(command, "r")
    #define CLOSE_PIPE(pipe) pclose(pipe)
    #define EXECUTABLE "./matrix_benchmark"
#endif

#define MAX_SIZES 100
#define MAX_LINE 1024

static int create_data_directory(void) {
    if (MAKE_DIR("data") == 0 || errno == EEXIST) {
        return 1;
    }

    perror("Could not create data directory");
    return 0;
}

static int run_tests(void) {
    char command[256];
    char line[MAX_LINE];

    snprintf(command, sizeof(command),
             "%s --test", EXECUTABLE);

    printf("Running correctness tests...\n");

    FILE *pipe = OPEN_PIPE(command);

    if (pipe == NULL) {
        fprintf(stderr, "Could not start matrix_benchmark.\n");
        return 0;
    }

    int passed = 0;

    while (fgets(line, sizeof(line), pipe) != NULL) {
        printf("%s", line);

        if (strstr(line, "PASS: all four correctness tests.")
                != NULL) {
            passed = 1;
        }
    }

    int status = CLOSE_PIPE(pipe);

    return passed && status == 0;
}

static int run_size(
    int n, int warmups, int repetitions, FILE *csv
) {
    char command[512];
    char line[MAX_LINE];

    snprintf(
        command, sizeof(command),
        "%s --benchmark --n %d --warmups %d --repetitions %d",
        EXECUTABLE, n, warmups, repetitions
    );

    printf("\nBenchmarking C with n=%d...\n", n);

    FILE *pipe = OPEN_PIPE(command);

    if (pipe == NULL) {
        fprintf(stderr, "Could not start benchmark for n=%d.\n", n);
        return 0;
    }

    int header_found = 0;
    int rows = 0;
    int valid = 1;

    while (fgets(line, sizeof(line), pipe) != NULL) {
        if (!header_found) {
            if (strcmp(
                    line,
                    "language,n,repetition,time_ns,"
                    "matrix_payload_bytes,checksum,correct\n"
                ) != 0) {
                fprintf(stderr, "Unexpected CSV header.\n");
                valid = 0;
                break;
            }

            header_found = 1;
            continue;
        }

        if (strstr(line, ",true") == NULL) {
            fprintf(stderr, "Incorrect benchmark result: %s", line);
            valid = 0;
            break;
        }

        fputs(line, csv);
        rows++;
    }

    int status = CLOSE_PIPE(pipe);

    if (!header_found || rows != repetitions || status != 0) {
        valid = 0;
    }

    if (!valid) {
        fprintf(stderr, "Benchmark failed for n=%d.\n", n);
    }

    return valid;
}

static int write_environment(
    const int sizes[], int count, int warmups, int repetitions
) {
    FILE *file = fopen("data/environment.txt", "w");

    if (file == NULL) {
        perror("Could not write environment.txt");
        return 0;
    }

    time_t now = time(NULL);
    char timestamp[64];

    struct tm *local_time = localtime(&now);

    if (local_time != NULL) {
        strftime(timestamp, sizeof(timestamp),
                 "%Y-%m-%d %H:%M:%S", local_time);
    } else {
        strcpy(timestamp, "not available");
    }

    fprintf(file, "Run timestamp: %s\n", timestamp);

#ifdef _WIN32
    fprintf(file, "Operating system: Windows\n");
#elif defined(__APPLE__)
    fprintf(file, "Operating system: macOS\n");
#elif defined(__linux__)
    fprintf(file, "Operating system: Linux\n");
#else
    fprintf(file, "Operating system: unknown\n");
#endif

#if defined(__x86_64__) || defined(_M_X64)
    fprintf(file, "Machine architecture: x86_64\n");
#elif defined(__aarch64__)
    fprintf(file, "Machine architecture: ARM64\n");
#elif defined(__i386__) || defined(_M_IX86)
    fprintf(file, "Machine architecture: x86\n");
#else
    fprintf(file, "Machine architecture: unknown\n");
#endif

    fprintf(file, "Processor information: add CPU model manually\n");
    fprintf(file, "Logical CPU count: add manually if needed\n");
    fprintf(file, "Language: C\n");
    fprintf(file, "Compiler: add compiler name and version manually\n");

    fprintf(file, "Matrix sizes:");

    for (int i = 0; i < count; i++) {
        fprintf(file, "%s%d", i == 0 ? " " : ", ", sizes[i]);
    }

    fprintf(file, "\nWarm-up runs per size: %d\n", warmups);
    fprintf(file, "Measured repetitions per size: %d\n", repetitions);
    fprintf(file, "Timing: multiplication kernel only.\n");
    fprintf(file,
            "Memory: estimated storage for three matrices; "
            "not peak process RAM.\n");

    fclose(file);
    return 1;
}

int main(int argc, char *argv[]) {
    int sizes[MAX_SIZES] = {10, 30, 60, 100, 150};
    int size_count = 5;
    int warmups = 1;
    int repetitions = 5;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            char *token = strtok(argv[++i], ",");
            size_count = 0;

            while (token != NULL) {
                if (size_count >= MAX_SIZES) {
                    fprintf(stderr, "Too many matrix sizes.\n");
                    return 1;
                }

                char *end;
                long value = strtol(token, &end, 10);

                if (*token == '\0' || *end != '\0'
                        || value < 1 || value > 100000) {
                    fprintf(stderr, "Invalid matrix size.\n");
                    return 1;
                }

                sizes[size_count++] = (int)value;
                token = strtok(NULL, ",");
            }
        } else if (strcmp(argv[i], "--repetitions") == 0
                   && i + 1 < argc) {
            repetitions = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmups") == 0
                   && i + 1 < argc) {
            warmups = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n",
                    argv[i]);
            return 1;
        }
    }

    if (size_count == 0 || repetitions < 1 || warmups < 0) {
        fprintf(stderr, "Invalid sizes, repetitions or warmups.\n");
        return 1;
    }

    for (int i = 0; i < size_count; i++) {
        if (sizes[i] < 1) {
            fprintf(stderr, "Matrix sizes must be positive.\n");
            return 1;
        }
    }

    if (!create_data_directory()) {
        return 1;
    }

    if (!run_tests()) {
        fprintf(stderr, "Correctness tests failed.\n");
        return 1;
    }

    FILE *csv = fopen("data/raw_measurements.csv", "w");

    if (csv == NULL) {
        perror("Could not create raw_measurements.csv");
        return 1;
    }

    fprintf(
        csv,
        "language,n,repetition,time_ns,"
        "matrix_payload_bytes,checksum,correct\n"
    );

    int success = 1;

    for (int i = 0; i < size_count; i++) {
        if (!run_size(sizes[i], warmups, repetitions, csv)) {
            success = 0;
            break;
        }
    }

    fclose(csv);

    if (!success) {
        fprintf(stderr, "Some benchmarks failed.\n");
        return 1;
    }

    if (!write_environment(sizes, size_count, warmups, repetitions)) {
        return 1;
    }

    printf("\nRaw measurements saved to data/raw_measurements.csv\n");
    printf("Environment saved to data/environment.txt\n");

    return 0;
}
