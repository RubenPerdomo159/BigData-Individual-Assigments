import java.util.Locale;

public class MatrixBenchmark {

    private static final int SEED = 7;

    private static double[][][] makeMatrices(int n) {
        double[][] a = new double[n][n];
        double[][] b = new double[n][n];
        double[][] c = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] =
                    (((i * 31 + j * 17 + SEED) % 101) - 50) / 50.0;

                b[i][j] =
                    (((i * 13 + j * 29 + SEED * 3) % 101) - 50) / 50.0;
            }
        }

        return new double[][][] {a, b, c};
    }

    private static void multiply(
            double[][] a, double[][] b, double[][] c) {

        int n = a.length;

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

    private static double checksum(double[][] c) {
        double total = 0.0;

        for (double[] row : c) {
            for (double value : row) {
                total += value;
            }
        }

        return total;
    }

    private static double[][] multiplyReference(
            double[][] a, double[][] b) {

        int n = a.length;
        double[][] result = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        return result;
    }

    private static void assertMatrixEqual(
            double[][] actual, double[][] expected) {

        double tolerance = 1e-9;

        if (actual.length != expected.length) {
            throw new AssertionError("Different number of rows");
        }

        for (int i = 0; i < expected.length; i++) {
            if (actual[i].length != expected[i].length) {
                throw new AssertionError("Different number of columns");
            }

            for (int j = 0; j < expected[i].length; j++) {
                double difference =
                    Math.abs(actual[i][j] - expected[i][j]);

                if (difference > tolerance) {
                    throw new AssertionError(
                        "Mismatch at (" + i + ", " + j + "): "
                        + actual[i][j] + " != " + expected[i][j]
                    );
                }
            }
        }
    }

    private static void runTests() {
        // Test 1: known result.
        double[][] a = {
            {1.0, 2.0},
            {3.0, 4.0}
        };

        double[][] b = {
            {5.0, 6.0},
            {7.0, 8.0}
        };

        double[][] c = new double[2][2];

        multiply(a, b, c);

        double[][] expected = {
            {19.0, 22.0},
            {43.0, 50.0}
        };

        assertMatrixEqual(c, expected);

        // Test 2: identity matrix.
        double[][] identity = {
            {1.0, 0.0},
            {0.0, 1.0}
        };

        c = new double[2][2];
        multiply(a, identity, c);
        assertMatrixEqual(c, a);

        // Test 3: zero matrix.
        double[][] zero = new double[2][2];

        c = new double[2][2];
        multiply(a, zero, c);
        assertMatrixEqual(c, zero);

        // Test 4: independent reference implementation.
        double[][][] matrices = makeMatrices(4);

        a = matrices[0];
        b = matrices[1];
        c = matrices[2];

        multiply(a, b, c);

        double[][] reference = multiplyReference(a, b);
        assertMatrixEqual(c, reference);

        System.out.println("PASS: all four correctness tests.");
    }

    private static void benchmark(
            int n, int warmups, int repetitions) {

        double[][][] matrices = makeMatrices(n);
        double[][] a = matrices[0];
        double[][] b = matrices[1];

        // Warm-up runs are not measured.
        for (int i = 0; i < warmups; i++) {
            double[][] c = new double[n][n];
            multiply(a, b, c);
            checksum(c);
        }

        // CSV header.
        System.out.println(
            "language,n,repetition,time_ns,"
            + "matrix_payload_bytes,checksum,correct"
        );

        long payloadBytes = 3L * n * n * 8L;

        for (int repetition = 1;
             repetition <= repetitions;
             repetition++) {

            double[][] c = new double[n][n];

            long start = System.nanoTime();
            multiply(a, b, c);
            long elapsed = System.nanoTime() - start;

            double resultChecksum = checksum(c);

            System.out.printf(
                Locale.US,
                "Java,%d,%d,%d,%d,%.17g,true%n",
                n,
                repetition,
                elapsed,
                payloadBytes,
                resultChecksum
            );
        }
    }

    public static void main(String[] args) {
        boolean test = false;
        boolean benchmark = false;

        int n = 10;
        int warmups = 1;
        int repetitions = 5;

        try {
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--test":
                        test = true;
                        break;

                    case "--benchmark":
                        benchmark = true;
                        break;

                    case "--n":
                        n = Integer.parseInt(args[++i]);
                        break;

                    case "--warmups":
                        warmups = Integer.parseInt(args[++i]);
                        break;

                    case "--repetitions":
                        repetitions = Integer.parseInt(args[++i]);
                        break;

                    default:
                        throw new IllegalArgumentException(
                            "Unknown argument: " + args[i]
                        );
                }
            }

            if (test) {
                runTests();
                return;
            }

            if (!benchmark) {
                throw new IllegalArgumentException(
                    "Use --test or --benchmark."
                );
            }

            if (n < 1 || warmups < 0 || repetitions < 1) {
                throw new IllegalArgumentException(
                    "n and repetitions must be positive; "
                    + "warmups cannot be negative."
                );
            }

            benchmark(n, warmups, repetitions);

        } catch (NumberFormatException e) {
            System.err.println("Numeric arguments must be integers.");
            System.exit(1);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.exit(1);
        }
    }
}
