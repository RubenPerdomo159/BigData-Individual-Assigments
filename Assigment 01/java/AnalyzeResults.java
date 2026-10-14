import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AnalyzeResults {

    private static final Path JAVA_DIR = getJavaDir();

    private static Path getJavaDir() {
        try {
            return Path.of(
                AnalyzeResults.class.getProtectionDomain()
                    .getCodeSource()
                    .getLocation()
                    .toURI()
            ).toAbsolutePath().normalize();

        } catch (java.net.URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    private static final Path DATA_DIR =
        JAVA_DIR.resolve("data");

    private static double quantile(
            List<Double> values, double fraction) {

        List<Double> sorted = new ArrayList<>(values);
        sorted.sort(Double::compare);

        if (sorted.size() == 1) {
            return sorted.get(0);
        }

        double position = (sorted.size() - 1) * fraction;
        int lower = (int) position;
        int upper = Math.min(lower + 1, sorted.size() - 1);
        double weight = position - lower;

        return sorted.get(lower) * (1 - weight)
            + sorted.get(upper) * weight;
    }

    private static double median(List<Double> values) {
        return quantile(values, 0.5);
    }

    public static void main(String[] args) throws IOException {
        Path rawFile = DATA_DIR.resolve("raw_measurements.csv");
        Path summaryFile = DATA_DIR.resolve("summary.csv");

        if (!Files.exists(rawFile)) {
            throw new IllegalStateException(
                "Raw measurements not found. "
                + "Run RunBenchmarks first."
            );
        }

        List<String> lines = Files.readAllLines(rawFile);

        Map<String, List<Double>> groups = new HashMap<>();
        Map<String, Long> payloads = new HashMap<>();

        for (int i = 1; i < lines.size(); i++) {
            String[] row = lines.get(i).split(",");

            String language = row[0];
            int n = Integer.parseInt(row[1]);
            double seconds =
                Long.parseLong(row[3]) / 1_000_000_000.0;

            String key = language + "," + n;

            groups.computeIfAbsent(
                key, ignored -> new ArrayList<>()
            ).add(seconds);

            payloads.put(
                key, Long.parseLong(row[4])
            );
        }

        List<String> keys = new ArrayList<>(groups.keySet());

        keys.sort(Comparator
            .comparingInt((String key) ->
                Integer.parseInt(key.split(",")[1]))
            .thenComparing(key -> key.split(",")[0]));

        List<String> output = new ArrayList<>();

        output.add(
            "language,n,repetitions,median_seconds,"
            + "q1_seconds,q3_seconds,iqr_seconds,"
            + "matrix_payload_bytes"
        );

        System.out.println("Summary of measured execution times:\n");

        for (String key : keys) {
            String[] parts = key.split(",");
            String language = parts[0];
            int n = Integer.parseInt(parts[1]);

            List<Double> times = groups.get(key);

            double q1 = quantile(times, 0.25);
            double q3 = quantile(times, 0.75);
            double med = median(times);
            double iqr = q3 - q1;

            output.add(String.format(
                java.util.Locale.US,
                "%s,%d,%d,%.12g,%.12g,%.12g,%.12g,%d",
                language,
                n,
                times.size(),
                med,
                q1,
                q3,
                iqr,
                payloads.get(key)
            ));

            System.out.printf(
                java.util.Locale.US,
                "%s, n=%d: median=%.6f s, "
                + "IQR=%.6f s, repetitions=%d%n",
                language,
                n,
                med,
                iqr,
                times.size()
            );
        }

        Files.write(summaryFile, output);

        System.out.println(
            "\nSummary saved to: " + summaryFile
        );

        System.out.println(
            "This Java version generates the summary CSV. "
            + "Use the existing Python analysis script for the graph."
        );
    }
}
