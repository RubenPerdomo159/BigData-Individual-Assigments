import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RunBenchmarks {

    private static final Path JAVA_DIR = getJavaDir();

    private static Path getJavaDir() {
        try {
            return Path.of(
                RunBenchmarks.class.getProtectionDomain()
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

    private static String runCommand(List<String> command)
            throws IOException, InterruptedException {

        System.out.println("+ " + String.join(" ", command));

        Process process = new ProcessBuilder(command)
            .redirectErrorStream(true)
            .start();

        String output = new String(
            process.getInputStream().readAllBytes()
        );

        int exitCode = process.waitFor();

        if (exitCode != 0) {
            throw new RuntimeException(
                "Command failed:\n" + output
            );
        }

        return output;
    }

    public static void main(String[] args) throws Exception {
        List<Integer> sizes =
            new ArrayList<>(Arrays.asList(10, 30, 60, 100, 150));

        int repetitions = 5;
        int warmups = 1;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--sizes":
                    sizes.clear();

                    for (String value : args[++i].split(",")) {
                        sizes.add(Integer.parseInt(value.trim()));
                    }
                    break;

                case "--repetitions":
                    repetitions = Integer.parseInt(args[++i]);
                    break;

                case "--warmups":
                    warmups = Integer.parseInt(args[++i]);
                    break;

                default:
                    throw new IllegalArgumentException(
                        "Unknown argument: " + args[i]
                    );
            }
        }

        if (sizes.isEmpty()
                || sizes.stream().anyMatch(n -> n < 1)) {
            throw new IllegalArgumentException(
                "All matrix sizes must be positive."
            );
        }

        if (repetitions < 1 || warmups < 0) {
            throw new IllegalArgumentException(
                "Repetitions must be positive and "
                + "warmups cannot be negative."
            );
        }

        Files.createDirectories(DATA_DIR);

        String javaExecutable = Path.of(
            System.getProperty("java.home"),
            "bin",
            System.getProperty("os.name").toLowerCase().contains("win")
                ? "java.exe" : "java"
        ).toString();

        List<String> testCommand = List.of(
            javaExecutable,
            "-cp", JAVA_DIR.toString(),
            "MatrixBenchmark",
            "--test"
        );

        System.out.println("Running correctness tests...");
        System.out.println(runCommand(testCommand));

        List<String> allRows = new ArrayList<>();

        String header =
            "language,n,repetition,time_ns,"
            + "matrix_payload_bytes,checksum,correct";

        for (int n : sizes) {
            System.out.println(
                "\nBenchmarking Java with n=" + n + "..."
            );

            List<String> command = List.of(
                javaExecutable,
                "-cp", JAVA_DIR.toString(),
                "MatrixBenchmark",
                "--benchmark",
                "--n", String.valueOf(n),
                "--warmups", String.valueOf(warmups),
                "--repetitions", String.valueOf(repetitions)
            );

            String output = runCommand(command);
            List<String> lines = output.lines().toList();

            if (lines.size() != repetitions + 1
                    || !lines.get(0).equals(header)) {
                throw new RuntimeException(
                    "Unexpected benchmark output for n=" + n
                );
            }

            for (int row = 1; row < lines.size(); row++) {
                String line = lines.get(row);

                if (!line.endsWith(",true")) {
                    throw new RuntimeException(
                        "Correctness flag failed for n=" + n
                    );
                }

                allRows.add(line);
            }
        }

        Path outputPath = DATA_DIR.resolve("raw_measurements.csv");

        List<String> csv = new ArrayList<>();
        csv.add(header);
        csv.addAll(allRows);

        Files.write(outputPath, csv);

        String environment = String.join("\n",
            "Run timestamp: " + ZonedDateTime.now(),
            "Operating system: " + System.getProperty("os.name"),
            "Machine architecture: " + System.getProperty("os.arch"),
            "Processor information: not available through standard Java APIs",
            "Logical CPU count: "
                + Runtime.getRuntime().availableProcessors(),
            "Java version: " + System.getProperty("java.version"),
            "Java VM: " + System.getProperty("java.vm.name"),
            "Matrix sizes: " + sizes,
            "Warm-up runs per size: " + warmups,
            "Measured repetitions per size: " + repetitions,
            "Timing: multiplication kernel only.",
            "Memory: estimated matrix-element storage, not peak process RAM.",
            ""
        );

        Path environmentPath =
            DATA_DIR.resolve("environment.txt");

        Files.writeString(environmentPath, environment);

        System.out.println(
            "\nRaw measurements saved to: " + outputPath
        );

        System.out.println(
            "Environment details saved to: " + environmentPath
        );

        System.out.println(
            "Add your CPU model, physical core count and installed RAM "
            + "to environment.txt."
        );
    }
}
