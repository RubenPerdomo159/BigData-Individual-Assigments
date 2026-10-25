import random

SEED = 7

def make_sparse_matrices(n, sparsity, seed=SEED):
    """Create two matrices with a given zero percentage."""
    generator = random.Random(seed)

    a = []
    b = []

    for _ in range(n):
        row_a = []
        row_b = []

        for _ in range(n):
            if generator.random() < sparsity:
                row_a.append(0.0)
            else:
                row_a.append(generator.choice(
                                        (-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                                         1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
                                ) / 10.0)

            if generator.random() < sparsity:
                row_b.append(0.0)
            else:
                row_b.append(generator.choice(
                                        (-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                                         1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
                                ) / 10.0)

        a.append(row_a)
        b.append(row_b)

    return a, b

def dense_to_csr(matrix):
    """Convert a dense matrix to CSR storage."""
    values = []
    column_indices = []
    row_pointers = [0]

    for row in matrix:
        for column, value in enumerate(row):
            if value != 0.0:
                values.append(value)
                column_indices.append(column)

        row_pointers.append(len(values))

    return values, column_indices, row_pointers, len(matrix)

def csr_to_dense(csr):
    """Convert a CSR matrix to dense storage."""
    values, column_indices, row_pointers, n = csr
    matrix = [[0.0] * n for _ in range(n)]

    for row in range(n):
        for position in range(row_pointers[row], row_pointers[row + 1]):
            column = column_indices[position]
            matrix[row][column] = values[position]

    return matrix

def multiply_csr(a, b):
    """Multiply two CSR and return the result in CSR."""
    a_values, a_columns, a_rows, n = a
    b_values, b_columns, b_rows, _ = b

    result_values = []
    result_columns = []
    result_rows = [0]

    for i in range(n):
        row_result = {}

        for a_position in range(a_rows[i], a_rows[i + 1]):
            k = a_columns[a_position]
            a_value = a_values[a_position]

            for b_position in range(b_rows[k], b_rows[k + 1]):
                j = b_columns[b_position]
                b_value = b_values[b_position]

                if j not in row_result:
                    row_result[j] = 0.0

                row_result[j] += a_value * b_value

        for column in sorted(row_result):
            value = row_result[column]

            if value != 0.0:
                result_columns.append(column)
                result_values.append(value)

        result_rows.append(len(result_values))

    return result_values, result_columns, result_rows, n

def assert_matrix_equal(actual, expected, tolerance=1e-9):
    """Check whether matrices are equal within a tolerance."""
    if len(actual) != len(expected):
        raise AssertionError("Different number of rows")

    for i in range(len(expected)):
        if len(actual[i]) != len(expected[i]):
            raise AssertionError("Different number of columns")

        for j in range(len(expected[i])):
            if abs(actual[i][j] - expected[i][j]) > tolerance:
                raise AssertionError(
                    f"Mismatch at ({i}, {j}): "
                    f"{actual[i][j]} != {expected[i][j]}"
                )

def count_nonzeros(matrix):
    """Count nonzero elements in a dense matrix."""
    return sum(
        1
        for row in matrix
        for value in row
        if value != 0.0
    )

def run_tests():
    """Test CSR conversion and sparse multiplication."""
    a = [
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0]
    ]

    b = [
        [7.0, 0.0, 8.0],
        [0.0, 2.0, 0.0],
        [9.0, 0.0, 1.0]
    ]

    csr_a = dense_to_csr(a)
    csr_b = dense_to_csr(b)

    assert_matrix_equal(csr_to_dense(csr_a), a)
    assert_matrix_equal(csr_to_dense(csr_b), b)

    sparse_result = multiply_csr(csr_a, csr_b)
    sparse_dense = csr_to_dense(sparse_result)

    expected = [[0.0] * 3 for _ in range(3)]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                expected[i][j] += a[i][k] * b[k][j]

    assert_matrix_equal(sparse_dense, expected)

    print("PASS: all sparse matrix tests.")

if __name__ == "__main__":
    run_tests()
