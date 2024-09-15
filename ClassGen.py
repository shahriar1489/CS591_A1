import numpy as np
import pandas as pd
def generate_case1_data():
    np.random.seed(24)
    case1_class1 = []
    while len(case1_class1) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        # Class 1: -x1 + x2 > 0
        filtered_samples = samples[samples[:, 1] > samples[:, 0]]
        case1_class1.extend(filtered_samples)

    case1_class1 = np.array(case1_class1[:100])

    case1_class2 = []
    while len(case1_class2) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        # Class 2: -x1 + x2 < 0
        filtered_samples = samples[samples[:, 1] < samples[:, 0]]
        case1_class2.extend(filtered_samples)

    case1_class2 = np.array(case1_class2[:100])

    X_train = np.vstack((case1_class1, case1_class2))
    y_train = np.hstack((np.ones(len(case1_class1)), -1 * np.ones(len(case1_class2))))

    case1_class1_test = []
    while len(case1_class1_test) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        filtered_samples = samples[samples[:, 1] > samples[:, 0]]
        case1_class1_test.extend(filtered_samples)

    case1_class1_test = np.array(case1_class1_test[:100])

    case1_class2_test = []
    while len(case1_class2_test) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        filtered_samples = samples[samples[:, 1] < samples[:, 0]]
        case1_class2_test.extend(filtered_samples)

    case1_class2_test = np.array(case1_class2_test[:100])

    X_test = np.vstack((case1_class1_test, case1_class2_test))
    y_test = np.hstack((np.ones(len(case1_class1_test)), -1 * np.ones(len(case1_class2_test))))

    return X_train, y_train, X_test, y_test


def generate_case2_data():
    np.random.seed(24)
    case2_class1 = []
    while len(case2_class1) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        # Class 1: x1 - 2x2 + 5 > 0
        filtered_samples = samples[samples[:, 0] - 2 * samples[:, 1] + 5 > 0]
        case2_class1.extend(filtered_samples)

    case2_class1 = np.array(case2_class1[:100])

    case2_class2 = []
    while len(case2_class2) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        # Class 2: x1 - 2x2 + 5 < 0
        filtered_samples = samples[samples[:, 0] - 2 * samples[:, 1] + 5 < 0]
        case2_class2.extend(filtered_samples)

    case2_class2 = np.array(case2_class2[:100])

    X_train = np.vstack((case2_class1, case2_class2))
    y_train = np.hstack((np.ones(len(case2_class1)), -1 * np.ones(len(case2_class2))))

    case2_class1_test = []
    while len(case2_class1_test) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        filtered_samples = samples[samples[:, 0] - 2 * samples[:, 1] + 5 > 0]
        case2_class1_test.extend(filtered_samples)

    case2_class1_test = np.array(case2_class1_test[:100])

    case2_class2_test = []
    while len(case2_class2_test) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        filtered_samples = samples[samples[:, 0] - 2 * samples[:, 1] + 5 < 0]
        case2_class2_test.extend(filtered_samples)

    case2_class2_test = np.array(case2_class2_test[:100])

    X_test = np.vstack((case2_class1_test, case2_class2_test))
    y_test = np.hstack((np.ones(len(case2_class1_test)), -1 * np.ones(len(case2_class2_test))))

    return X_train, y_train, X_test, y_test

def to_csv(X_train, y_train, X_test, y_test, case_number):
    train_df = pd.DataFrame(X_train, columns=['x1', 'x2'])
    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test, columns=['x1', 'x2'])
    test_df['label'] = y_test

    train_df.to_csv(f'case{case_number}_train.csv', index=False)
    test_df.to_csv(f'case{case_number}_test.csv', index=False)

def main():
    # Case 1
    X_train_1, y_train_1, X_test_1, y_test_1 = generate_case1_data()
    to_csv(X_train_1, y_train_1, X_test_1, y_test_1, case_number=1)

    # Case 2
    X_train_2, y_train_2, X_test_2, y_test_2 = generate_case2_data()
    to_csv(X_train_2, y_train_2, X_test_2, y_test_2, case_number=2)

if __name__ == "__main__":
    main()

def generate_case3_data():
    case3_class1 = []
    while len(case3_class1) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        # CLass 1: {(x1, x2, x3, x4) | 0.5x1 − x2 − 10x3 + x4 + 50 > 0}
        filtered_samples = samples[0.5 * samples[:, 0] - samples[:, 1], - 10 * samples[:, 2] + samples[:, 3] + 50 > 0]
        case3_class1.extend(filtered_samples)

    case3_class1 = np.array(case3_class1[:100])

    case3_class2 = []
    while len(case3_class2) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        # Class 2: {(x1, x2, x3, x4) | 0.5x1 − x2 − 10x3 + x4 + 50 < 0}
        filtered_samples = samples[0.5 * samples[:, 0] - samples[:, 1], - 10 * samples[:, 2] + samples[:, 3] + 50 < 0]
        case3_class2.extend(filtered_samples)

    case3_class2 = np.array(case3_class2[:100])

    X_train = np.vstack((case3_class1, case3_class2))
    y_train = np.hstack((np.ones(len(case3_class1)), -1 * np.ones(len(case3_class2))))

    case3_class1_test = []
    while len(case3_class1_test) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        filtered_samples = samples[0.5 * samples[:, 0] - samples[:, 1], - 10 * samples[:, 2] + samples[:, 3] + 50 > 0]
        case3_class1_test.extend(filtered_samples)

    case3_class1_test = np.array(case3_class1_test[:100])

    case3_class2_test = []
    while len(case3_class2_test) < 100:
        samples = np.random.uniform(-25, 25, (100, 2))
        filtered_samples = samples[0.5 * samples[:, 0] - samples[:, 1], - 10 * samples[:, 2] + samples[:, 3] + 50 < 0]
        case3_class2_test.extend(filtered_samples)

    case3_class2_test = np.array(case3_class2_test[:100])

    X_test = np.vstack((case3_class1_test, case3_class2_test))
    y_test = np.hstack((np.ones(len(case3_class1_test)), -1 * np.ones(len(case3_class2_test))))

    return X_train, y_train, X_test, y_test
