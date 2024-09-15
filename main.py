from ClassGen import generate_case1_data, generate_case2_data, generate_case3_data
from trainer import train_and_evaluate

def main():
    print("Case 1:")
    X_train, y_train, X_test, y_test = generate_case1_data()
    train_and_evaluate(X_train, y_train, X_test, y_test, 2, use_gd=True)

    print("Case 2:")
    X_train, y_train, X_test, y_test = generate_case2_data()
    train_and_evaluate(X_train, y_train, X_test, y_test, 2, use_gd=True)

    #print("Case 3:")
    #X_train, y_train, X_test, y_test = generate_case3_data()
    #train_and_evaluate(X_train, y_train, X_test, y_test, 2)

if __name__ == "__main__":
    main()
