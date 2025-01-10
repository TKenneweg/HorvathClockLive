from mlp import MethylationDataset
from config import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

if __name__ == "__main__":
    dataset = MethylationDataset(SERIES_NAMES, DATA_FOLDER)
    X_data = dataset.X.numpy()
    y_data = dataset.y.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    # model = LinearRegression()
    # model = Ridge(alpha=5e3)
    model = Lasso(alpha=1e-2)

    model.fit(X_train, y_train)
    print("Model coefficients:", model.coef_)
    num_large_coeffs = np.sum(np.abs(model.coef_) > 0.01)
    print(f"Number of coefficients larger than 0.01: {num_large_coeffs}")
    plt.figure()
    plt.scatter(range(len(model.coef_)), model.coef_, alpha=0.5)
    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")
    plt.title("Scatter Plot of Model Coefficients")
    plt.tight_layout()
    plt.savefig("coefficients_scatter_plot.png")
    plt.show()




    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    median = np.median(np.abs(preds - y_test))
    print(f"[RESULT] Test MAE: {mae:.2f}")
    print(f"[RESULT] Test Median Absolute Error: {median:.2f}")
    


    plt.figure()
    # plt.scatter(y_train, predstrain, alpha=0.5)
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Actual vs Predicted Age")
    plt.tight_layout()
    plt.savefig("age_scatter_plot.png")
    plt.show()
