import numpy as np
import statsmodels.api as sm
import statsmodels.regression.linear_model as lm
import warnings


np.random.seed(18022004)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def create_model(size: int = 200) -> tuple[lm.RegressionResults, int]:
    X = np.random.normal(0, 1, (size, 3))
    e = np.random.normal(0, 1, size)
    y = 1 + 3 * X[:, 0] - 2 * X[:, 1] + X[:, 2] + e

    X_const = sm.add_constant(X)
    model: lm.RegressionResults = sm.OLS(y, X_const).fit()

    return model, size


def main():
    model, _ = create_model()
    print(model.summary())


if __name__ == "__main__":
    main()