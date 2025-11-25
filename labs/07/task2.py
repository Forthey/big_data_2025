import numpy as np
import statsmodels.regression.linear_model as lm

from task1 import create_model


def get_model_statistic(model: lm.RegressionResults, size: int) -> float:
    RSS:            float = model.ssr
    df_residual:    int = size - 4
    RSE:            float = np.sqrt(RSS / df_residual)
    R_squared:      float = model.rsquared
    adj_R_squared:  float = model.rsquared_adj

    print("\nМетрики модели:")
    print(f"{"RSS (Сумма квадратов остатков):":<40} {RSS:.4f}")
    print(f"{"RSE (Стандартная ошибка остатков):":<40} {RSE:.4f}")
    print(f"{"R² (Коэффициент детерминации):":<40} {R_squared:.4f}")
    print(f"{"Скорректированный R²:":<40} {adj_R_squared:.4f}")

    return R_squared


def check_coeffs(model: lm.RegressionResults, R_squared: float):
    true_coeffs = [1, 3, -2, 1]
    estimated_coeffs = model.params
    
    print(f"Истинные коэффициенты: {", ".join(map(str, true_coeffs))}")
    print(f"Оцененные коэффициенты: {", ".join([f'{c:.4f}' for c in estimated_coeffs])}")

    print("\nОценка адекватности модели:")
    if R_squared > 0.9:
        print("R² > 0.9 указывает, что модель объясняет более 90% дисперсии")
        print("Линейная модель высоко адекватна для сгенерированных данных")
    else:
        print("Модель может нуждаться в улучшении")


def main():
    model, size = create_model()
    R_squared = get_model_statistic(model, size)
    check_coeffs(model, R_squared)


if __name__ == "__main__":
    main()
