from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
import statsmodels.regression.linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as pyplot
import pandas as pd


def improved_forward_selection(y, X_poly, feature_names, max_features=10):
    selected = []
    remaining = list(range(X_poly.shape[1]))
    results = []
    
    print(f"\nПроцесс прямого отбора (макс. {max_features} признаков):")
    print("-" * 60)
    
    for step in range(1, max_features + 1):
        best_feature = None
        best_score = -np.inf
        best_model = None
        
        for feature in remaining:
            candidate_features = selected + [feature]
            X_candidate = sm.add_constant(X_poly[:, candidate_features])
            
            model = sm.OLS(y, X_candidate).fit()
            score = model.rsquared_adj
            
            if score > best_score:
                best_score = score
                best_feature = feature
                best_model = model
        
        if best_feature is not None:
            selected.append(best_feature)
            remaining.remove(best_feature)
            
            feature_name = feature_names[best_feature]
            results.append({
                'step': step,
                'feature': feature_name,
                'R²': best_model.rsquared,
                'Adj R²': best_model.rsquared_adj,
                'AIC': best_model.aic,
                'BIC': best_model.bic
            })
            
            print(f"Шаг {step}: Добавлен {feature_name:15} | "
                  f"R² = {best_model.rsquared:.4f} | "
                  f"Adj R² = {best_model.rsquared_adj:.4f} | "
                  f"AIC = {best_model.aic:.2f}")
    
    X_final = sm.add_constant(X_poly[:, selected])
    final_model = sm.OLS(y, X_final).fit()
    
    return final_model, selected, results


def get_data(filename: str) -> tuple[pd.DataFrame, int, int]:
    assert filename[-4:] == ".csv"

    df = pd.read_csv(filename)

    min: int = df["year"].min()
    max: int = df["year"].max()

    print(f"Загружены данные с {min} по {max} год")
    start: int = int(input("Выберите начальный год: "))
    end: int = int(input("Выберите начальный год: "))

    assert min <= start < end <= max

    df = df[(df['year'] >= start) & (df['year'] <= end)]

    print(f"На выбранном отрезке: {(df['temp'] == 999.9).sum()} выбросов")

    year_mean = df["year"].mean()
    df["year_centered"] = df["year"] - year_mean

    return df, start, end


@dataclass
class AdditionalData:
    poly:               PolynomialFeatures
    selected_features:  list
    final_model:        lm.RegressionResults


def process_data(data: pd.DataFrame) -> AdditionalData:
    centered_df = pd.DataFrame({'year_centered': data['year_centered']})
    poly = PolynomialFeatures(degree=10, include_bias=False)
    X_poly = poly.fit_transform(centered_df)
    poly_features = poly.get_feature_names_out(['year_centered'])

    final_model, selected_features, _ = improved_forward_selection(
        data['temp'], X_poly, poly_features, max_features=10
    )

    print(f"\nВыбранные признаки: {[poly_features[i] for i in selected_features]}")
    print(f"R² финальной модели: {final_model.rsquared:.4f}")
    print(f"Скорректированный R² финальной модели: {final_model.rsquared_adj:.4f}")

    return AdditionalData(poly, selected_features, final_model)


def print_graph(data: pd.DataFrame, additional: AdditionalData, years: tuple[int, int]):
    pyplot.figure(figsize=(12, 8))

    pyplot.scatter(data["year"], data["temp"], 
            color='blue', s=60, alpha=0.7, label='Фактические данные', zorder=5)

    years_continuous = np.linspace(data["year"].min(), data["year"].max(), 300)
    years_continuous_centered = years_continuous - data["year"].mean()

    cont_df = pd.DataFrame({'year_centered': years_continuous_centered})
    X_cont_poly = additional.poly.transform(cont_df)
    X_cont_selected = sm.add_constant(X_cont_poly[:, additional.selected_features])
    y_cont_pred = additional.final_model.predict(X_cont_selected)

    pyplot.plot(years_continuous, y_cont_pred, 
            color='red', linewidth=2.5, 
            label=f'Полиномиальная подгонка (R² = {additional.final_model.rsquared:.3f})')

    predictions = additional.final_model.get_prediction(X_cont_selected)
    frame = predictions.summary_frame(alpha=0.05)
    pyplot.fill_between(years_continuous, 
                    frame['mean_ci_lower'], 
                    frame['mean_ci_upper'],
                    color='red', alpha=0.2, label='95% доверительный интервал')

    pyplot.xlabel('Год', fontsize=12)
    pyplot.ylabel('Среднегодовая температура (°C)', fontsize=12)
    pyplot.title(f'Полиномиальная регрессия с прямым отбором\nТемпературы ({years[0]}-{years[1]})', 
            fontsize=14, fontweight='bold')
    pyplot.legend(fontsize=10)
    pyplot.grid(True, alpha=0.3)
    pyplot.tight_layout()
    pyplot.show()


def main():
    data, start_year, end_year = get_data(input("Путь до файла: "))
    additional_data = process_data(data)
    print_graph(data, additional_data, (start_year, end_year))


if __name__ == "__main__":
    main()
