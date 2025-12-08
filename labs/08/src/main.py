import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
import os
import pandas as pd

# Исходные данные
x = np.array([-2, -1, 0, 1, 2]).reshape(-1, 1)
y_true = np.array([-7, 0, 1, 2, 9])

# Точки для гладкого графика
x_plot = np.linspace(-2.5, 2.5, 200).reshape(-1, 1)

# Истинная функция: y = x^3 + 1
y_true_func = x_plot ** 3 + 1

# Список alpha
alphas = [0, 0.001, 0.01, 0.1, 1.0, 10.0]


# Функция для вывода коэффициентов
def print_coefficients(model, model_name, alpha, degree=11):
    """Вывод коэффициентов модели в читаемом формате"""

    # Получаем коэффициенты из модели
    if model_name == 'Ridge':
        coefs = model.named_steps['ridge'].coef_
        intercept = model.named_steps['ridge'].intercept_
    else:  # Lasso
        coefs = model.named_steps['lasso'].coef_
        intercept = model.named_steps['lasso'].intercept_

    # Создаем DataFrame для красивого вывода
    coef_dict = {'Степень': [], 'Коэффициент': [], '|Коэф|': []}

    # Добавляем свободный член
    coef_dict['Степень'].append('Intercept')
    coef_dict['Коэффициент'].append(intercept)
    coef_dict['|Коэф|'].append(abs(intercept))

    # Добавляем коэффициенты для степеней
    for i in range(degree + 1):
        if i == 0:  # intercept уже добавили
            continue
        coef_dict['Степень'].append(f'x^{i}')
        coef_dict['Коэффициент'].append(coefs[i - 1] if i - 1 < len(coefs) else 0)
        coef_dict['|Коэф|'].append(abs(coefs[i - 1] if i - 1 < len(coefs) else 0))

    df = pd.DataFrame(coef_dict)

    print(f"\n{'=' * 60}")
    print(f"Модель: {model_name}, α = {alpha}")
    print(f"{'=' * 60}")

    # Выводим только ненулевые коэффициенты или первые несколько
    print("Коэффициенты (ненулевые или значимые):")
    print(df[df['|Коэф|'] > 1e-6].to_string(index=False))

    # Сводная статистика
    print(f"\nСводная информация:")
    print(f"Количество ненулевых коэффициентов: {(df['|Коэф|'] > 1e-6).sum() - 1}")
    print(f"Сумма |коэффициентов|: {df['|Коэф|'].sum():.4f}")
    print(f"Максимальный |коэффициент|: {df['|Коэф|'].max():.4f}")

    return df


# Задача 1: Без шума

print("ЗАДАЧА 1: БЕЗ ШУМА")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(19, 11))
axes = axes.ravel()

for idx, alpha in enumerate(alphas):
    ax = axes[idx]

    # Ridge модель
    ridge_model = make_pipeline(PolynomialFeatures(11), Ridge(alpha=max(alpha, 1e-15)))  # защита от alpha=0
    ridge_model.fit(x, y_true)
    y_ridge = ridge_model.predict(x_plot)
    ax.plot(x_plot, y_ridge, color='blue', lw=2.7, label='Ridge', alpha=0.9)

    if alpha == 0:
        ax.plot(x_plot, y_ridge, color='green', lw=2.7, linestyle='--', label='МНК (Lasso не поддерживает α=0)',
                alpha=0.9)
        title = 'α = 0 (МНК)'

        # Вывод коэффициентов для Ridge с alpha=0
        print_coefficients(ridge_model, 'Ridge', alpha)
    else:
        # Lasso модель
        lasso_model = make_pipeline(
            PolynomialFeatures(11),
            Lasso(alpha=alpha, max_iter=100000, tol=1e-5, warm_start=True)
        )
        lasso_model.fit(x, y_true)
        y_lasso = lasso_model.predict(x_plot)
        ax.plot(x_plot, y_lasso, color='green', lw=2.5, linestyle='--', label='Lasso', alpha=0.85)
        title = f'α = {alpha}'

        # Вывод коэффициентов для обеих моделей
        print_coefficients(ridge_model, 'Ridge', alpha)
        print_coefficients(lasso_model, 'Lasso', alpha)

    ax.scatter(x, y_true, color='red', s=100, zorder=5, edgecolors='k', label='Данные')
    ax.plot(x_plot, x_plot.ravel() ** 3 + 1, 'k:', lw=2.5, label='Истинная: x³+1')

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylim(-12, 16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

plt.suptitle("Задача 1: Сравнение Ridge и Lasso при p=11 (без шума)", fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("plots/task1_comparison.png", dpi=300, bbox_inches='tight')

# Задача 2: С шумом
noises = [0.1, 0.2, 0.3]
np.random.seed(42)  # для воспроизводимости

for noise_std in noises:
    y_noisy = y_true + np.random.normal(0, noise_std, size=y_true.shape)

    print(f"\n\n{'=' * 60}")
    print(f"ЗАДАЧА 2: С ШУМОМ N(0, {noise_std})")
    print(f"{'=' * 60}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]

        # Ridge
        ridge_pipe = make_pipeline(PolynomialFeatures(11), Ridge(alpha=alpha))
        ridge_pipe.fit(x, y_noisy)
        y_ridge = ridge_pipe.predict(x_plot)

        ax.scatter(x, y_noisy, color='red', s=80, label='Данные с шумом', zorder=5)
        ax.plot(x_plot, y_true_func, 'k--', lw=2, label='Истинная: x³+1')
        ax.plot(x_plot, y_ridge, color='blue', lw=2.5, label='Ridge')

        # Lasso (кроме alpha=0)
        if alpha > 0:
            lasso_pipe = make_pipeline(PolynomialFeatures(11),
                                       Lasso(alpha=alpha, max_iter=20000, tol=1e-4))
            lasso_pipe.fit(x, y_noisy)
            y_lasso = lasso_pipe.predict(x_plot)
            ax.plot(x_plot, y_lasso, color='green', lw=2, alpha=0.8, label='Lasso')

            # Вывод коэффициентов для шумных данных
            print(f"\n{'=' * 60}")
            print(f"Шум N(0, {noise_std}), α = {alpha}")
            print_coefficients(ridge_pipe, 'Ridge', alpha)
            print_coefficients(lasso_pipe, 'Lasso', alpha)
        else:
            print(f"\n{'=' * 60}")
            print(f"Шум N(0, {noise_std}), α = {alpha} (только Ridge)")
            print_coefficients(ridge_pipe, 'Ridge', alpha)

        ax.set_title(f"α = {alpha}" + (" (Ridge только)" if alpha == 0 else " (Ridge + Lasso)"))
        ax.set_ylim(-15, 20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.suptitle(f"Задача 2: Шум N(0, {noise_std}) — Ridge и Lasso (p=11)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"plots/task2_noise_{noise_std}.png", dpi=300, bbox_inches='tight')

# Дополнительный анализ: сравнение влияния alpha на коэффициенты
print("\n\n" + "=" * 60)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ КОЭФФИЦИЕНТОВ")
print("=" * 60)

# Анализ для одного уровня шума (например, 0.1)
noise_std = 0.1
y_noisy = y_true + np.random.normal(0, noise_std, size=y_true.shape)

print(f"\nАнализ для шума N(0, {noise_std}):")
print("-" * 40)

# Создаем DataFrame для сравнения коэффициентов
coef_comparison = pd.DataFrame()

for alpha in alphas:
    if alpha == 0:
        model = make_pipeline(PolynomialFeatures(11), Ridge(alpha=1e-15))
        model_name = f"Ridge_α={alpha}"
    else:
        model = make_pipeline(PolynomialFeatures(11), Ridge(alpha=alpha))
        model_name = f"Ridge_α={alpha}"

    model.fit(x, y_noisy)
    coefs = model.named_steps['ridge'].coef_
    intercept = model.named_steps['ridge'].intercept_

    # Добавляем в DataFrame
    for i in range(min(12, len(coefs) + 1)):  # первые 12 коэффициентов
        if i == 0:
            coef_comparison.loc['Intercept', model_name] = intercept
        else:
            coef_comparison.loc[f'x^{i - 1}', model_name] = coefs[i - 1] if i - 1 < len(coefs) else 0

print("\nСравнение коэффициентов Ridge для разных α (первые 6 степеней):")
print(coef_comparison.head(12))

coef_comparison_lasso = pd.DataFrame()

for alpha in alphas:
    if alpha == 0:
        continue  # Lasso не поддерживает alpha=0

    model = make_pipeline(PolynomialFeatures(11),
                          Lasso(alpha=alpha, max_iter=20000, tol=1e-4))
    model.fit(x, y_noisy)
    coefs = model.named_steps['lasso'].coef_
    intercept = model.named_steps['lasso'].intercept_

    # Добавляем в DataFrame
    model_name = f"Lasso_α={alpha}"
    coef_comparison_lasso.loc['Intercept', model_name] = intercept

    for i in range(1, 13):  # первые 12 степеней (1-12)
        if i - 1 < len(coefs):
            coef_comparison_lasso.loc[f'x^{i - 1}', model_name] = coefs[i - 1]
        else:
            coef_comparison_lasso.loc[f'x^{i - 1}', model_name] = 0

print("\nСравнение коэффициентов Lasso для разных α (первые 6 степеней):")
print(coef_comparison_lasso.head(12))
