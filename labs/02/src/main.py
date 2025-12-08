import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


# 1. Генерация модельного ряда
def generate_series(K=500, h=0.1, seed=0):
    np.random.seed(seed)
    k = np.arange(K + 1)
    true_trend = 0.5 * np.sin(k * h)
    noise = np.random.normal(0, 1, size=k.size)
    x = true_trend + noise
    return k, true_trend, x


# 2. Экспоненциальное скользящее среднее
def compute_ema(series, alphas):
    return {alpha: pd.Series(series).ewm(alpha=alpha, adjust=False).mean() for alpha in alphas}


# 3. Сравнение трендов с точным значением
def compare_trends(true_trend, ema_dict):
    results = {}
    for alpha, ema in ema_dict.items():
        mse = mean_squared_error(true_trend, ema.values)
        corr = np.corrcoef(true_trend, ema.values)[0, 1]
        results[alpha] = dict(mse=mse, corr=corr)
    return results


# 4. Амплитудный спектр Фурье
def fourier_spectrum(series, h):
    n = len(series)
    fft_vals = np.fft.rfft(series)
    fft_freqs = np.fft.rfftfreq(n, d=h)
    amp = np.abs(fft_vals) / n
    main_idx = np.argmax(amp[1:]) + 1 if len(amp) > 1 else 0
    return fft_freqs, amp, fft_freqs[main_idx]


# 5. Статистические тесты остатков
def runs_test(residuals):
    med = np.median(residuals)
    signs = residuals > med
    runs = 1 + np.sum(signs[:-1] != signs[1:])
    n1 = np.sum(signs)
    n2 = len(signs) - n1
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    expected = 1 + (2 * n1 * n2) / (n1 + n2)
    var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    z = (runs - expected) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def test_residuals(series, ema_dict):
    results = {}
    for alpha, ema in ema_dict.items():
        resid = np.array(series) - ema.values
        t_stat, p_t = stats.ttest_1samp(resid, 0)
        sh_stat, p_sh = stats.shapiro(resid)
        lb = acorr_ljungbox(resid, lags=[10], return_df=True)
        lb_p = lb['lb_pvalue'].iloc[0]

        z_runs, p_runs = runs_test(resid)
        results[alpha] = dict(
            mean=resid.mean(), std=resid.std(ddof=1), p_t=p_t,
            p_sh=p_sh, ljung_p=lb_p, runs_p=p_runs
        )
    return results


def print_trend_comparison(k, x, true_trend, ema_dict, alphas, figsize=(14, 10)):
    n_alphas = len(alphas)
    fig, axes = plt.subplots(n_alphas + 1, 1, figsize=figsize)

    # Первый график: исходный ряд и истинный тренд
    ax = axes[0]
    ax.plot(k, x, 'b-', alpha=0.5, linewidth=0.8, label='Исходный ряд')
    ax.plot(k, true_trend, 'r-', linewidth=2, label='Истинный тренд')
    ax.set_title('Исходный временной ряд и истинный тренд', fontsize=12, fontweight='bold')
    ax.set_xlabel('Время (k)')
    ax.set_ylabel('Значение')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Графики для каждого значения alpha
    for i, alpha in enumerate(alphas, 1):
        ax = axes[i]
        ema = ema_dict[alpha]

        ax.plot(k, x, 'b-', alpha=0.3, linewidth=0.5, label='Исходный ряд')
        ax.plot(k, true_trend, 'r-', linewidth=1.5, label='Истинный тренд', alpha=0.7)
        ax.plot(k, ema.values, 'g-', linewidth=2, label=f'EMA (α={alpha})')

        # Добавим MSE и корреляцию на график
        mse = mean_squared_error(true_trend, ema.values)
        corr = np.corrcoef(true_trend, ema.values)[0, 1]

        text_str = f'MSE: {mse:.4f}\nCorr: {corr:.4f}'
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_title(f'Сравнение трендов для α={alpha}', fontsize=11)
        ax.set_xlabel('Время (k)')
        ax.set_ylabel('Значение')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("trend_comparison.png")

    # Дополнительно: график ошибок (разница между EMA и истинным трендом)
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    axes2 = axes2.flatten()

    for i, alpha in enumerate(alphas):
        ax = axes2[i]
        ema = ema_dict[alpha]
        error = ema.values - true_trend

        ax.plot(k, error, 'm-', linewidth=1, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.fill_between(k, 0, error, alpha=0.3, color='m')

        ax.set_title(f'Ошибка EMA (α={alpha})', fontsize=11)
        ax.set_xlabel('Время (k)')
        ax.set_ylabel('Ошибка')
        ax.grid(True, alpha=0.3)

        # Статистика ошибок
        mean_err = error.mean()
        std_err = error.std()
        ax.text(0.02, 0.98, f'Mean: {mean_err:.4f}\nStd: {std_err:.4f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.suptitle('Ошибки оценки тренда для различных α', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Главный сценарий
def main():
    k, true_trend, x = generate_series()
    alphas = [0.01, 0.05, 0.1, 0.3]

    ema_dict = compute_ema(x, alphas)
    comparison = compare_trends(true_trend, ema_dict)

    print("\nСравнение трендов (MSE и корреляция):")
    for a, r in comparison.items():
        print(f"alpha={a}: MSE={r['mse']:.4f}, corr={r['corr']:.4f}")

    print_trend_comparison(k, x, true_trend, ema_dict, alphas)

    freqs, amp, main_freq = fourier_spectrum(x, h=0.1)
    print(f"\nГлавная частота спектра: {main_freq:.5f}")

    residual_tests = test_residuals(x, ema_dict)
    print("\nТесты остатков:")
    for a, r in residual_tests.items():
        print(
            f"alpha={a}: mean={r['mean']:.4f}, std={r['std']:.4f}, p_t={r['p_t']:.4f}, p_sh={r['p_sh']:.4f}, ljung_p={r['ljung_p']}, runs_p={r['runs_p']:.4f}")


if __name__ == "__main__":
    main()
