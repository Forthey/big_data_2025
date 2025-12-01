# Вывод формул
## Гребневая регрессия

Задача - минимизировать

$ Q_R(\beta) = \sum_{j=1}^p (y_j - \beta_j)^2 + \lambda \sum_{j=1}^p \beta_j^2 \to \min_{\beta} $

Аналогично с МНК, все координаты независимы, то есть достаточно для каждой j решить независимо

$ (y_j - \beta_j)^2 + \lambda \beta_j^2 \to \min_{\beta_j} $

$ d/d\beta_j: -2(y_j - \beta_j) + 2\lambda \beta_j = 0 \quad \Rightarrow \quad y_j - \beta_j + \lambda \beta_j = 0 \quad \Rightarrow \quad \beta_j (1 + \lambda) = y_j $

$ \hat{\beta}_j^{(r)} = \frac{y_j}{1 + \lambda} $

## Лассо регрессия

Задача - минимизировать

$ Q_L(\beta) = \sum_{j=1}^p (y_j - \beta_j)^2 + \lambda \sum_{j=1}^p |\beta_j| \to \min_{\beta} $

Аналогично - формула по j

$ f(\beta_j) = (y_j - \beta_j)^2 + \lambda |\beta_j| \to \min_{\beta_j \in \mathbb{R}} $

Функция $f(\beta_j)$ дифференцируема везде, кроме точки $\beta_j = 0$, где $|β_j|$ имеет "угол". Поэтому рассматриваем поддифференциал в точке $\beta_j = 0$ и обычные производные в остальных точках.

### Анализ производных по областям

#### Область $\beta_j > 0$:

$ f(\beta_j) = (y_j - \beta_j)^2 + \lambda \beta_j $

$ f'(\beta_j) = -2(y_j - \beta_j) + \lambda = 0 \quad \Rightarrow \quad \beta_j = y_j - \frac{\lambda}{2} $

Это критическая точка только если $ \beta_j = y_j - \frac{\lambda}{2} > 0 $, т.е. $ y_j > \frac{\lambda}{2} $.

#### Область $\beta_j < 0$:

$ f(\beta_j) = (y_j - \beta_j)^2 - \lambda \beta_j $

$ f'(\beta_j) = -2(y_j - \beta_j) - \lambda = 0 \quad \Rightarrow \quad \beta_j = y_j + \frac{\lambda}{2} $

Это критическая точка только если $ \beta_j = y_j + \frac{\lambda}{2} < 0$, т.е. $y_j < -\frac{\lambda}{2} $.

#### Точка $\beta_j = 0$ (недифференцируемая):

Поддифференциал в $\beta_j = 0$:

$ \partial f(0) = \partial[(y_j - 0)^2] + \partial[\lambda |0|] = \{0\} + [-\lambda, \lambda] = [-\lambda, \lambda] $

Точка $ \beta_j = 0 $ — минимум, если $ 0 \in \partial f(0) $, т.е. если $[-\lambda, \lambda]$ содержит 0, что всегда выполняется. Но для глобального минимума нужно сравнить значения функции.

### Анализ поведения функции

Рассмотрим три случая для $y_j$:

#### Случай 1: $|y_j| \leq \frac{\lambda}{2}$

В этом случае критические точки из областей либо не существуют, либо лежат вне соответствующих полуплоскостей. 

Проверим значение функции:

В $\beta_j = 0$: $f(0) = y_j^2$

Если бы существовала критическая точка $\beta_j = y_j - \frac{\lambda}{2} > 0$, то $f(y_j - \frac{\lambda}{2}) = \frac{\lambda^2}{4}$, но $y_j \leq \frac{\lambda}{2}$ $\Rightarrow$ $y_j - \frac{\lambda}{2} \leq 0$, противоречие.

Аналогично для отрицательной области. Следовательно, минимум достигается в $\beta_j = 0$.

#### Случай 2: $y_j > \frac{\lambda}{2}$

Критическая точка $\beta_j^* = y_j - \frac{\lambda}{2} > 0$ лежит в области $\beta_j > 0$. Сравним значения:

$$f(\beta_j^*) = f\left(y_j - \frac{\lambda}{2}\right) = \left(y_j - \left(y_j - \frac{\lambda}{2}\right)\right)^2 + \lambda \left(y_j - \frac{\lambda}{2}\right) = \frac{\lambda^2}{4}$$

$$f(0) = y_j^2$$

Поскольку $y_j > \frac{\lambda}{2}$, то $y_j^2 > \frac{\lambda^2}{4}$, поэтому $f(\beta_j^*) < f(0)$. Минимум в $\beta_j^* = y_j - \frac{\lambda}{2}$.

#### Случай 3: $y_j < -\frac{\lambda}{2}$

Аналогично: критическая точка $\beta_j^* = y_j + \frac{\lambda}{2} < 0$, и $f(\beta_j^*) = \frac{\lambda^2}{4} < y_j^2 = f(0)$. Минимум в $\beta_j^* = y_j + \frac{\lambda}{2}$.

Итоговое решение

$$\hat{\beta}_j^{(L)} = 
\begin{cases}
y_j - \frac{\lambda}{2},  & \text{если } y_j > \frac{\lambda}{2} \\
y_j + \frac{\lambda}{2},  & \text{если } y_j < -\frac{\lambda}{2} \\
0,                    & \text{если } |y_j| \leq \frac{\lambda}{2}
\end{cases}$$
