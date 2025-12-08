import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Генерация данных
def generate_data(n=200, random_state=42):
    np.random.seed(random_state)

    centers = np.array([[3, 3], [9, 2], [9, 6]])
    covs = [np.diag([1.5, 1.5]), np.diag([1.0, 1.0]), np.diag([1.0, 1.0])]

    x_list, y_list = [], []
    for i, (c, cov) in enumerate(zip(centers, covs)):
        Xi = np.random.multivariate_normal(mean=c, cov=cov, size=n)
        yi = np.full(n, i)
        x_list.append(Xi)
        y_list.append(yi)

    x = np.vstack(x_list)
    y = np.concatenate(y_list)
    df = pd.DataFrame(x, columns=["x1", "x2"])
    df["label"] = y

    return x, y, df


# Кластеризация KMeans
def run_kmeans(X, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    return labels, centers


# Таблица соответствия между истинными метками и кластерами KMeans
def print_cluster_correspondence(y_true, kmeans_labels, df=None):
    """
    Создает и печатает таблицу соответствия между истинными метками
    и найденными кластерами KMeans.
    """
    # Создаем DataFrame для анализа соответствия
    correspondence_df = pd.DataFrame({
        'true_label': y_true,
        'kmeans_cluster': kmeans_labels
    })

    # Создаем кросс-таблицу (contingency table)
    cross_tab = pd.crosstab(
        correspondence_df['true_label'],
        correspondence_df['kmeans_cluster'],
        margins=True,
        margins_name="Total"
    )

    # Добавляем проценты по строкам
    row_percent = pd.crosstab(
        correspondence_df['true_label'],
        correspondence_df['kmeans_cluster'],
        normalize='index'
    ) * 100

    print("\n" + "=" * 60)
    print("ТАБЛИЦА СООТВЕТСТВИЯ: Истинные метки vs KMeans кластеры")
    print("=" * 60)

    print("\nАбсолютные частоты:")
    print(cross_tab.to_string())

    print("\n\nПроцентное распределение по строкам (истинные метки -> кластеры):")
    for true_label in row_percent.index:
        print(f"\nИстинная метка {true_label}:")
        for cluster in row_percent.columns:
            percent = row_percent.loc[true_label, cluster]
            if percent > 0:  # Показываем только ненулевые значения
                print(f"  -> Кластер {cluster}: {percent:.1f}%")

    # Определяем доминирующее соответствие
    print("\n\nДОМИНИРУЮЩЕЕ СООТВЕТСТВИЕ:")
    mapping = {}
    for true_label in np.unique(y_true):
        mask = y_true == true_label
        cluster_counts = pd.Series(kmeans_labels[mask]).value_counts()
        if len(cluster_counts) > 0:
            dominant_cluster = cluster_counts.idxmax()
            count = cluster_counts.max()
            total = mask.sum()
            percentage = 100 * count / total
            mapping[true_label] = dominant_cluster
            print(f"Истинная метка {true_label} -> Кластер {dominant_cluster}: "
                  f"{count}/{total} ({percentage:.1f}%)")

    # Проверка уникальности соответствия
    if len(set(mapping.values())) == len(mapping):
        print("\n✓ Соответствие взаимно-однозначное (биекция)")
    else:
        print("\n⚠ Внимание: несколько истинных меток соответствуют одному кластеру")

    return cross_tab, row_percent, mapping


# Визуализация кластеров
def plot_clusters(x, y_true, kmeans_labels, centers, filename="kmeans_result.png"):
    plt.figure(figsize=(12, 5))

    # Истинные метки
    plt.subplot(1, 2, 1)
    for lab in np.unique(y_true):
        mask = y_true == lab
        plt.scatter(x[mask, 0], x[mask, 1], s=15, label=f"true {lab}")
    plt.title("Истинные метки")
    plt.legend()

    # Найденные кластеры
    plt.subplot(1, 2, 2)
    for lab in np.unique(kmeans_labels):
        mask = kmeans_labels == lab
        plt.scatter(x[mask, 0], x[mask, 1], s=15, label=f"km {lab}")
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, edgecolor='k', linewidth=1.5, label='centers')
    plt.title("KMeans: найденные кластеры")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)


# Классификация k-NN и Naive Bayes
def classify_knn_gnb(x, y, test_size=0.3, random_state=1):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # k-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred_gnb = gnb.predict(x_test)

    y_train_knn = knn.predict(x_train)
    y_train_gnb = gnb.predict(x_train)

    return (x_test, y_test, y_pred_knn, y_pred_gnb, y_train_knn, y_train_gnb, x_train)


# Оценка классификации
def evaluate_classification(y_test, y_pred_knn, y_pred_gnb, save_csv="classification_results.csv"):
    report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
    report_gnb = classification_report(y_test, y_pred_gnb, output_dict=True)

    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_gnb = accuracy_score(y_test, y_pred_gnb)

    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_gnb = confusion_matrix(y_test, y_pred_gnb)

    results = pd.DataFrame({
        "method": ["k-NN (k=5)", "GaussianNB"],
        "accuracy": [acc_knn, acc_gnb],
        "precision_macro": [report_knn['macro avg']['precision'], report_gnb['macro avg']['precision']],
        "recall_macro": [report_knn['macro avg']['recall'], report_gnb['macro avg']['recall']],
        "f1_macro": [report_knn['macro avg']['f1-score'], report_gnb['macro avg']['f1-score']]
    })

    results.to_csv(save_csv, index=False)

    return results, cm_knn, cm_gnb


# Визуализация предсказаний классификаторов
def plot_classification_predictions(x_test, y_pred_knn, y_pred_gnb, filename="classification_predictions.png"):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for lab in np.unique(y_pred_knn):
        mask = y_pred_knn == lab
        plt.scatter(x_test[mask, 0], x_test[mask, 1], s=25, label=f"knn pred {lab}")
    plt.title("k-NN: предсказания на тесте")
    plt.legend()

    plt.subplot(1, 2, 2)
    for lab in np.unique(y_pred_gnb):
        mask = y_pred_gnb == lab
        plt.scatter(x_test[mask, 0], x_test[mask, 1], s=25, label=f"gnb pred {lab}")
    plt.title("GaussianNB: предсказания на тесте")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)


# Основной сценарий
if __name__ == "__main__":
    x, y_true, df = generate_data()

    kmeans_labels, centers_km = run_kmeans(x)
    plot_clusters(x, y_true, kmeans_labels, centers_km)

    # Добавлен вызов функции печати таблицы соответствия
    cross_tab, row_percent, mapping = print_cluster_correspondence(y_true, kmeans_labels, df)

    x_test, y_test, y_pred_knn, y_pred_gnb, y_train_knn, y_train_gnb, x_train = classify_knn_gnb(x, y_true)
    results, cm_knn, cm_gnb = evaluate_classification(y_test, y_pred_knn, y_pred_gnb)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
    print("=" * 60)
    print("\n=== Results table ===")
    print(results)

    print("\n=== Confusion matrix k-NN ===")
    print(cm_knn)

    print("\n=== Confusion matrix GaussianNB ===")
    print(cm_gnb)

    plot_classification_predictions(x_test, y_pred_knn, y_pred_gnb)
    plot_classification_predictions(x_train, y_train_knn, y_train_gnb, "train_result.png")
