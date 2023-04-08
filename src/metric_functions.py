import pandas as pd
from sklearn import metrics as m


def get_metrics_classification(y_test, y_pred, y_score, name) -> pd.DataFrame:
    """
    Генерация таблицы с метриками для задачи бинарной классификации
    :param y_test: истинные значения целевой переменной
    :param y_pred: предсказанный класс
    :param y_score: предсказанные вероятности классов
    :param name: имя модели
    :return: датафрейм с метриками
    """
    df_metrics = pd.DataFrame()

    df_metrics['model'] = [name]
    df_metrics['Precision'] = m.precision_score(y_test, y_pred)
    df_metrics['Recall'] = m.recall_score(y_test, y_pred)
    df_metrics['f1'] = m.f1_score(y_test, y_pred)
    df_metrics['ROC_AUC'] = m.roc_auc_score(y_test, y_score[:, 1])
    df_metrics['GINI'] = 2 * df_metrics['ROC_AUC'] - 1

    return df_metrics


def check_overfitting_classification(model, X_train, y_train, X_test, y_test) -> None:
    """
    Проверка на overfitting для бинарной классификации
    :param model: обученная модель
    :param X_train: матрица объект-признак train
    :param y_train: целевая переменная train
    :param X_test: матрица объект-признак test
    :param y_test: целевая переменная test
    :return: None
    """
    roc_auc_train = m.roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    roc_auc_test = m.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print("ROC-AUC train = %.3f" % roc_auc_train)
    print("ROC-AUC test = %.3f" % roc_auc_test)
    print(
        f'delta = {round((roc_auc_train - roc_auc_test) / roc_auc_test * 100, 2)}%'
    )


def get_metrics_multiclass(y_test_bin, y_test, y_pred, y_prob, name,
                           type_multi='ovo') -> pd.DataFrame:
    """
    Генерация таблицы с метриками для задачи мультиклассовой классификации
    :param y_test_bin: бинаризованные тестовые метки класса
    :param y_test: метки класса без бинаризации
    :param y_pred: предсказанный класс
    :param y_prob: предсказанные вероятности классов
    :param type_multi: тип многоклассовой классификации для ROC-AUC (ovo/ovr)
    :param name: имя модели
    :return: датафрейм с метриками
    """

    df_metrics = pd.DataFrame()

    df_metrics['model'] = [name]

    df_metrics['ROC_AUC'] = m.roc_auc_score(y_test_bin,
                                            y_prob,
                                            multi_class=type_multi)
    df_metrics['Precision_micro'] = m.precision_score(y_test,
                                                      y_pred,
                                                      average='micro')
    df_metrics['Precision_macro'] = m.precision_score(y_test,
                                                      y_pred,
                                                      average='macro')
    df_metrics['Recall_micro'] = m.recall_score(y_test, y_pred, average='micro')
    df_metrics['Recall_macro'] = m.recall_score(y_test, y_pred, average='macro')
    df_metrics['F1_micro'] = m.f1_score(y_test, y_pred, average='micro')
    df_metrics['F1_macro'] = m.f1_score(y_test, y_pred, average='macro')
    df_metrics['F1_weighted'] = m.f1_score(y_test, y_pred, average='weighted')
    df_metrics['Logloss'] = m.log_loss(y_test, y_prob)

    return df_metrics


def check_overfitting_multiclass(model, X_train, y_train, X_test, y_test):
    """
    Проверка на overfitting для мультиклассовой классификации
    :param model: обученная модель
    :param X_train: матрица объект-признак train
    :param y_train_bin: бинаризованная целевая переменная train
    :param X_test: матрица объект-признак test
    :param y_test_bin: бинаризованная целевая переменная test
    :return: None
    """
    f1_weighted_train = m.f1_score(y_train, model.predict(X_train), average='weighted')
    f1_weighted_test = m.f1_score(y_test, model.predict(X_test), average='weighted')

    print("f1-weighted train = %.3f" % f1_weighted_train)
    print("f1-weighted test = %.3f" % f1_weighted_test)
    print(
        f'delta = {round((f1_weighted_train - f1_weighted_test) / f1_weighted_test * 100, 2)}%'
    )
