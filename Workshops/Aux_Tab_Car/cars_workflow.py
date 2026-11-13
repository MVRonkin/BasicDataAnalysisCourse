"""Инструменты практики: все операции получают данные явным аргументом."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REFERENCE_YEAR = 2022  # Year + Age == 2022 в исходной таблице Cars.
NUMBERS = ["Year", "Distance", "Engine", "km_year", "AutoTransmission"]
CATEGORIES = ["Make", "Model", "Fuel_type", "Style"]
EXTRA_NAMES = ["log_distance", "model_frequency", "mileage_deviation"]
REQUIRED = ["Year", "Distance", "Engine", "AutoTransmission", *CATEGORIES]


def quality_report(data: pd.DataFrame) -> pd.DataFrame:
    """Проблемы данных; дорогие и редкие машины автоматически не удаляются."""
    numeric = data.reindex(columns=["Year", "Distance", "Engine", "Price"]).apply(pd.to_numeric, errors="coerce")
    counts = {
        "Строк": len(data),
        "Точных дублей": int(data.duplicated().sum()),
        "Пропущенных значений": int(data.isna().sum().sum()),
        "Отрицательный пробег": int(numeric["Distance"].lt(0).sum()),
        "Год вне 1900–2022": int((numeric["Year"].lt(1900) | numeric["Year"].gt(REFERENCE_YEAR)).sum()),
        "Отрицательный объём двигателя": int(numeric["Engine"].lt(0).sum()),
        "Неположительная цена": int(numeric["Price"].le(0).sum()),
    }
    return pd.DataFrame.from_dict(counts, orient="index", columns=["Количество"])


def clean_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Фиксированные правила без обучения статистик и без изменения исходника."""
    missing = set(REQUIRED).difference(data.columns)
    if missing:
        raise ValueError(f"Отсутствуют входные столбцы: {sorted(missing)}")
    result = data.copy()
    for column in ["Year", "Distance", "Engine"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result.loc[~result["Year"].between(1900, REFERENCE_YEAR), "Year"] = np.nan
    result.loc[result["Distance"].lt(0), "Distance"] = np.nan
    result.loc[result["Engine"].lt(0), "Engine"] = np.nan
    auto_codes = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "0": 0.0, "1": 1.0}
    result["AutoTransmission"] = result["AutoTransmission"].map(auto_codes).astype(float)
    age = (REFERENCE_YEAR - result["Year"]).clip(lower=1)
    result["km_year"] = result["Distance"] / age
    if "Age" in result:
        result["Age"] = REFERENCE_YEAR - result["Year"]
    for column in CATEGORIES:
        result[column] = result[column].fillna("Unknown").astype(str)
    return result


@dataclass
class Study:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    split_info: dict
    experiments: dict = field(default_factory=dict)
    opened_test: str | None = None
    test_result: dict | None = None
    selected_name: str | None = None
    preparation: pd.DataFrame | None = None

    def __repr__(self):
        return f"Study(train={len(self.train)}, valid={len(self.valid)}, test={len(self.test)}, опытов={len(self.experiments)})"

    def _repr_html_(self):
        if self.preparation is not None:
            return self.preparation.to_html()
        return pd.DataFrame([self.split_info]).to_html(index=False)


def split_data(data: pd.DataFrame, random_state: int = 42) -> Study:
    """60/20/20. Точные дубли удаляются до разбиения, чтобы не попасть в разные части."""
    unique = data.drop_duplicates().copy()
    price = pd.to_numeric(unique["Price"], errors="coerce")
    eligible = unique.loc[price.notna() & price.gt(0)].copy()
    eligible["Price"] = price.loc[eligible.index]
    train, holdout = train_test_split(eligible, test_size=0.4, random_state=random_state)
    valid, test = train_test_split(holdout, test_size=0.5, random_state=random_state + 1)
    info = {"Исходных строк": len(data), "Удалено точных дублей": len(data) - len(unique),
            "Исключено без корректной цели": len(unique) - len(eligible),
            "Train": len(train), "Validation": len(valid), "Test": len(test)}
    return Study(train.copy(), valid.copy(), test.copy(), info)


def prepare_data(study: Study) -> Study:
    """Применить фиксированные правила ко входам; медианы выучит Pipeline при fit(train)."""
    if study.preparation is not None:
        return study
    if study.experiments:
        raise ValueError("Подготовку выполните до обучения моделей в этом Study.")
    before = quality_report(study.train)
    study.train = clean_inputs(study.train)
    study.valid = clean_inputs(study.valid)
    study.test = clean_inputs(study.test)
    after = quality_report(study.train)
    study.preparation = before.join(after, lsuffix=" до", rsuffix=" после")
    return study


def eda_report(data: pd.DataFrame):
    """Квантили цены и гистограмма; данные заданы явным аргументом."""
    display(data["Price"].quantile([0, 0.5, 0.9, 0.99, 1]).rename("Цена, евро").to_frame())
    np.log1p(data["Price"]).plot.hist(bins=30, figsize=(7, 3.5), edgecolor="white")
    plt.xlabel("log1p(Price)")
    plt.ylabel("Число автомобилей")
    plt.title("Распределение цены в переданной выборке")
    plt.tight_layout()
    plt.show()


def transmission_report(data: pd.DataFrame, make: str = "Toyota"):
    """Общие средние и сравнение внутри заданной марки и пятилетнего года."""
    overall = data.groupby("AutoTransmission")["Price"].agg(["mean", "size"])
    display(overall.rename(columns={"mean": "Средняя цена", "size": "Автомобилей"}).round(0))
    subset = data.loc[data["Make"].eq(make)].copy()
    subset["Группа года"] = subset["Year"].floordiv(5).mul(5)
    groups = subset.groupby(["Группа года", "AutoTransmission"])["Price"].agg(["mean", "size"])
    display(groups.rename(columns={"mean": "Средняя цена", "size": "Автомобилей"}).round(0))
    means = groups["mean"].unstack("AutoTransmission")
    means.rename(columns={False: "Механика", True: "Автомат"}).plot.bar(figsize=(8, 3.5))
    plt.title(f"{make}: описательное сравнение средних")
    plt.ylabel("Цена, евро")
    plt.tight_layout()
    plt.show()


class PriceFeatures(TransformerMixin, BaseEstimator):
    """Три дополнительных признака; статистики fit только по переданному train."""
    def __init__(self, extended=False):
        self.extended = extended

    def fit(self, X, y=None):
        frame = clean_inputs(X)
        self.fitted_ids_ = frame.index.to_numpy().copy()
        if self.extended:
            self.frequency_ = frame.groupby(["Make", "Model"]).size() / len(frame)
            bands = frame["Year"].floordiv(5).mul(5)
            self.mileage_ = frame.assign(year_band=bands).groupby(["Make", "year_band"])["Distance"].median()
            self.default_mileage_ = float(frame["Distance"].median())
        return self

    def transform(self, X):
        frame = clean_inputs(X)
        result = frame[NUMBERS + CATEGORIES].copy()
        if self.extended:
            model_keys = pd.MultiIndex.from_frame(frame[["Make", "Model"]])
            bands = frame["Year"].floordiv(5).mul(5)
            mileage_keys = pd.MultiIndex.from_arrays([frame["Make"], bands])
            result["log_distance"] = np.log1p(frame["Distance"])
            result["model_frequency"] = self.frequency_.reindex(model_keys).fillna(0).to_numpy()
            typical = self.mileage_.reindex(mileage_keys).fillna(self.default_mileage_).to_numpy()
            result["mileage_deviation"] = frame["Distance"].to_numpy() - typical
        return result


def feature_preview(study: Study, count: int = 5) -> pd.DataFrame:
    """Новые столбцы для validation, статистики только из train."""
    maker = PriceFeatures(extended=True).fit(study.train)
    sample = study.valid.head(count)
    columns = ["Make", "Model", "Distance", *EXTRA_NAMES]
    return maker.transform(sample)[columns]


def augment_train(data: pd.DataFrame, fraction: float = 0.25, random_state: int = 42):
    """Локальная интерполяция пробега и цены; одинаковые остальные входы."""
    if not 0 <= fraction <= 0.5:
        raise ValueError("fraction должен быть от 0 до 0.5.")
    real = clean_inputs(data)
    group_columns = ["Make", "Model", "Year", "Engine", "Fuel_type", "Style", "AutoTransmission"]
    known = real[group_columns].notna().all(axis=1)
    known &= ~real[CATEGORIES].isin(["Rare", "Unknown"]).any(axis=1)
    eligible = real.loc[known]
    pairs = []
    for _, group in eligible.groupby(group_columns, dropna=False, sort=False):
        ids = group.sort_values("Distance").index.to_numpy()
        for left, right in zip(ids[:-1], ids[1:]):
            a, b = real.loc[left], real.loc[right]
            close = abs(a["Distance"] - b["Distance"]) <= 50000
            comparable_price = max(a["Price"], b["Price"]) <= 2 * min(a["Price"], b["Price"])
            if close and comparable_price and a["Distance"] != b["Distance"]:
                pairs.append((left, right))
    rng = np.random.default_rng(random_state)
    count = min(int(np.floor(len(real) * fraction)), len(pairs))
    selected = rng.choice(len(pairs), size=count, replace=False) if count else []
    synthetic, evidence = [], []
    for number in selected:
        left, right = pairs[number]
        a, b = real.loc[left], real.loc[right]
        weight = float(rng.uniform(0.25, 0.75))
        row = a.copy()
        row["Distance"] = weight * a["Distance"] + (1 - weight) * b["Distance"]
        row["Price"] = weight * a["Price"] + (1 - weight) * b["Price"]
        row["km_year"] = row["Distance"] / max(1, REFERENCE_YEAR - row["Year"])
        synthetic.append(row)
        evidence.append({"Make": a["Make"], "Model": a["Model"], "Year": a["Year"],
                         "id A": left, "id B": right, "Вес A": weight,
                         "Пробег A": a["Distance"], "Пробег B": b["Distance"], "Новый пробег": row["Distance"],
                         "Цена A": a["Price"], "Цена B": b["Price"], "Новая цена": row["Price"]})
    added = pd.DataFrame(synthetic, columns=real.columns)
    added.index = -np.arange(1, count + 1)
    combined = pd.concat([real, added]) if count else real.copy()
    audit = pd.DataFrame(evidence)
    report = {"Реальных строк": len(real), "Подходящих пар": len(pairs), "Синтетических строк": count,
              "Строк для fit": len(combined)}
    return combined, pd.DataFrame([report]), audit


def make_model(model: str = "rf", extended: bool = False):
    numbers = NUMBERS + (EXTRA_NAMES if extended else [])
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    preprocess = ColumnTransformer([
        ("numeric", numeric, numbers),
        ("category", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORIES),
    ])
    if model == "lr":
        regressor = LinearRegression()
    elif model == "rf":
        regressor = RandomForestRegressor(n_estimators=64, max_depth=16, min_samples_leaf=2,
                                          max_features=0.7, n_jobs=2, random_state=42)
    else:
        raise ValueError("model: 'lr' или 'rf'")
    pipeline = Pipeline([("features", PriceFeatures(extended)), ("preprocess", preprocess), ("model", regressor)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def metrics(actual, predicted):
    return {"MAE, евро": mean_absolute_error(actual, predicted),
            "RMSE, евро": root_mean_squared_error(actual, predicted)}


@dataclass
class Experiment:
    name: str
    pipeline: object
    scores: pd.DataFrame
    valid_prediction: np.ndarray
    train_ids: np.ndarray
    augmentation: pd.DataFrame
    synthetic_examples: pd.DataFrame

    def _repr_html_(self):
        return self.scores.round(0).to_html(index=False)

    def __repr__(self):
        return self.scores.round(0).to_string(index=False)


def run_experiment(study: Study, model: str = "rf", extended: bool = False, augment: bool = False) -> Experiment:
    """Один заранее заданный вариант; оценки только на validation."""
    if study.opened_test is not None:
        raise ValueError("Test уже открыт. В этом Study новые эксперименты запрещены.")
    name = f"{model.upper()} / log" + (" + признаки" if extended else "") + (" + аугментация" if augment else "")
    if name in study.experiments:
        return study.experiments[name]
    train = study.train
    report, examples = pd.DataFrame(), pd.DataFrame()
    if augment:
        train, report, examples = augment_train(train)
    started = time.perf_counter()
    estimator = make_model(model, extended)
    estimator.fit(train.drop(columns="Price"), train["Price"])
    prediction = estimator.predict(study.valid.drop(columns="Price"))
    row = {"Вариант": name, "Строк fit": len(train), **metrics(study.valid["Price"], prediction),
           "Время fit + predict, с": time.perf_counter() - started}
    result = Experiment(name, estimator, pd.DataFrame([row]), prediction, train.index.to_numpy(), report, examples)
    study.experiments[name] = result
    study.selected_name = None
    return result


def compare_models(study: Study, metric: str = "MAE") -> pd.DataFrame:
    if not study.experiments:
        raise ValueError("Сначала выполните run_experiment.")
    if metric not in {"MAE", "RMSE"}:
        raise ValueError("metric: MAE или RMSE")
    rows = pd.concat([x.scores for x in study.experiments.values()], ignore_index=True)
    return rows.sort_values(f"{metric}, евро", kind="stable").reset_index(drop=True)


def select_model(study: Study, metric: str = "MAE") -> Experiment:
    name = compare_models(study, metric).iloc[0]["Вариант"]
    if study.opened_test is not None and name != study.opened_test:
        raise ValueError("Выбор зафиксирован до открытия test; менять его после просмотра test нельзя.")
    study.selected_name = name
    return study.experiments[name]


def error_report(study: Study, experiment: Experiment, count: int = 5) -> pd.DataFrame:
    """Крупные ошибки на validation, не на test."""
    result = study.valid[["Make", "Model", "Year", "Distance", "Price"]].copy()
    result["Прогноз"] = experiment.valid_prediction
    result["Ошибка, евро"] = result["Прогноз"] - result["Price"]
    result["Модуль ошибки"] = result["Ошибка, евро"].abs()
    return result.nlargest(count, "Модуль ошибки")


def predict_prices(experiment: Experiment, data: pd.DataFrame) -> pd.DataFrame:
    """Весь обученный pipeline, включая генерацию признаков и обратный логарифм."""
    result = data.copy()
    result["Прогноз цены, евро"] = experiment.pipeline.predict(data.drop(columns="Price", errors="ignore"))
    return result


def evaluate_test(study: Study, experiment: Experiment) -> dict:
    if study.selected_name != experiment.name:
        raise ValueError("Сначала select_model(study): выбор только по validation.")
    if study.opened_test and study.opened_test != experiment.name:
        raise ValueError("Test уже открыт для другой модели. Выбирать по test нельзя.")
    if study.test_result is None:
        predicted = experiment.pipeline.predict(study.test.drop(columns="Price"))
        table = study.test[["Make", "Model", "Year", "Distance", "Price"]].copy()
        table["Прогноз"] = predicted
        table["Модуль ошибки"] = abs(table["Price"] - predicted)
        score = pd.DataFrame([{"Вариант": experiment.name, "Строк test": len(table), **metrics(table["Price"], predicted)}])
        study.test_result = {"scores": score, "predictions": table}
        study.opened_test = experiment.name
    return study.test_result


@dataclass
class PipelineResult:
    study: Study
    best: Experiment
    validation_scores: pd.DataFrame
    test_scores: pd.DataFrame
    test_predictions: pd.DataFrame
    artifacts: str

    def _repr_html_(self):
        return self.test_scores.round(0).to_html(index=False)


def full_pipeline(data: pd.DataFrame, metric: str = "MAE", output_dir: str = "artifacts") -> PipelineResult:
    """Повторить четыре опыта, выбрать только по validation, открыть test и сохранить результат."""
    study = split_data(data)
    prepare_data(study)
    run_experiment(study, model="lr")
    run_experiment(study, model="rf")
    run_experiment(study, model="rf", extended=True)
    run_experiment(study, model="rf", extended=True, augment=True)
    best = select_model(study, metric)
    scores = compare_models(study, metric)
    final = evaluate_test(study, best)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.pipeline, folder / "cars_price.joblib")
    scores.to_csv(folder / "validation_scores.csv", index=False)
    final["scores"].to_csv(folder / "test_scores.csv", index=False)
    final["predictions"].to_csv(folder / "test_predictions.csv", index_label="car_id")
    (folder / "run_info.json").write_text(json.dumps({"selection_metric": metric, "selected": best.name,
        "split": study.split_info, "reference_year": REFERENCE_YEAR}, ensure_ascii=False, indent=2), encoding="utf-8")
    return PipelineResult(study, best, scores, final["scores"], final["predictions"], str(folder))
