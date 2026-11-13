"""Локальный агент с Python-инструментом. Служебный код практики."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import io
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from IPython.utils.capture import capture_output
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

import cars_workflow as workflow

API = """
Preloaded: pd, np, plt, display, df (original Cars DataFrame).
Tools (all already defined; do not reimplement):
quality_report(data) -> DataFrame of data quality counts.
eda_report(data) -> displays Price quantiles 0,0.5,0.9,0.99,1 and histogram of log1p(Price).
transmission_report(data, make='Toyota') -> displays overall transmission means/counts, within-make 5-year means/counts and chart.
split_data(df) -> Study with .train, .valid, .test, .split_info; removes exact duplicates before split; 60/20/20, fixed seed.
prepare_data(study) -> same prepared Study object, with .train/.valid/.test and .preparation (before/after quality table).
Typical preparation: work = prepare_data(work); display(work.preparation). Do this before training.
run_experiment(study, model='lr' or 'rf', extended=False, augment=False) -> Experiment.
Experiment fields: .scores (one-row DataFrame), .name, .augmentation (counts), .synthetic_examples (pair evidence), .pipeline.
Both algorithms predict log1p(Price), return euros; extended adds log_distance, model_frequency, mileage_deviation learned on train.
feature_preview(study, count=5) -> DataFrame of new feature values on validation, fitted on train.
augment_train(study.train) -> tuple (augmented_frame, counts_DataFrame, pairs_DataFrame); demonstration only, train unchanged.
compare_models(study, metric='MAE' or 'RMSE') -> DataFrame sorted by validation metric.
select_model(study, metric='MAE' or 'RMSE') -> best Experiment by validation.
error_report(study, experiment, count=5) -> biggest validation errors.
evaluate_test(study, experiment) -> dict with 'scores' and 'predictions' DataFrames. Only after selection, then no new experiments in that study.
predict_prices(experiment, data) -> DataFrame with prediction column; whole fitted pipeline.
full_pipeline(df, metric='MAE', output_dir='artifacts') -> PipelineResult: .best (Experiment), .validation_scores, .test_scores, .test_predictions, .artifacts.
full_pipeline runs LR, RF, RF+features, RF+features+augmentation; selects on validation; evaluates only selected model on test; saves pipeline and CSVs.
"""

SYSTEM = """You are a local data analyst agent working on Cars. Answer the user in Russian.
Execute real Python to obtain numbers. Output ONE Python code block per action, then WAIT for its result.
Complete related operations in one action when possible (assign results, display their tables). Do not add redundant calls or plots.
Use at most 8 short lines per action. Use the preloaded tools for ML. Do not redefine them.
After seeing results, either execute the next necessary action or finish with 'Готово.'.
Only if the task explicitly asks for interpretation, use at most 30 Russian words. Do not repeat tables in prose.
Follow the exact user request and metric. Do not invent numbers, model scores or conclusions.
Keep df unchanged. For modeling use work = split_data(df), and learn statistics only on work.train.
For EDA, ALWAYS pass work.train, never df. df may only be passed to split_data or full_pipeline.
Prefer eda_report(work.train) and transmission_report(work.train, make='Toyota') for corresponding EDA tasks.
Study is an object: work.train, NOT work['train']. Experiment is an object, NOT a dict.
Only inspect test after choosing by validation. Expensive or unusual cars are not automatically invalid.
Use existing variables when continuing. Do not repeat completed experiments unless asked.
If Python reports an error, fix the code and execute it. Do not finish after an error.
Use display(table) and plt.show() for tables and plots. All imports needed are already available.
Do not install packages, download files, call external services, or access the filesystem directly.
For a complete pipeline, use full_pipeline. Forecasts are calculated by the fitted regressor, never by you.
""" + API


@dataclass
class AgentRun:
    question: str
    trace: list
    answer: str
    seconds: float
    status: str

    def _repr_html_(self):
        calls = len(self.trace)
        return f"<small>Python-шагов: {calls} · Время: {self.seconds:.1f} с · Статус: {self.status}</small>"


class _StopCode(StoppingCriteria):
    def __init__(self, tokenizer, start):
        self.tokenizer, self.start = tokenizer, start

    def __call__(self, input_ids, scores, **kwargs):
        tail = input_ids[0, max(self.start, input_ids.shape[1] - 12):]
        found = "```" in self.tokenizer.decode(tail, skip_special_tokens=False)
        return torch.tensor([found], device=input_ids.device, dtype=torch.bool)


class AnalystAgent:
    def __init__(self, data: pd.DataFrame, model_path=None, threads=4):
        torch.set_num_threads(threads)
        checkpoint = model_path or "Qwen/Qwen3-4B-Instruct-2507"
        revision = None if model_path else "cdbee75f17c01a7cc42f958dc650907174af0554"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        self.llm = AutoModelForCausalLM.from_pretrained(
            checkpoint, revision=revision, torch_dtype=dtype,
            device_map={"": self.device}, low_cpu_mem_usage=True,
            attn_implementation="sdpa", use_safetensors=True,
        ).eval()
        print("Устройство:", self.device)
        self.namespace = {"pd": pd, "np": np, "plt": plt, "display": display, "df": data.copy()}
        names = ["quality_report", "eda_report", "transmission_report", "split_data", "prepare_data", "run_experiment", "feature_preview",
                 "augment_train", "compare_models", "select_model", "error_report", "evaluate_test",
                 "predict_prices", "full_pipeline"]
        self.namespace.update({name: getattr(workflow, name) for name in names})
        self._base_names = set(self.namespace)
        self.history = []

    def _state(self):
        lines = [f"df columns: {list(self.namespace['df'].columns)}; rows: {len(self.namespace['df'])}"]
        for name, value in list(self.namespace.items()):
            if name in self._base_names or name.startswith("_"):
                continue
            if isinstance(value, workflow.Study):
                text = repr(value)
            elif isinstance(value, workflow.Experiment):
                text = repr(value)
            elif isinstance(value, workflow.PipelineResult):
                text = f"PipelineResult(best={value.best.name}; fields best, validation_scores, test_scores, test_predictions, artifacts)"
            elif isinstance(value, pd.DataFrame):
                text = f"DataFrame shape={value.shape}, columns={list(value.columns)}"
            elif isinstance(value, (str, int, float, bool, dict)):
                text = str(value)[:250]
            else:
                text = type(value).__name__
            lines.append(f"{name}: {text}")
        return "\n".join(lines[-20:])

    def _generate(self, messages, force_python=False):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if force_python:
            prompt += "```python\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        length = inputs["input_ids"].shape[1]
        stops = StoppingCriteriaList([_StopCode(self.tokenizer, length)]) if force_python else None
        with torch.inference_mode():
            output = self.llm.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                repetition_penalty=1.05, stopping_criteria=stops,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated = output[0, length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        closed = force_python and "```" in text
        if len(generated) >= 512 and not closed:
            raise ValueError("Ответ слишком длинный. Разделите запрос на более короткие шаги.")
        if force_python:
            text = text.split("```", 1)[0].strip()
            return "```python\n" + text + "\n```"
        return text

    @staticmethod
    def _code(text):
        match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
        return match.group(1).strip() if match else None

    def _execute(self, code):
        if not code.strip():
            raise ValueError("Нужен непустой Python-расчёт для запроса.")
        tree = ast.parse(code)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                roots = [x.name.split('.')[0] for x in node.names] if isinstance(node, ast.Import) else [node.module.split('.')[0]]
                if not set(roots).issubset({'numpy', 'pandas', 'matplotlib', 'cars_workflow'}):
                    raise ValueError("Используйте предзагруженные pd, np, plt и инструменты практики.")
            if isinstance(node, ast.Name) and node.id == 'df':
                parent = parents.get(node)
                if isinstance(parent, ast.keyword):
                    parent = parents.get(parent)
                allowed = isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id in {'split_data', 'full_pipeline'}
                metadata = isinstance(parent, ast.Attribute) and parent.attr in {'columns', 'dtypes', 'shape'}
                if not (allowed or metadata):
                    raise ValueError("Для EDA используйте work.train. df разрешён только в split_data(df) и full_pipeline(df).")
            if isinstance(node, ast.Attribute) and node.attr == 'test' and isinstance(node.value, ast.Name):
                value = self.namespace.get(node.value.id)
                if isinstance(value, workflow.Study) and value.opened_test is None:
                    raise ValueError("Test закрыт: сначала выбор на validation, затем evaluate_test.")
            if isinstance(node, ast.Name) and node.id in {"open", "exec", "eval", "__import__", "compile", "input"}:
                raise ValueError("Этот инструмент выполняет только расчёты над таблицей.")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError("Доступ к внутренним атрибутам не требуется.")
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        source_df = self.namespace["df"].copy()
        try:
            with capture_output() as captured:
                exec(compile(tree, "<agent-python>", "exec"), self.namespace)
                if last:
                    value = eval(compile(ast.Expression(last.value), "<agent-python>", "eval"), self.namespace)
                    if value is not None:
                        display(value)
            if not self.namespace["df"].equals(source_df):
                self.namespace["df"] = source_df
                raise ValueError("Исходный df восстановлен: изменяйте рабочие копии, а не df.")
        except Exception:
            self.namespace["df"] = source_df
            raise
        observation = captured.stdout
        for output in captured.outputs:
            observation += "\n" + output.data.get("text/plain", "[График]")
        observation += "\nVariables after execution:\n" + self._state()
        return captured, observation[-5000:]

    def run(self, question: str, max_steps=5, show_code=True, show_answer=False) -> AgentRun:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Variables:\n{self._state()}\n\nTask: {question}"}]
        trace, answer = [], ""
        started = time.perf_counter()
        for number in range(max_steps):
            raw = self._generate(messages, force_python=not trace or not trace[-1]['ok'])
            code = self._code(raw)
            if code is None:
                if not trace or not trace[-1]['ok']:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": "Сначала выполни Python для этой задачи. Используй доступные инструменты."})
                    continue
                answer = raw
                if answer and show_answer:
                    display(Markdown(answer))
                result = AgentRun(question, trace, answer, time.perf_counter() - started, "finished")
                self.history.append(result)
                return result
            if show_code:
                display(Markdown(f"**Python · шаг {number + 1}**\n```python\n{code}\n```"))
            try:
                captured, observation = self._execute(code)
                captured.show()
                event = {"code": code, "ok": True, "observation": observation}
            except Exception as error:
                observation = f"{type(error).__name__}: {error}"
                display(Markdown(f"**Результат Python:** `{observation[:500]}`"))
                event = {"code": code, "ok": False, "observation": observation}
            trace.append(event)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Результат выполнения Python:\n" + observation + "\nПродолжи или кратко подведи итог."})
        result = AgentRun(question, trace, "Лимит шагов. Уточните задачу по журналу.", time.perf_counter() - started, "limit")
        display(Markdown(result.answer))
        self.history.append(result)
        return result

    def get(self, name):
        """Получить именованный результат из рабочей области агента."""
        return self.namespace[name]

    def set(self, name, value):
        """Явно передать агенту новую таблицу или параметр."""
        if name in self._base_names:
            raise ValueError("Выберите новое имя для передаваемого объекта.")
        self.namespace[name] = value

    def log(self):
        """Фактические действия и время, без дополнительной генерации LLM."""
        return pd.DataFrame([{"Запрос": x.question, "Python-шагов": len(x.trace),
                              "Ошибок Python": sum(not t["ok"] for t in x.trace),
                              "Время, с": round(x.seconds, 1), "Статус": x.status} for x in self.history])
