"""
uv run python scripts/plot_sgeppy_active_terms_pareto.py \
  --input-root output/sgeppy_better_sweep \
  --output-dir output/sgeppy_better_sweep/pareto
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_MODELS = ("gt", "hw", "ih", "nh2", "nh4")
K_PATTERN = re.compile(r"k_(\d+)$")
LP_PATTERN = re.compile(r"lp_(.+)$")
LATEX_NAMES = {
    "K1": "K_1",
    "K2": "K_2",
    "Jm1": "J - 1",
    "logI13": r"\log(I_1/3)",
    "logI23": r"\log(I_2/3)",
}
LATEX_FUNCTIONS = {
    "sin": r"\sin",
    "cos": r"\cos",
    "log": r"\log",
}


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _k_from_path(path: Path) -> int:
    for parent in path.parents:
        match = K_PATTERN.match(parent.name)
        if match:
            return int(match.group(1))
    raise ValueError("Could not infer active-term epsilon from path: %s" % path)


def _lp_label_from_path(path: Path) -> str:
    for parent in path.parents:
        match = LP_PATTERN.match(parent.name)
        if match:
            return match.group(1)
    return ""


def _config_model_name(input_root: Path, summary_path: Path) -> str:
    try:
        return summary_path.relative_to(input_root).parts[0]
    except ValueError:
        return summary_path.parents[2].name


def _read_summary(input_root: Path, summary_path: Path) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    config = payload.get("config", {})
    model_config = config.get("model", {})
    weak_config = config.get("weak_form", {})
    epsilons = model_config.get("epsilons") or []
    k = int(float(epsilons[0])) if epsilons and epsilons[0] is not None else _k_from_path(summary_path)
    active_terms = int(metrics.get("num_parameters", 0))
    penalty_lp = float(weak_config.get("penalty_lp", "nan"))
    return {
        "model": _config_model_name(input_root, summary_path),
        "k": k,
        "penalty_lp": penalty_lp,
        "penalty_lp_label": _lp_label_from_path(summary_path),
        "active_terms": active_terms,
        "rmse": float(metrics.get("rmse", float("inf"))),
        "rss": float(metrics.get("rss", float("inf"))),
        "aic": float(metrics.get("aic", float("inf"))),
        "aicc": float(metrics.get("aicc", float("inf"))),
        "feasible": active_terms <= k,
        "best_expression": payload.get("best_expression", ""),
        "summary_path": str(summary_path),
    }


def load_rows(input_root: Path, models: Iterable[str]) -> list[dict]:
    rows = []
    for model in models:
        model_dir = input_root / model
        for summary_path in sorted(model_dir.glob("k_*/lp_*/summary.json")):
            rows.append(_read_summary(input_root, summary_path))
    return rows


def _is_dominated(row: dict, candidates: list[dict]) -> bool:
    for other in candidates:
        if other is row:
            continue
        no_worse = other["active_terms"] <= row["active_terms"] and other["rmse"] <= row["rmse"]
        strictly_better = other["active_terms"] < row["active_terms"] or other["rmse"] < row["rmse"]
        if no_worse and strictly_better:
            return True
    return False


def mark_pareto(rows: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    for model_rows in by_model.values():
        for row in model_rows:
            row["pareto"] = not _is_dominated(row, model_rows)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "k",
        "penalty_lp",
        "penalty_lp_label",
        "active_terms",
        "rmse",
        "rss",
        "aic",
        "aicc",
        "feasible",
        "pareto",
        "best_expression",
        "summary_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _strip_outer_parens(text: str) -> str:
    if text.startswith(r"\left(") and text.endswith(r"\right)"):
        return text[6:-7]
    return text


def _latex_node(node: ast.AST) -> str:
    if isinstance(node, ast.Expression):
        return _latex_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return "%.8g" % node.value
        return str(node.value)
    if isinstance(node, ast.Name):
        return LATEX_NAMES.get(node.id, node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return r"-\left(%s\right)" % _strip_outer_parens(_latex_node(node.operand))
    if isinstance(node, ast.BinOp):
        left = _latex_node(node.left)
        right = _latex_node(node.right)
        if isinstance(node.op, ast.Add):
            return r"\left(%s + %s\right)" % (left, right)
        if isinstance(node.op, ast.Sub):
            return r"\left(%s - %s\right)" % (left, right)
        if isinstance(node.op, ast.Mult):
            return r"\left(%s\,%s\right)" % (left, right)
        if isinstance(node.op, ast.Div):
            return r"\frac{%s}{%s}" % (_strip_outer_parens(left), _strip_outer_parens(right))
        if isinstance(node.op, ast.Pow):
            return r"\left(%s\right)^{%s}" % (_strip_outer_parens(left), _strip_outer_parens(right))
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else "f"
        args = [_latex_node(arg) for arg in node.args]
        if name == "square" and len(args) == 1:
            return r"\left(%s\right)^2" % _strip_outer_parens(args[0])
        if name == "sqrt" and len(args) == 1:
            return r"\sqrt{%s}" % _strip_outer_parens(args[0])
        if name == "exp" and len(args) == 1:
            return r"\exp\left(%s\right)" % _strip_outer_parens(args[0])
        if name in LATEX_FUNCTIONS and len(args) == 1:
            return r"%s\left(%s\right)" % (LATEX_FUNCTIONS[name], _strip_outer_parens(args[0]))
        return r"%s\left(%s\right)" % (name, ", ".join(_strip_outer_parens(arg) for arg in args))
    return ast.unparse(node)


def expression_to_latex(expression: str) -> str:
    if not expression:
        return ""
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        latex = expression
        for name, replacement in LATEX_NAMES.items():
            latex = re.sub(r"\b%s\b" % re.escape(name), replacement, latex)
        latex = latex.replace("*", r"\,")
        return latex
    return _strip_outer_parens(_latex_node(parsed))


def write_expression_report(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# SGEPPY Active-Term Best Expressions",
        "",
        "| Model | k | penalty_lp | Active terms | RMSE | Pareto | Expression |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        latex_expression = expression_to_latex(row["best_expression"])
        md_lines.append(
            "| {model} | {k} | {penalty_lp:.6g} | {active_terms} | {rmse:.6e} | {pareto} | ${expr}$ |".format(
                expr=latex_expression,
                **row,
            )
        )
    (output_dir / "best_expressions.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _run_label(row: dict) -> str:
    return "k%d lp%s" % (row["k"], "%.3g" % row["penalty_lp"])


def _plot_model(rows: list[dict], model: str, output_path: Path, log_y: bool) -> None:
    model_rows = sorted(
        [row for row in rows if row["model"] == model],
        key=lambda row: (row["active_terms"], row["rmse"], row["k"], row["penalty_lp"]),
    )
    if not model_rows:
        return

    pareto_rows = sorted(
        [row for row in model_rows if row["pareto"]],
        key=lambda row: (row["active_terms"], row["rmse"]),
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.scatter(
        [row["active_terms"] for row in model_rows],
        [row["rmse"] for row in model_rows],
        s=70,
        alpha=0.65,
        label="sweep run",
    )
    if pareto_rows:
        ax.plot(
            [row["active_terms"] for row in pareto_rows],
            [row["rmse"] for row in pareto_rows],
            marker="o",
            linewidth=2.0,
            label="Pareto front",
        )
    for row in model_rows:
        ax.annotate(
            _run_label(row),
            (row["active_terms"], row["rmse"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_title("%s SGEPPY Active-Term Pareto Front" % model.upper())
    ax.set_xlabel("Active terms")
    ax.set_ylabel("RMSE")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_combined(rows: list[dict], output_path: Path, log_y: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for model in sorted({row["model"] for row in rows}):
        model_rows = sorted(
            [row for row in rows if row["model"] == model],
            key=lambda row: (row["active_terms"], row["rmse"]),
        )
        pareto_rows = sorted(
            [row for row in model_rows if row["pareto"]],
            key=lambda row: (row["active_terms"], row["rmse"]),
        )
        ax.scatter(
            [row["active_terms"] for row in model_rows],
            [row["rmse"] for row in model_rows],
            alpha=0.35,
        )
        if pareto_rows:
            ax.plot(
                [row["active_terms"] for row in pareto_rows],
                [row["rmse"] for row in pareto_rows],
                marker="o",
                linewidth=2.0,
                label=model.upper(),
            )
    ax.set_title("SGEPPY Active-Term Pareto Fronts")
    ax.set_xlabel("Active terms")
    ax.set_ylabel("RMSE")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Model")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate and plot SGEPPY active-term sweep Pareto fronts.")
    parser.add_argument("--input-root", default="output/sgeppy_active_terms")
    parser.add_argument("--output-dir", default="output/sgeppy_active_terms/pareto")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--linear-y", action="store_true", help="Use a linear RMSE axis instead of log scale.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    rows = mark_pareto(load_rows(input_root, _parse_csv_strings(args.models)))
    if not rows:
        raise FileNotFoundError("No SGEPPY active-term summary files found under %s." % input_root)

    rows.sort(key=lambda row: (row["model"], row["k"], row["penalty_lp"]))
    write_csv(rows, output_dir / "pareto_summary.csv")
    write_expression_report(rows, output_dir)

    log_y = not args.linear_y
    for model in sorted({row["model"] for row in rows}):
        _plot_model(rows, model, output_dir / ("%s_pareto.png" % model), log_y)
    _plot_combined(rows, output_dir / "combined_pareto.png", log_y)

    print("Wrote Pareto CSV: %s" % (output_dir / "pareto_summary.csv"))
    print("Wrote expression report: %s" % (output_dir / "best_expressions.md"))
    print("Wrote Pareto plots under: %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
