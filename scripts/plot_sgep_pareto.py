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


DEFAULT_MODELS = ("nh2", "nh4", "ih", "hw", "gt")
EPSILON_PATTERN = re.compile(r"eps(\d+)$")
LATEX_NAMES = {
    "K1": "K_1",
    "K2": "K_2",
    "Jm1": "J - 1",
}
LATEX_FUNCTIONS = {
    "sin": r"\sin",
    "cos": r"\cos",
    "log": r"\log",
}


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _epsilon_from_path(path: Path) -> int:
    match = EPSILON_PATTERN.match(path.parent.name)
    if not match:
        raise ValueError("Could not infer epsilon from path: %s" % path)
    return int(match.group(1))


def _active_complexity(payload: dict) -> int:
    return int(sum(int(gene.get("complexity", 0)) for gene in payload.get("active_genes", [])))


def _read_summary(model: str, summary_path: Path) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    config = payload.get("config", {})
    active_terms = int(metrics.get("num_parameters", len(payload.get("active_genes", []))))
    epsilon = int(config.get("active_terms_epsilon", _epsilon_from_path(summary_path)))
    return {
        "model": model,
        "epsilon": epsilon,
        "active_terms": active_terms,
        "active_complexity": _active_complexity(payload),
        "rmse": float(metrics.get("rmse", float("inf"))),
        "rss": float(metrics.get("rss", float("inf"))),
        "aicc": float(metrics.get("aicc", float("inf"))),
        "feasible": active_terms <= epsilon,
        "best_expression": payload.get("best_expression", ""),
        "summary_path": str(summary_path),
    }


def load_rows(input_root: Path, models: Iterable[str]) -> list[dict]:
    rows = []
    for model in models:
        model_dir = input_root / model
        for summary_path in sorted(model_dir.glob("eps*/summary.json"), key=_epsilon_from_path):
            rows.append(_read_summary(model, summary_path))
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
        "epsilon",
        "active_terms",
        "active_complexity",
        "rmse",
        "rss",
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


def write_expression_reports(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expression_rows = []
    for row in rows:
        expression_rows.append(
            {
                **row,
                "latex_expression": expression_to_latex(row["best_expression"]),
            }
        )

    md_lines = [
        "# SGEP Best Expressions by Epsilon",
        "",
        "| Model | Epsilon | Active terms | RMSE | Pareto | Expression |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in expression_rows:
        md_lines.append(
            "| {model} | {epsilon} | {active_terms} | {rmse:.6e} | {pareto} | ${latex_expression}$ |".format(
                **row
            )
        )
    (output_dir / "best_expressions.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"% SGEP best expressions by epsilon.",
        r"% Generated by scripts/plot_sgep_pareto.py",
        "",
    ]
    for row in expression_rows:
        tex_lines.extend(
            [
                r"\paragraph{%s, $\epsilon=%s$}" % (row["model"].upper(), row["epsilon"]),
                r"\[",
                r"W = %s" % row["latex_expression"],
                r"\]",
                "",
            ]
        )
    (output_dir / "best_expressions.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def _plot_model(rows: list[dict], model: str, output_path: Path, log_y: bool) -> None:
    model_rows = sorted([row for row in rows if row["model"] == model], key=lambda row: row["epsilon"])
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
        label="epsilon run",
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
            "eps%d" % row["epsilon"],
            (row["active_terms"], row["rmse"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_title("%s SGEP Pareto Front" % model.upper())
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
        model_rows = sorted([row for row in rows if row["model"] == model], key=lambda row: row["epsilon"])
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
    ax.set_title("SGEP Pareto Fronts")
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
    parser = argparse.ArgumentParser(description="Evaluate and plot SGEP epsilon-sweep Pareto fronts.")
    parser.add_argument("--input-root", default="output/sgep_direct_piola")
    parser.add_argument("--output-dir", default="output/sgep_direct_piola/pareto")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--linear-y", action="store_true", help="Use a linear RMSE axis instead of log scale.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    rows = mark_pareto(load_rows(input_root, _parse_csv_strings(args.models)))
    if not rows:
        raise FileNotFoundError("No SGEP summary files found under %s." % input_root)

    rows.sort(key=lambda row: (row["model"], row["epsilon"]))
    write_csv(rows, output_dir / "pareto_summary.csv")
    write_expression_reports(rows, output_dir)

    log_y = not args.linear_y
    for model in sorted({row["model"] for row in rows}):
        _plot_model(rows, model, output_dir / ("%s_pareto.png" % model), log_y)
    _plot_combined(rows, output_dir / "combined_pareto.png", log_y)

    print("Wrote Pareto CSV: %s" % (output_dir / "pareto_summary.csv"))
    print("Wrote expression reports: %s" % (output_dir / "best_expressions.md"))
    print("Wrote Pareto plots under: %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
