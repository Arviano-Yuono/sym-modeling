from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import sympy as sp


DEFAULT_INPUT_ROOT = "output/sgeppy_results_jax"
DEFAULT_MODELS = ("nh2", "nh4", "ih", "hw", "gt", "ab")
DEFAULT_TITLE = "SGEPPY Best Expressions"

LATEX_NAMES = {
    "K1": "K_1",
    "K2": "K_2",
    "Jm1": "J - 1",
    "logI13": r"\log(I_1 / 3)",
    "logI23": r"\log(I_2 / 3)",
}
LATEX_SYMBOL_NAMES = {
    "K1": "K_1",
    "K2": "K_2",
    "I1": "I_1",
    "I2": "I_2",
    "I3": "I_3",
    "Jm1": r"\left(J - 1\right)",
    "logI13": r"\log(I_1 / 3)",
    "logI23": r"\log(I_2 / 3)",
}
TEXT_NAMES = {
    "K1": "K1",
    "K2": "K2",
    "Jm1": "(J - 1)",
    "logI13": "log(I1 / 3)",
    "logI23": "log(I2 / 3)",
}
DISPLAY_FUNCTIONS = {
    "protected_div": lambda a, b: a / b,
    "protected_sqrt": sp.sqrt,
    "protected_log": sp.log,
    "protected_exp": sp.exp,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "sin": sp.sin,
    "cos": sp.cos,
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
}


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _strip_outer_parens(text: str) -> str:
    if text.startswith(r"\left(") and text.endswith(r"\right)"):
        return text[6:-7]
    return text


def _format_number(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return "%.8g" % value
    return str(value)


def _expression_names(expression: str) -> tuple[set[str], set[str]]:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return set(), set()
    called_names = {
        node.func.id
        for node in ast.walk(parsed)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    names = {node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)}
    return names, called_names


def _round_sympy_numbers(expression, decimals: int | None):
    if decimals is None:
        return expression
    replacements = {}
    for number in expression.atoms(sp.Float):
        value = float(number)
        if not math.isfinite(value):
            continue
        if decimals == 0:
            replacements[number] = sp.Integer(round(value))
        else:
            replacements[number] = sp.Float(f"{value:.{decimals}f}")
    return expression.xreplace(replacements)


def _sympify_display_expression(expression: str):
    names, called_names = _expression_names(expression)
    parser_locals = {
        name: sp.Symbol(name)
        for name in names
        if name not in called_names and name not in DISPLAY_FUNCTIONS
    }
    parser_locals.update(
        {
            name: sp.Function(name)
            for name in called_names
            if name not in DISPLAY_FUNCTIONS
        }
    )
    parser_locals.update(DISPLAY_FUNCTIONS)
    return sp.sympify(expression, locals=parser_locals)


def simplify_expression_for_display(expression: str, round_decimals: int | None = None) -> str:
    if not expression or not expression.strip():
        return expression
    try:
        simplified = sp.simplify(_sympify_display_expression(expression))
    except Exception:
        return expression
    return str(_round_sympy_numbers(simplified, round_decimals))


def _latex_node(node: ast.AST) -> str:
    if isinstance(node, ast.Expression):
        return _latex_node(node.body)
    if isinstance(node, ast.Constant):
        return _format_number(node.value)
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
        if name == "protected_div" and len(args) == 2:
            return r"\frac{%s}{%s}" % (_strip_outer_parens(args[0]), _strip_outer_parens(args[1]))
        if name in {"square"} and len(args) == 1:
            return r"\left(%s\right)^2" % _strip_outer_parens(args[0])
        if name in {"cube"} and len(args) == 1:
            return r"\left(%s\right)^3" % _strip_outer_parens(args[0])
        if name in {"protected_sqrt", "sqrt"} and len(args) == 1:
            return r"\sqrt{%s}" % _strip_outer_parens(args[0])
        if name in {"protected_log", "log"} and len(args) == 1:
            return r"\log\left(%s\right)" % _strip_outer_parens(args[0])
        if name in {"protected_exp", "exp"} and len(args) == 1:
            return r"\exp\left(%s\right)" % _strip_outer_parens(args[0])
        if name == "sin" and len(args) == 1:
            return r"\sin\left(%s\right)" % _strip_outer_parens(args[0])
        if name == "cos" and len(args) == 1:
            return r"\cos\left(%s\right)" % _strip_outer_parens(args[0])
        return r"%s\left(%s\right)" % (name, ", ".join(_strip_outer_parens(arg) for arg in args))
    return ast.unparse(node)


def expression_to_latex(expression: str) -> str:
    if not expression:
        return ""
    try:
        parsed = _sympify_display_expression(expression)
    except Exception:
        parsed = None
    if parsed is not None:
        symbol_names = {
            symbol: LATEX_SYMBOL_NAMES.get(symbol.name, symbol.name)
            for symbol in parsed.free_symbols
        }
        return sp.latex(parsed, symbol_names=symbol_names)
    try:
        parsed_ast = ast.parse(expression, mode="eval")
    except SyntaxError:
        latex = expression
        for name, replacement in LATEX_NAMES.items():
            latex = re.sub(r"\b%s\b" % re.escape(name), replacement, latex)
        return latex.replace("*", r"\,")
    return _strip_outer_parens(_latex_node(parsed_ast))


def expression_to_text(expression: str) -> str:
    text = expression or ""
    for name, replacement in TEXT_NAMES.items():
        text = re.sub(r"\b%s\b" % re.escape(name), replacement, text)
    text = text.replace("**", "^")
    text = text.replace("protected_div", "div")
    text = text.replace("protected_sqrt", "sqrt")
    text = text.replace("protected_log", "log")
    text = text.replace("protected_exp", "exp")
    return text


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_optional_scientific(value: float | None) -> str:
    return "" if value is None else "%.6e" % value


def _load_payload(model_dir: Path) -> tuple[dict, Path] | None:
    for filename in ("summary.json", "best_so_far.json"):
        path = model_dir / filename
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8")), path
    return None


def _row_from_payload(model: str, payload: dict, source_path: Path, round_decimals: int | None = None) -> dict:
    metrics = payload.get("metrics", {})
    best_fitness = payload.get("best_fitness") or []
    rmse = _safe_float(metrics.get("rmse"))
    if rmse is None and best_fitness:
        rmse = _safe_float(best_fitness[0])
    expression = simplify_expression_for_display(payload.get("best_expression", ""), round_decimals=round_decimals)
    active_terms = metrics.get("num_parameters")
    if active_terms is None:
        active_terms = metrics.get("active_terms")
    return {
        "model": model,
        "rmse": rmse,
        "rss": _safe_float(metrics.get("rss")),
        "aicc": _safe_float(metrics.get("aicc")),
        "active_terms": active_terms,
        "source_path": str(source_path),
        "best_expression": expression,
        "text_expression": expression_to_text(expression),
        "latex_expression": expression_to_latex(expression),
    }


def load_expression_rows(input_root: Path, models: Iterable[str], round_decimals: int | None = None) -> list[dict]:
    rows = []
    for model in models:
        model_dir = input_root / model
        if not model_dir.is_dir():
            continue
        loaded = _load_payload(model_dir)
        if loaded is None:
            continue
        payload, source_path = loaded
        rows.append(_row_from_payload(model, payload, source_path, round_decimals=round_decimals))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "rmse",
        "rss",
        "aicc",
        "active_terms",
        "best_expression",
        "text_expression",
        "latex_expression",
        "source_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path, title: str = DEFAULT_TITLE) -> None:
    lines = [
        "# %s" % title,
        "",
        "| Model | RMSE | Active terms | Expression |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        rmse = "" if row["rmse"] is None else "%.6e" % row["rmse"]
        active_terms = "" if row["active_terms"] is None else str(row["active_terms"])
        lines.append(
            "| {model} | {rmse} | {active_terms} | ${latex_expression}$ |".format(
                model=row["model"].upper(),
                rmse=rmse,
                active_terms=active_terms,
                latex_expression=row["latex_expression"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_text(rows: list[dict], path: Path) -> None:
    lines = []
    for row in rows:
        rmse = "" if row["rmse"] is None else "  RMSE: %.6e" % row["rmse"]
        lines.append("%s%s" % (row["model"].upper(), rmse))
        lines.append("  W = %s" % row["text_expression"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(rows: list[dict], path: Path, title: str = DEFAULT_TITLE) -> None:
    lines = [
        "%% %s." % title,
        r"% Generated by sym-util-print.",
        "",
        r"\begin{tabular}{lllp{0.68\linewidth}}",
        r"\hline",
        r"Model & RMSE & Active terms & Expression \\",
        r"\hline",
    ]
    for row in rows:
        active_terms = "" if row["active_terms"] is None else str(row["active_terms"])
        lines.append(
            r"%s & %s & %s & $W = %s$ \\"
            % (
                row["model"].upper(),
                _format_optional_scientific(row["rmse"]),
                active_terms,
                row["latex_expression"],
            )
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export best symbolic expressions as readable text and LaTeX."
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model folders to include.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Report directory. Defaults to <input-root>/expression_report.",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Title used in Markdown and LaTeX reports.",
    )
    parser.add_argument(
        "--round-decimals",
        type=_nonnegative_int,
        default=4,
        help="Round numeric coefficients in exported expressions to this many decimal places.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else input_root / "expression_report"
    rows = load_expression_rows(input_root, _parse_csv_strings(args.models), round_decimals=args.round_decimals)
    if not rows:
        raise FileNotFoundError("No expression summaries found under %s" % input_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "best_expressions.csv")
    write_markdown(rows, output_dir / "best_expressions.md", title=args.title)
    write_text(rows, output_dir / "best_expressions.txt")
    write_latex(rows, output_dir / "best_expressions.tex", title=args.title)

    print("Wrote expression report: %s" % (output_dir / "best_expressions.md"))
    print("Wrote LaTeX report: %s" % (output_dir / "best_expressions.tex"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
