#!/usr/bin/env python3
"""Generate API.md from shared module introspection.

Run after any Step 8 export:
    python3 course/shared/generate_api.py

Produces course/shared/API.md with every public function/class,
its signature, and its first docstring line.
"""

import ast
import textwrap
from pathlib import Path

SHARED_DIR = Path(__file__).parent

# Module ordering: data first, then core, domain, stubs last
MODULE_ORDER = [
    # Data layer
    ("data.py", "Data", "Cross-week download cache and universe constants"),
    # Core
    ("metrics.py", "Core", "Signal quality, statistical tests, deflated Sharpe"),
    ("temporal.py", "Core", "Temporal CV splitters (walk-forward, purged, CPCV)"),
    ("evaluation.py", "Core", "Rolling cross-sectional prediction harness"),
    ("backtesting.py", "Core", "Portfolio construction, performance, transaction costs"),
    ("dl_training.py", "Core", "Neural network fit/predict utilities"),
    # Domain
    ("portfolio.py", "Domain", "Portfolio optimization (Markowitz, HRP, Black-Litterman)"),
    ("derivatives.py", "Domain", "Options pricing, Greeks, implied volatility"),
    ("microstructure.py", "Domain", "Order flow, spreads, execution models"),
    ("regime.py", "Domain", "Regime detection, cointegration, mean reversion"),
    # Stubs
    ("nlp.py", "Stub", "NLP text processing (partial impl + stubs)"),
    ("causal.py", "Stub", "Causal inference (stubs)"),
    ("rl_env.py", "Stub", "RL trading environments (stubs)"),
]


def extract_public_api(filepath: Path) -> list[dict]:
    """Extract public functions, classes, and constants from a module."""
    source = filepath.read_text()
    tree = ast.parse(source)
    items = []

    for node in ast.iter_child_nodes(tree):
        # Top-level functions
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            sig = _extract_signature(node)
            doc = _first_docstring_line(node)
            dict_keys = _extract_dict_keys(node)
            items.append({
                "kind": "function",
                "name": node.name,
                "signature": sig,
                "doc": doc,
                "dict_keys": dict_keys,
                "line": node.lineno,
            })

        # Top-level classes
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            doc = _first_docstring_line(node)
            methods = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef) and not child.name.startswith("_"):
                    msig = _extract_signature(child)
                    mdoc = _first_docstring_line(child)
                    methods.append({
                        "name": child.name,
                        "signature": msig,
                        "doc": mdoc,
                    })
            # Also extract __init__ signature
            init_sig = None
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                    init_sig = _extract_signature(child, skip_self=True)
                    break
            items.append({
                "kind": "class",
                "name": node.name,
                "init_signature": init_sig,
                "doc": doc,
                "methods": methods,
                "line": node.lineno,
            })

        # Top-level constants (UPPER_CASE assignments)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = _extract_constant_names(node)
            for name in names:
                if name.isupper() and not name.startswith("_"):
                    items.append({
                        "kind": "constant",
                        "name": name,
                        "line": node.lineno,
                    })

    return items


def _extract_signature(node: ast.FunctionDef, skip_self: bool = False) -> str:
    """Build a signature string from a FunctionDef AST node."""
    args = node.args
    parts = []

    # Positional args
    all_args = args.args[:]
    if skip_self and all_args and all_args[0].arg == "self":
        all_args = all_args[1:]
    elif all_args and all_args[0].arg == "self":
        all_args = all_args[1:]

    # Defaults align to the END of the args list
    n_defaults = len(args.defaults)
    n_args = len(all_args)

    for i, arg in enumerate(all_args):
        name = arg.arg
        ann = _unparse_annotation(arg.annotation) if arg.annotation else None
        default_idx = i - (n_args - n_defaults)
        if default_idx >= 0:
            default = _unparse_node(args.defaults[default_idx])
            if ann:
                parts.append(f"{name}: {ann} = {default}")
            else:
                parts.append(f"{name}={default}")
        else:
            if ann:
                parts.append(f"{name}: {ann}")
            else:
                parts.append(name)

    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")

    # keyword-only
    for i, kwarg in enumerate(args.kwonlyargs):
        name = kwarg.arg
        ann = _unparse_annotation(kwarg.annotation) if kwarg.annotation else None
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            default = _unparse_node(args.kw_defaults[i])
            if ann:
                parts.append(f"{name}: {ann} = {default}")
            else:
                parts.append(f"{name}={default}")
        else:
            if ann:
                parts.append(f"{name}: {ann}")
            else:
                parts.append(name)

    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    # Return annotation
    ret = ""
    if node.returns:
        ret = f" -> {_unparse_annotation(node.returns)}"

    return f"({', '.join(parts)}){ret}"


def _unparse_annotation(node) -> str:
    """Best-effort unparse of a type annotation node."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def _unparse_node(node) -> str:
    """Best-effort unparse of a default value node."""
    try:
        result = ast.unparse(node)
        # Truncate long defaults
        if len(result) > 40:
            return result[:37] + "..."
        return result
    except Exception:
        return "..."


def _first_docstring_line(node) -> str:
    """Extract the first non-empty line of a docstring."""
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)):
        val = node.body[0].value
        raw = val.value if isinstance(val, ast.Constant) and isinstance(val.value, str) else ""
        for line in raw.strip().splitlines():
            line = line.strip()
            if line:
                return line
    return ""


def _extract_dict_keys(node) -> list[str]:
    """Extract dict return keys from a docstring's Returns block.

    Looks for patterns like:
        Returns:
            dict with keys:
                key_a: type — description.
                key_b: type — description.

    Returns a list of "key: description" strings, or empty list.
    """
    raw = _full_docstring(node)
    if not raw:
        return []

    lines = raw.splitlines()
    in_returns = False
    in_dict_keys = False
    keys = []
    dict_indent = 0

    for line in lines:
        stripped = line.strip()

        # Detect Returns: section
        if stripped.startswith("Returns:"):
            in_returns = True
            continue

        # Stop at next section header (Args:, Raises:, Note:, etc.)
        if in_returns and stripped and stripped[0].isupper() and stripped.endswith(":"):
            if stripped not in ("Returns:",):
                break

        if not in_returns:
            continue

        # Detect "dict with keys:" or "dict with:" pattern
        if not in_dict_keys and ("dict with keys:" in stripped.lower()
                                  or "dict with:" in stripped.lower()):
            in_dict_keys = True
            # Determine indent of next level
            dict_indent = len(line) - len(line.lstrip())
            continue

        if in_dict_keys:
            if not stripped:
                continue
            indent = len(line) - len(line.lstrip())
            # Key lines are indented deeper than the "dict with keys:" line
            if indent > dict_indent and ":" in stripped:
                # Extract "key_name: description"
                keys.append(stripped)
            elif indent <= dict_indent and stripped:
                # Back to same or lesser indent = end of dict keys block
                break

    return keys


def _full_docstring(node) -> str:
    """Extract the full docstring from an AST node."""
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)):
        val = node.body[0].value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return val.value
    return ""


def _extract_constant_names(node) -> list[str]:
    """Extract names from an assignment node."""
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, ast.Assign):
        names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        names.append(elt.id)
        return names
    return []


def _group_constants(constants: list[dict]) -> dict[str, list[str]]:
    """Group constants by shared prefix for readable rendering.

    Constants like FRED_TREASURY_YIELDS, FRED_MACRO, FRED_CREDIT get
    grouped under "FRED". Single constants without a shared prefix go
    into the "" group.
    """
    from collections import defaultdict

    names = [c["name"] for c in constants]
    # Count occurrences of each prefix (first word before _)
    prefix_counts = defaultdict(list)
    for name in names:
        parts = name.split("_", 1)
        prefix = parts[0] if len(parts) > 1 else ""
        prefix_counts[prefix].append(name)

    # Only group if prefix has 2+ members
    groups = {}
    ungrouped = []
    for prefix, members in prefix_counts.items():
        if len(members) >= 2 and prefix:
            groups[prefix] = members
        else:
            ungrouped.extend(members)
    if ungrouped:
        groups["Other"] = ungrouped

    # Sort groups: "Other" last
    sorted_groups = {}
    for k in sorted(groups.keys(), key=lambda x: (x == "Other", x)):
        sorted_groups[k] = groups[k]
    return sorted_groups


def _key_name(key_line: str) -> str:
    """Extract just the key name from a 'key: type — description' line."""
    # "mean_ic: float — average IC across periods."  ->  "`mean_ic`"
    name = key_line.split(":")[0].strip()
    return f"`{name}`"


def render_markdown(module_apis: list[tuple[str, str, str, list[dict]]]) -> str:
    """Render all modules into a single markdown doc."""
    lines = [
        "# Shared Library API Reference",
        "",
        "> **Auto-generated** by `generate_api.py`. Do not edit manually.",
        "> Regenerate: `python3 course/shared/generate_api.py`",
        "",
        "This file is consumed by Step 4A agents to discover available shared",
        "infrastructure. Import from specific modules:",
        "```python",
        "from shared.metrics import pearson_ic, rank_ic, deflated_sharpe_ratio",
        "from shared.temporal import PurgedWalkForwardCV, PurgedKFold",
        "from shared.backtesting import long_short_portfolio, sharpe_ratio",
        "```",
        "",
        "---",
        "",
    ]

    current_band = None
    for filename, band, description, items in module_apis:
        if band != current_band:
            current_band = band
            lines.append(f"## {band} Modules")
            lines.append("")

        module_name = filename.replace(".py", "")
        lines.append(f"### `shared/{module_name}` — {description}")
        lines.append("")

        # Constants first — grouped by prefix
        constants = [i for i in items if i["kind"] == "constant"]
        if constants:
            groups = _group_constants(constants)
            if len(groups) == 1 and "" in groups:
                # No meaningful grouping — render flat
                const_names = ", ".join(f"`{c}`" for c in groups[""])
                lines.append(f"**Constants:** {const_names}")
            else:
                lines.append("**Constants:**")
                for prefix, names in groups.items():
                    label = prefix if prefix else "Other"
                    const_names = ", ".join(f"`{n}`" for n in names)
                    lines.append(f"- *{label}*: {const_names}")
            lines.append("")

        # Functions
        functions = [i for i in items if i["kind"] == "function"]
        if functions:
            for f in functions:
                doc_part = f" — {f['doc']}" if f['doc'] else ""
                lines.append(f"- **`{f['name']}`**`{f['signature']}`{doc_part}")
                # Show dict return keys if present
                dict_keys = f.get("dict_keys", [])
                if dict_keys:
                    lines.append(f"  - Returns dict: {', '.join(_key_name(k) for k in dict_keys)}")
            lines.append("")

        # Classes
        classes = [i for i in items if i["kind"] == "class"]
        if classes:
            for c in classes:
                init_part = f"`{c['init_signature']}`" if c['init_signature'] else ""
                doc_part = f" — {c['doc']}" if c['doc'] else ""
                lines.append(f"- **`{c['name']}`**{init_part}{doc_part}")
                if c["methods"]:
                    for m in c["methods"]:
                        mdoc = f" — {m['doc']}" if m['doc'] else ""
                        lines.append(f"  - `.{m['name']}``{m['signature']}`{mdoc}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    module_apis = []
    for filename, band, description in MODULE_ORDER:
        filepath = SHARED_DIR / filename
        if not filepath.exists():
            continue
        items = extract_public_api(filepath)
        if items:
            module_apis.append((filename, band, description, items))

    md = render_markdown(module_apis)
    out_path = SHARED_DIR / "API.md"
    out_path.write_text(md)
    print(f"Generated {out_path} ({len(md)} chars, {md.count(chr(10))} lines)")

    # Summary
    total_funcs = sum(
        1 for _, _, _, items in module_apis
        for i in items if i["kind"] == "function"
    )
    total_classes = sum(
        1 for _, _, _, items in module_apis
        for i in items if i["kind"] == "class"
    )
    total_consts = sum(
        1 for _, _, _, items in module_apis
        for i in items if i["kind"] == "constant"
    )
    print(f"  {len(module_apis)} modules, {total_funcs} functions, "
          f"{total_classes} classes, {total_consts} constants")


if __name__ == "__main__":
    main()
