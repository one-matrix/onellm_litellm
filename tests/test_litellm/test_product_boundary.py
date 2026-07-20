"""Architecture guards for keeping the LiteLLM runtime product-neutral."""

from __future__ import annotations

import ast
from pathlib import Path


LITELLM_ROOT = Path(__file__).parents[2] / "litellm"


def test_litellm_source_does_not_import_onellm() -> None:
    """The upstream runtime must never import the OneLLM control plane."""

    violations: list[str] = []
    for path in LITELLM_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        # Most of LiteLLM is unrelated. Avoid parsing thousands of files when
        # the forbidden module name is not even present in the source.
        if "onellm" not in source.lower():
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                imported_modules.append(node.args[0].value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                imported_modules.append(node.args[0].value)

            if any(
                module == "onellm" or module.startswith("onellm.")
                for module in imported_modules
            ):
                violations.append(
                    f"{path.relative_to(LITELLM_ROOT)}:{getattr(node, 'lineno', '?')}"
                )

    assert violations == [], "LiteLLM imports OneLLM at: " + ", ".join(violations)
