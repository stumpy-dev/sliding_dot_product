import ast
import importlib
import pkgutil
import warnings

import sdp


def func_exists(mod_path, func_name):
    try:
        with open(mod_path, "r") as file:
            module_content = file.read()
    except FileNotFoundError as e:
        warnings.warn(f"SKIPPED: {mod_path},{func_name}: \n{e}")
        return False  # Module file not found

    try:
        tree = ast.parse(module_content)
    except SyntaxError as e:
        warnings.warn(f"SKIPPED: {mod_path},{func_name}: \n{e}")
        return False  # Syntax error in module

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return True
    e = f"Function {func_name} not found in {mod_path}"
    warnings.warn(f"SKIPPED: {mod_path},{func_name}: \n{e}")
    return False


def import_sdp_mods(include=None, ignore=None):
    mods = []
    for m in sorted(list(pkgutil.iter_modules(sdp.__path__))):
        mod_path = f"sdp/{m[1]}.py"
        if (
            include is not None
            and len(include)
            and not any(mod in mod_path for mod in include)
        ):
            continue
        if (
            ignore is not None
            and len(ignore)
            and any(mod in mod_path for mod in ignore)
        ):
            continue

        if (
            "sdp" in m[1]
            and func_exists(mod_path, "sliding_dot_product")
            and func_exists(mod_path, "setup")
        ):
            mod_name = f"sdp.{m[1]}"
            mod = importlib.import_module(mod_name)
            mods.append(mod)

    return mods