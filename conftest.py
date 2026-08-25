"""
Root conftest.py.

Its mere presence at the project root (which has no `__init__.py`, i.e. is
not itself a package) makes pytest insert the project root onto `sys.path`
in its default "prepend" import mode, so test modules can `import src...`
without needing any pyproject.toml pytest configuration.
"""
