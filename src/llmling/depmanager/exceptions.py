from __future__ import annotations


class DependencyError(Exception):
    """Error during dependency management."""


class ScriptError(DependencyError):
    """Error related to script loading/processing."""


class ImportPathError(DependencyError):
    """Error related to import path resolution."""
