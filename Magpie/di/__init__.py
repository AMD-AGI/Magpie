# ruff: noqa: N999 -- retain the existing public Magpie package name.
"""Managed disaggregated inference on prepared Ray worker runtimes."""

from .native_ray import NativeRayDISession as DISession

__all__ = ["DISession"]
