"""Compatibility shims for supported NVIDIA CuTe DSL releases."""

import cutlass
import cutlass.cute as cute


def ensure_quack_compat() -> None:
    """Restore CuTe type aliases required when importing Quack 0.5.0."""

    core = getattr(cute, "core", None)
    missing_types = []
    for type_name in ("ThrMma", "ThrCopy"):
        if core is not None and hasattr(core, type_name):
            continue
        public_type = getattr(cute, type_name, None)
        if core is None or public_type is None:
            missing_types.append(type_name)
            continue
        setattr(core, type_name, public_type)
    if missing_types:
        version = getattr(cutlass, "__version__", "unknown")
        missing = ", ".join(missing_types)
        raise RuntimeError(
            "quack-kernels==0.5.0 is incompatible with "
            f"NVIDIA CuTe DSL {version}: missing CuTe types {missing}"
        )


__all__ = ["ensure_quack_compat"]
