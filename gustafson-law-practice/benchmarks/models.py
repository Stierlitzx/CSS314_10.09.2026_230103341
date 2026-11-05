"""Analytical scaling models; fractions have explicitly different reference frames."""
import math


def validate(processors, fraction):
    if isinstance(processors, bool) or not isinstance(processors, int) or processors < 1:
        raise ValueError("processors must be a positive integer")
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be finite and in [0, 1]")


def amdahl(processors, serial_fraction):
    """Fixed-work speedup, with fraction measured in the one-core reference."""
    validate(processors, serial_fraction)
    return 1 / (serial_fraction + (1 - serial_fraction) / processors)


def gustafson(processors, alpha):
    """Scaled-work speedup, with alpha measured on the parallel machine."""
    validate(processors, alpha)
    return processors - alpha * (processors - 1)


def equivalent_serial_fraction(processors, alpha):
    validate(processors, alpha)
    return alpha / gustafson(processors, alpha)