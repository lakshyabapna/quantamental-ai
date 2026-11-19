"""Quantamental AI - Capital Allocation Line (compatibility re-export shim).

Re-exports from app.portfolio_engine.cal for callers using the financial_engine namespace.
"""

from app.portfolio_engine.cal import (
    CALCoordinate,
    CALPoint,
    CALResult,
    CapitalAllocationLineResult,
    calculate_cal_return,
    calculate_cal_slope,
    generate_cal_coordinates,
    generate_capital_allocation_line,
)

__all__ = [
    "CALCoordinate",
    "CALPoint",
    "CALResult",
    "CapitalAllocationLineResult",
    "calculate_cal_return",
    "calculate_cal_slope",
    "generate_cal_coordinates",
    "generate_capital_allocation_line",
]
