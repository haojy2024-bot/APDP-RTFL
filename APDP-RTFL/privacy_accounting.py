"""Dependency-free RDP accounting for client-side sample-level DP-SGD.

The accountant deliberately models Poisson-subsampled Gaussian DP-SGD.  It is
used as a conservative execution ledger: each client owns an independent
privacy budget and a client is never allowed to run a step that would exceed
that budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


DEFAULT_ORDERS = tuple(list(range(2, 33)) + [40, 48, 56, 64])


def _log_add(log_x: float, log_y: float) -> float:
    if log_x == -math.inf:
        return log_y
    if log_y == -math.inf:
        return log_x
    upper, lower = (log_x, log_y) if log_x >= log_y else (log_y, log_x)
    return upper + math.log1p(math.exp(lower - upper))


def sampled_gaussian_rdp(sample_rate: float, noise_multiplier: float, order: int) -> float:
    """RDP of one Poisson-subsampled Gaussian step for an integer order."""
    if not 0.0 < sample_rate <= 1.0:
        raise ValueError("sample_rate must lie in (0, 1]")
    if noise_multiplier <= 0:
        return math.inf
    if order < 2 or int(order) != order:
        raise ValueError("RDP orders must be integers >= 2")
    q = float(sample_rate)
    if q == 1.0:
        return order / (2.0 * noise_multiplier**2)
    log_total = -math.inf
    for i in range(order + 1):
        log_term = (
            math.lgamma(order + 1)
            - math.lgamma(i + 1)
            - math.lgamma(order - i + 1)
            + i * math.log(q)
            + (order - i) * math.log1p(-q)
            + (i * i - i) / (2.0 * noise_multiplier**2)
        )
        log_total = _log_add(log_total, log_term)
    return log_total / (order - 1)


def epsilon_from_rdp(rdp: np.ndarray, orders: tuple[int, ...], delta: float) -> float:
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    candidates = [float(value) + math.log(1.0 / delta) / (order - 1) for value, order in zip(rdp, orders)]
    return float(min(candidates))


def calibrate_noise_multiplier(sample_rate: float, steps: int, target_epsilon: float, delta: float, orders=DEFAULT_ORDERS) -> float:
    """Find the smallest Gaussian noise multiplier meeting a planned budget."""
    if steps <= 0:
        return 0.0
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    orders = tuple(int(order) for order in orders)

    def epsilon_for(sigma: float) -> float:
        per_step = np.asarray([sampled_gaussian_rdp(sample_rate, sigma, order) for order in orders])
        return epsilon_from_rdp(per_step * steps, orders, delta)

    lower, upper = 1e-3, 1.0
    while epsilon_for(upper) > target_epsilon:
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("unable to calibrate a finite DP-SGD noise multiplier")
    for _ in range(60):
        middle = (lower + upper) / 2.0
        if epsilon_for(middle) <= target_epsilon:
            upper = middle
        else:
            lower = middle
    return float(upper)


@dataclass
class PrivacyEvent:
    round_num: int
    steps: int
    sample_rate: float
    noise_multiplier: float
    epsilon: float
    incremental_epsilon: float
    status: str


class RDPAccountant:
    """Per-client RDP ledger for DP-SGD execution."""

    def __init__(self, target_epsilon: float, delta: float, orders=DEFAULT_ORDERS):
        self.target_epsilon = float(target_epsilon)
        self.delta = float(delta)
        self.orders = tuple(int(order) for order in orders)
        self.rdp = np.zeros(len(self.orders), dtype=float)
        self.events: list[PrivacyEvent] = []

    @property
    def epsilon(self) -> float:
        if not np.any(self.rdp):
            return 0.0
        return epsilon_from_rdp(self.rdp, self.orders, self.delta)

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.target_epsilon - self.epsilon)

    def projected_epsilon(self, steps: int, sample_rate: float, noise_multiplier: float) -> float:
        if steps <= 0:
            return self.epsilon
        increment = np.asarray([
            sampled_gaussian_rdp(sample_rate, noise_multiplier, order) for order in self.orders
        ]) * steps
        return epsilon_from_rdp(self.rdp + increment, self.orders, self.delta)

    def can_spend(self, steps: int, sample_rate: float, noise_multiplier: float) -> bool:
        return self.projected_epsilon(steps, sample_rate, noise_multiplier) <= self.target_epsilon + 1e-10

    def spend(self, round_num: int, steps: int, sample_rate: float, noise_multiplier: float) -> PrivacyEvent:
        before = self.epsilon
        projected = self.projected_epsilon(steps, sample_rate, noise_multiplier)
        if not self.can_spend(steps, sample_rate, noise_multiplier):
            event = PrivacyEvent(round_num, steps, sample_rate, noise_multiplier, before, 0.0, "budget_exhausted")
            self.events.append(event)
            return event
        if steps:
            self.rdp += np.asarray([
                sampled_gaussian_rdp(sample_rate, noise_multiplier, order) for order in self.orders
            ]) * steps
        after = self.epsilon
        event = PrivacyEvent(round_num, steps, sample_rate, noise_multiplier, after, after - before, "spent")
        self.events.append(event)
        return event
