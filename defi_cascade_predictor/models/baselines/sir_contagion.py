"""
SIR Contagion Model baseline.
Adapts the epidemiological Susceptible-Infected-Recovered model
to DeFi protocol contagion dynamics.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from loguru import logger


class SIRContagionModel:
    """SIR epidemic model adapted for DeFi liquidation cascades.

    Models each protocol as either:
      - S (Susceptible): operating normally but exposed to cascade risk
      - I (Infected): experiencing liquidation cascade / TVL crisis
      - R (Recovered): stabilized after cascade

    The infection rate beta depends on the network adjacency (shared
    collateral, oracle dependencies, etc.), and the recovery rate gamma
    depends on protocol resilience factors.
    """

    def __init__(
        self,
        num_protocols: int,
        adjacency_matrix: np.ndarray,
        beta_range: tuple = (0.01, 0.5),
        gamma_range: tuple = (0.01, 0.3),
        n_simulations: int = 1000,
        prediction_horizons: list[int] = [24, 72, 168, 720],
    ):
        self.num_protocols = num_protocols
        self.adj = adjacency_matrix
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        self.n_simulations = n_simulations
        self.prediction_horizons = prediction_horizons

        # Fitted parameters
        self.beta = None
        self.gamma = None
        self.protocol_vulnerability = np.ones(num_protocols)

    def _sir_ode(self, y, t, beta, gamma, adj):
        """SIR ODE system adapted for network contagion."""
        n = self.num_protocols
        S = y[:n]
        I = y[n:2*n]
        R = y[2*n:]

        # Network-mediated infection: each node's infection rate depends
        # on the infection level of its neighbors weighted by adjacency
        neighbor_infection = adj @ I  # weighted sum of infected neighbors
        infection_rate = beta * S * neighbor_infection * self.protocol_vulnerability

        recovery_rate = gamma * I

        dSdt = -infection_rate
        dIdt = infection_rate - recovery_rate
        dRdt = recovery_rate

        return np.concatenate([dSdt, dIdt, dRdt])

    def fit(
        self,
        cascade_events: list[dict],
        protocol_tvl_changes: np.ndarray,
    ):
        """Fit SIR parameters to historical cascade events.

        Args:
            cascade_events: List of cascade event dicts with severity info.
            protocol_tvl_changes: [num_events, num_protocols] TVL changes
                during each event (negative = affected).
        """
        logger.info("Fitting SIR contagion model parameters")

        def objective(params):
            beta, gamma = params
            total_error = 0.0

            for i, event in enumerate(cascade_events):
                if i >= len(protocol_tvl_changes):
                    break

                observed = protocol_tvl_changes[i]
                # Determine initially infected protocols
                initial_infected = (observed < -0.05).astype(float)
                if initial_infected.sum() == 0:
                    continue

                # Run SIR simulation
                predicted = self._simulate_single(
                    beta, gamma, initial_infected, duration=7
                )
                # Compare predicted infection spread with observed
                total_error += np.mean(
                    (predicted - (observed < -0.05).astype(float)) ** 2
                )

            return total_error

        # Optimize
        result = minimize(
            objective,
            x0=[0.1, 0.1],
            bounds=[self.beta_range, self.gamma_range],
            method="L-BFGS-B",
        )

        self.beta, self.gamma = result.x
        logger.info(
            f"Fitted SIR params: beta={self.beta:.4f}, gamma={self.gamma:.4f}"
        )

        # Estimate per-protocol vulnerability from historical data
        if len(protocol_tvl_changes) > 0:
            avg_impact = np.mean(
                np.abs(protocol_tvl_changes), axis=0
            )
            self.protocol_vulnerability = avg_impact / (avg_impact.mean() + 1e-8)

    def _simulate_single(
        self,
        beta: float,
        gamma: float,
        initial_infected: np.ndarray,
        duration: int = 7,
    ) -> np.ndarray:
        """Run a single SIR simulation.

        Returns:
            Peak infection level per protocol.
        """
        n = self.num_protocols
        S0 = 1.0 - initial_infected
        I0 = initial_infected.copy()
        R0 = np.zeros(n)

        y0 = np.concatenate([S0, I0, R0])
        t = np.linspace(0, duration, duration * 24)  # hourly

        solution = odeint(
            self._sir_ode, y0, t, args=(beta, gamma, self.adj)
        )

        # Extract peak infection level per protocol
        I_timeseries = solution[:, n:2*n]
        peak_infection = I_timeseries.max(axis=0)

        return peak_infection

    def predict(
        self,
        current_state: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Predict cascade probability using Monte Carlo SIR simulations.

        Args:
            current_state: [num_protocols] current risk state (0-1).

        Returns:
            Dict with cascade probability per horizon.
        """
        if self.beta is None:
            # Use default parameters
            self.beta = 0.1
            self.gamma = 0.1

        predictions = {f"cascade_{h}h": [] for h in self.prediction_horizons}

        for _ in range(self.n_simulations):
            # Random perturbation to initial state
            rng = np.random.default_rng()
            noise = rng.normal(0, 0.05, self.num_protocols)
            perturbed = np.clip(current_state + noise, 0, 1)

            # Initial infected = protocols with high risk
            initial_infected = (perturbed > 0.5).astype(float)

            # Simulate
            peak = self._simulate_single(
                self.beta, self.gamma, initial_infected, duration=7
            )

            # Check if cascade occurs at each horizon
            for h in self.prediction_horizons:
                # Cascade = >30% of protocols infected above threshold
                cascade = (peak > 0.3).sum() >= self.num_protocols * 0.3
                predictions[f"cascade_{h}h"].append(float(cascade))

        # Aggregate across simulations
        result = {}
        for key, vals in predictions.items():
            result[key] = np.array(vals).mean()

        return result

    def predict_batch(
        self, states: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Predict for a batch of states.

        Args:
            states: [batch_size, num_protocols] risk states.

        Returns:
            Dict with cascade probability arrays per horizon.
        """
        batch_results = {
            f"cascade_{h}h": [] for h in self.prediction_horizons
        }

        for state in states:
            pred = self.predict(state)
            for key in batch_results:
                batch_results[key].append(pred[key])

        return {k: np.array(v) for k, v in batch_results.items()}
