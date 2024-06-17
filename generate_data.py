import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_vasicek_data(num_periods, r0, alpha, mu, sigma, dt=1/252):
    """
    Generates synthetic interest rate data using the Vasicek model.

    Parameters:
    - num_periods (int): Number of periods (or time steps) for the simulation.
    - r0 (float): Initial interest rate.
    - alpha (float): Mean-reversion parameter.
    - mu (float): Long-term mean of the interest rate.
    - sigma (float): Volatility parameter.
    - dt (float): Time step (default: 1/252, assuming daily data).

    Returns:
    - pd.Series: Simulated interest rate series.
    """
    rates = [r0]
    for _ in range(1, num_periods):
        dr = alpha * (mu - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates.append(rates[-1] + dr)
    return pd.Series(rates)

if __name__ == "__main__":
    num_periods = 252  # Number of trading days in a year
    r0 = 0.02  # Initial interest rate
    alpha = 0.1  # Mean-reversion parameter
    mu = 0.025  # Long-term mean
    sigma = 0.005  # Volatility parameter

    interest_rates = generate_vasicek_data(num_periods, r0, alpha, mu, sigma)
    plt.figure(figsize=(10, 6))
    plt.plot(interest_rates, label='Simulated Interest Rates')
    plt.title('Simulated Interest Rates using Vasicek Model')
    plt.xlabel('Time Steps')
    plt.ylabel('Interest Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
