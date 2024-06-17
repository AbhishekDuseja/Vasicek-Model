import matplotlib.pyplot as plt
import numpy as np
from generate_data import generate_vasicek_data
from vasicek import fit_vasicek

def plot_vasicek_variations(num_periods, r0, alpha, mu, sigma, num_samples=5):
    plt.figure(figsize=(12, 8))
    for _ in range(num_samples):
        interest_rates = generate_vasicek_data(num_periods, r0, alpha, mu, sigma)
        r0_fit, alpha_fit, mu_fit, sigma_fit = fit_vasicek(interest_rates)
        
        plt.plot(interest_rates, label=f'Simulated Data (r0={r0}, alpha={alpha}, mu={mu}, sigma={sigma})')
        plt.plot(np.arange(num_periods), [mu_fit] * num_periods, linestyle='--', label='Fitted mu', color='black')
        
        plt.title('Vasicek Model Variations')
        plt.xlabel('Time Steps')
        plt.ylabel('Interest Rate')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_periods = 252
    r0 = 0.02
    alpha = 0.1
    mu = 0.025
    sigma = 0.005
    num_samples = 5

    plot_vasicek_variations(num_periods, r0, alpha, mu, sigma, num_samples)
