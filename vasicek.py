import numpy as np
from scipy.optimize import minimize

def vasicek_likelihood(params, data):
    r0, alpha, mu, sigma = params
    dt = 1/252
    N = len(data)
    
    sum_sq_diff = np.sum((data - mu * (1 - np.exp(-alpha * dt)) / alpha - np.exp(-alpha * dt) * r0)**2)
    sum_sq_diff += np.sum(((np.exp(-alpha * dt) * sigma**2) / (2 * alpha)) * (1 - np.exp(-2 * alpha * np.arange(1, N + 1) * dt)))
    
    return sum_sq_diff

def fit_vasicek(data):
    r0_guess = data[0]
    alpha_guess = 0.5
    mu_guess = np.mean(data)
    sigma_guess = np.std(data)
    
    params_guess = [r0_guess, alpha_guess, mu_guess, sigma_guess]
    bounds = [(0, None), (0, 1), (0, None), (0, None)]
    
    result = minimize(vasicek_likelihood, params_guess, args=(data,), bounds=bounds)
    return result.x

if __name__ == "__main__":
    # Example usage to fit Vasicek model parameters to generated data
    from generate_data import generate_vasicek_data

    num_periods = 252
    r0 = 0.02
    alpha = 0.1
    mu = 0.025
    sigma = 0.005

    interest_rates = generate_vasicek_data(num_periods, r0, alpha, mu, sigma)
    r0_fit, alpha_fit, mu_fit, sigma_fit = fit_vasicek(interest_rates)

    print(f"Fitted r0: {r0_fit}")
    print(f"Fitted alpha: {alpha_fit}")
    print(f"Fitted mu: {mu_fit}")
    print(f"Fitted sigma: {sigma_fit}")
