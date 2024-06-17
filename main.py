from generate_data import generate_vasicek_data
from vasicek import fit_vasicek
from plot_vasicek import plot_vasicek_variations

def main():
    num_periods = 252
    r0 = 0.02
    alpha = 0.1
    mu = 0.025
    sigma = 0.005
    num_samples = 5

    # Generate synthetic data and fit Vasicek model
    interest_rates = generate_vasicek_data(num_periods, r0, alpha, mu, sigma)
    r0_fit, alpha_fit, mu_fit, sigma_fit = fit_vasicek(interest_rates)

    print(f"Fitted r0: {r0_fit}")
    print(f"Fitted alpha: {alpha_fit}")
    print(f"Fitted mu: {mu_fit}")
    print(f"Fitted sigma: {sigma_fit}")

    # Plot variations of the Vasicek model
    plot_vasicek_variations(num_periods, r0, alpha, mu, sigma, num_samples)

if __name__ == "__main__":
    main()
