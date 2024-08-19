import numpy as np


def rayleigh_pdf(x, offset, sigma):
    pdf = np.zeros_like(x)
    mask = x >= offset

    # Boolean mask for x values greater than or equal to loc
    pdf[mask] = ((x[mask] - offset) / sigma ** 2) * np.exp(-((x[mask] - offset) ** 2) / (2 * sigma ** 2))

    return pdf  # Return the entire array, with zeros for x < loc


# Define the Rayleigh PDF function
def rayleigh_heaviside_pdf(x, offset, sigma):
    scale = 1.0e+3
    base = 1.0e-2
    linear_part = base * np.where(x < offset, x / offset, 0.0)  # Linear growth below offset
    rayleigh_part = ((base + scale * ((x - offset) / sigma ** 2) * np.exp((-(x - offset) ** 2) / (2 * sigma ** 2))) *
                     np.heaviside(x - offset, 0))  # Rayleigh above offset
    return linear_part + rayleigh_part


def draw_rewards():
    import matplotlib.pyplot as plt
    # Parameters
    scale = 5e+3  # Adjust as needed
    min_ = 0e+3
    offset = 70e+3
    max_ = 100e+3
    # Calculate PDF
    x = np.linspace(min_, max_, 1000)
    # pdf = rayleigh_pdf(x, offset, scale)
    pdf = rayleigh_heaviside_pdf(x, offset, scale)
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, 'r-', lw=2, label='PDF')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Rayleigh Distribution')
    plt.show()
