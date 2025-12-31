
"""
--------------------TITLE--------------------
Measuring Gamma width of Z boson
---------------------------------------------
This python code takes data files reads, filters and merges them. 
Then 2D fitting is carried out using chi squared minimisation.
Mesh arrays are created to calculate the uncertainties on m_zz and gamma_zz values. 
Then code plots graphs of fitting and data points. Chi-square contour elipses are plotted as well

01/05/2025
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

GAMMA_EE = 83.91 * (10**-3)  # partial width in GeV
H_REDUCED = 6.582 * (10**-24) # reduced h value in GeV/s
M_Z_START = 90 #starting value of m_Z in GeV/c^2
GAMMA_Z_START = 3 #starting value of gamma_z in GeV

def is_float(parameter):
    """
    Function checks if a parameter value is a real number.

    Parameters
    ----------
    parameter : STRING
        parameter that is to be checked.

    Returns
    -------
    bool
        if the parameter is real number returns True.

    """
    try:
        float(parameter)
        return True
    except ValueError:
        return False

def validation(parameter):
    """
    Function validates data by checking if data is a real number.

    Parameters
    ----------
    parameter : STRING
        Takes a parameter to check.

    Returns
    -------
    bool
        if parameter value is real number and not a NaN.

    """
    return bool(is_float(parameter) and not math.isnan(float(parameter)))
def read_file(file_name):
    """
    Function reads, validates and filters data from text.

    Parameters
    ----------
    file_name : STRING
        name of the file.

    Returns
    -------
    ARRAY OF FLOAT
        Array of energy values from file.
    ARRAY OF FLOAT
        Array of cross section values from file.
    ARRAY OF FLOAT
        Array of uncertainties in cross section from file.

    """
    #Reads the file
    try:
        with open(file_name, 'r') as open_file:
            data_array = np.empty((0, 3))
            for line in open_file:
                split_up = line.split(',')
                if validation(split_up[0]) and validation(split_up[1]) and validation(split_up[2]):
                    if float(split_up[2]) != 0:
                        e_value = float(split_up[0])
                        sigma_value = float(split_up[1])
                        uncertainty_value = float(split_up[2])
                        if np.abs(sigma_value - model(e_value, 91.2, 2.5)) < 3 * uncertainty_value:
                            temp = np.array([e_value, sigma_value, uncertainty_value])
                            data_array = np.vstack((data_array, temp))
        open_file.close()
        return data_array[:, 0], data_array[:, 1], data_array[:, 2]
    #Stops the program if file is not found
    except FileNotFoundError:
        print("File not found")

def merge_data():
    """
    Function merges the data from two files and sorts it by energy values.

    Returns
    -------
    merged_e : ARRAY OF FLOAT
        Array of combined energy values from files.
    merged_sigma : ARRAY OF FLOAT
        Array of combined cross section values from files.
    merged_uncertainty : ARRAY OF FLOAT
        Array of combined uncertainties values from files.

    """
    e_1, sigma_1, uncertainty_1 = read_file("z_boson_data_1.csv")
    e_2, sigma_2, uncertainty_2 = read_file("z_boson_data_2.csv")

    j = 0
    merged_e = np.empty(0)
    merged_sigma = np.empty(0)
    merged_uncertainty = np.empty(0)
    for i, value in enumerate(e_1):
        while value >= e_2[j] and j < (len(e_2) - 1):
            merged_e = np.append(merged_e, e_2[j])
            merged_sigma = np.append(merged_sigma, sigma_2[j])
            merged_uncertainty = np.append(merged_uncertainty, uncertainty_2[j])
            j += 1
        merged_e = np.append(merged_e, value)
        merged_sigma = np.append(merged_sigma, sigma_1[i])
        merged_uncertainty = np.append(merged_uncertainty, uncertainty_1[i])
    return merged_e, merged_sigma, merged_uncertainty

def model(e_value, m_z, gamma_z):
    """
    Function calculates the cross section value
    from theory formula with given 
    m_z, gamma_z and energy values.

    Parameters
    ----------
    e_value : FLOAT
        energy value.
    m_z : FLOAT
        mass of z bozon.
    gamma_z : FLOAT
        partial width of z bozon.
        
    Returns
    -------
    FLOAT
        Returns the cross section value calculated from theory.

    """
    factor = (12 * np.pi / m_z**2) * 0.3894 * (10**6)
    numerator = (e_value * GAMMA_EE)**2
    denominator = (e_value**2 - m_z**2)**2 + (m_z * gamma_z)**2
    return factor * (numerator / denominator)

def chi_square(parameters, E, sigma, uncertainty):
    """
    Function calculates the chi squared
    values with given m_z, gamma_z.

    Parameters
    ----------
    parameters : FLOAT
        fitting paramteres (m_z and gamma_z).
    E : ARRAY OF FLOAT
        Energy values.
    sigma : ARRAY OF FLOAT
        Cross section values.
    uncertainty : ARRAY OF FLOAT
        Uncertainty values.

    Returns
    -------
    FLOAT
        Chi squared value calculated with given parameters.

    """
    m_z, gamma_z = parameters
    return np.sum(((sigma - model(E, m_z, gamma_z)) / uncertainty)**2)

def min_chi_square(E, sigma, uncertainty):
    """
    Function calculates minimal chi square
   values and corresponding m_z, gamma_z values.

    Parameters
    ----------
    E : ARRAY OF FLOAT
        Energy values.
    sigma : ARRAY OF FLOAT
        Cross section values.
    uncertainty : ARRAY OF FLOAT
        Uncertainty values.

    Returns
    -------
    optimal_params : ARRAY OF FLOAT
        List of optimal m_z, gamma_z
    min_chi : FLOAT
        minimum chi squared values.

    """
    initial_values = [M_Z_START, GAMMA_Z_START]
    optimal_params, min_chi, *_ = fmin(lambda p: chi_square(p, E, sigma, uncertainty),
                                        initial_values, full_output=True, disp=False)
    return optimal_params, min_chi

def mesh_arrays(optimal_values, E, sigma, uncertainty):
    """
    Function creates mesh arrays for m_z, gamma_z and chi square.

    Parameters
    ----------
    optimal_values : ARRAY OF FLOAT
        List of optimal m_z, gamma_z
    E : ARRAY OF FLOAT
        Energy values.
    sigma : ARRAY OF FLOAT
        Cross section values.
    uncertainty : ARRAY OF FLOAT
        Uncertainty values.

    Returns
    -------
    m_mesh : 2D ARRAY OF FLOAT
        mesh array of m_z values linearly spaced.
    gamma_mesh : 2D ARRAY OF FLOAT
        mesh array of gamma_z values linearly spaced.
    chi_square_mesh : 2D ARRAY OF FLOAT
        mesh array of chi square values coresponding to m_z and gamma_z values.


    """
    m_array = np.linspace(optimal_values[0] - 0.05, optimal_values[0] + 0.05, 400)
    gamma_array = np.linspace(optimal_values[1] - 0.05, optimal_values[1] + 0.05, 400)
    m_mesh, gamma_mesh = np.meshgrid(m_array, gamma_array)
    chi_square_vectorized = np.vectorize(lambda m, g: chi_square([m, g], E, sigma, uncertainty))
    chi_square_mesh = chi_square_vectorized(m_mesh, gamma_mesh)
    return m_mesh, gamma_mesh, chi_square_mesh

def output_data(optimal_values, min_chi_square_value, E):
    """
    Function prints m_z value in GeV/c^2,
    gamma_z value in GeV,
    Reduced chi squared value, 
    Tau_z value in s.

    Parameters
    ----------
    optimal_values : ARRAY OF FLOAT
        List of optimal m_z, gamma_z
    min_chi_square_value : FLOAT
        minimum chi squared.
    E : ARRAY OF FLOAT
        Energy values.

    Returns
    -------
    None.

    """
    reduced_chi_square = min_chi_square_value / (len(E) - 2)
    tau_z = H_REDUCED / optimal_values[1]
    print(f"m_z = {optimal_values[0]:.2f} GeV/c^2")
    print(f"Gamma_z = {optimal_values[1]:.3f} GeV")
    print(f"Reduced chi-squared = {reduced_chi_square:.3f}")
    print(f"Tau_z = {tau_z:.2e} s")

def plot_data(optimal_values, min_chi_square_value, E, sigma, uncertainty):
    """
    Function plots data and fitted plot,
    plots chi square contour elipses.
    
    Parameters
    ----------
    optimal_values : ARRAY OF FLOAT
        List of optimal m_z, gamma_z
    min_chi_square_value : FLOAT
        minimum chi squared.
    E : ARRAY OF FLOAT
        Energy values.
    sigma : ARRAY OF FLOAT
        Cross section values.
    uncertainty : ARRAY OF FLOAT
        Uncertainty values.
    
    Returns
    -------
    None.

    """

    m_mesh, gamma_mesh, chi_mesh = mesh_arrays(optimal_values, E, sigma, uncertainty)
    x_array = np.linspace(85, 95, 200)

    plt.figure()
    plt.errorbar(E, sigma, yerr=uncertainty, fmt='o', markersize=3, color='purple', label="Data")
    plt.plot(x_array, model(x_array, *optimal_values), color='orange', label="Fit")
    plt.xlabel("E / GeV")
    plt.ylabel("sigma / nb")
    plt.title("Data and line of best fit")
    plt.legend()
    plt.grid()
    plt.savefig("C:/Labs/assignment_2/results/data_line_of_best_fit.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.contour(m_mesh, gamma_mesh, chi_mesh, levels=[min_chi_square_value + 1, min_chi_square_value + 2.3,
                                                     min_chi_square_value + 5.99])
    plt.xlabel("m_z / GeV/c^2")
    plt.ylabel("Gamma_z / GeV")
    plt.title("Chi-square Contour Ellipses")
    plt.grid()
    plt.savefig("C:/Labs/assignment_2/results/chi_square_contour_ellipses.png", dpi=300, bbox_inches='tight')
    plt.show()

def uncertainties_func(optimal_values, min_chi_square_value, E, sigma, uncertainty):
    """
    Function calculates the uncertainties for m_z and gamma_z.
    And prints them.

    Parameters
    ----------
    optimal_values : ARRAY OF FLOAT
        List of optimal m_z, gamma_z
    min_chi_square_value : FLOAT
        minimum chi squared.
    E : ARRAY OF FLOAT
        Energy values.
    sigma : ARRAY OF FLOAT
        Cross section values.
    uncertainty : ARRAY OF FLOAT
        Uncertainty values.
    Returns
    -------
    None.

    """
    m_mesh, gamma_mesh, chi_mesh = mesh_arrays(optimal_values, E, sigma, uncertainty)
    tolerance = 0.01
    contour_value = np.abs(chi_mesh - (min_chi_square_value + 1)) < tolerance
    m_contour = m_mesh[contour_value]
    gamma_contour = gamma_mesh[contour_value]
    m_uncertainty = (np.max(m_contour) - np.min(m_contour)) / 2
    gamma_uncertainty = (np.max(gamma_contour) - np.min(gamma_contour)) / 2
    print(f"Uncertainty in m_z: {m_uncertainty:.4f} GeV/c^2")
    print(f"Uncertainty in gamma_z: {gamma_uncertainty:.4f} GeV")

if __name__ == "__main__":
    E, sigma, uncertainty = merge_data()
    optimal_values, min_chi_sq_val = min_chi_square(E, sigma, uncertainty)
    output_data(optimal_values, min_chi_sq_val, E)
    plot_data(optimal_values, min_chi_sq_val, E, sigma, uncertainty)
    uncertainties_func(optimal_values, min_chi_sq_val, E, sigma, uncertainty)


