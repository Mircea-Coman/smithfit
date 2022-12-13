import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import brentq
from scipy.optimize import fsolve
import sys
import skrf as rf

sys.path.append("smithplot")
from smithplot import SmithAxes

def qfit6_fit_function(freqs, m, f_L):
    f_wlst = freqs[0]
    y = np.true_divide(1.0, 1 + 2j * (m[5] * freqs / f_wlst - m[4]))
    gamma = (m[0] + 1j * m[1] + (m[2] + 1j * m[3]) * y)
    return gamma


def qfit7_fit_function(freqs, m, f_L):
    f_wlst = freqs[0]
    y = np.true_divide(1.0, 1 + 2j * (m[5] * freqs / f_wlst - m[4]))
    gamma = (m[0] + 1j * m[1] + (m[2] + 1j * m[3]) * y)*np.exp(1j * m[6] * (freqs - f_L)/f_wlst)
    return gamma

def qfit8_fit_function(freqs, m, f_L):
    f_wlst = freqs[0]
    t = 2 * (freqs-f_L) / f_L
    y = 1.0 / (1 + 2j * (m[5] * freqs / f_wlst - m[4]))
    gamma = (m[0] + 1j * m[1] +  (m[6] + 1j * m[7]) * t + (m[2] + 1j * m[3]) * y)
    return gamma

def gamma_fit_function(t, a):
    return np.true_divide(a[0] * t + a[1], a[2] * t + 1)

def abs_gamma_fit_function(t,a):
    return np.absolute(gamma(t, a))

def z_to_Gamma(z):
    return np.true_divide(z-1,z+1)

# Fit using Gamma = (a1 * t + a2) / (a3 * t + 1)
def fit_gamma(ts, gammas, weights):
    shape = np.shape(ts)
    e = np.empty([shape[0],3], dtype = 'complex_')
    e[:,0] = ts
    e[:,1] = np.ones(shape)
    e[:,2] = -ts * gammas
    # P = np.diag(weights)
    # print(" Calculating the C matrix...", end="\r")
    C = make_C(e, weights)
    # print(" Calculating the g matrix...", end="\r")
    g = make_g(gammas, e, weights)
    # print("Calculating the inverse of C...", end="\r")
    C_inv = np.linalg.inv(C)
    # print(" Calculating the fit parameters...", end="\r")
    fit_params = np.matmul(C_inv, g)
    # print(" Calculating the errors...", end="\r")
    eps = fit_params[0] * ts + fit_params[1] - fit_params[2] * ts * gammas - gammas
    eps_abs_sq = eps * np.conjugate(eps)
    S_sq = np.dot(weights, eps_abs_sq)
    sigma_sq = np.empty([3,1])
    denominator = C[0,0] * C_inv[0,0] + C[1,1] * C_inv[1,1] + C[2,2] * C_inv[2,2]
    for i in range(0, 3):
        sigma_sq[i] =  np.real(C_inv[i,i] * S_sq / denominator)
    return sigma_sq[:,0], fit_params[:,0]

def fit_gamma_scalar(ts, abs_gammas_sq, weights):
    shape = np.shape(ts)
    e = np.empty([shape[0],4], dtype = 'complex_')
    e[:,0] = -ts**2 * abs_gammas_sq
    e[:,1] = ts**2
    e[:,2] = np.ones(shape[0])
    e[:,3] = ts

    # P = np.diag(weights)
    # print(" Calculating the C matrix...", end="\r")
    C = make_C(e, weights)
    # print(" Calculating the g matrix...", end="\r")
    g = make_g(abs_gammas_sq, e, weights)
    # print("Calculating the inverse of C...", end="\r")
    C_inv = np.linalg.inv(C)
    # print(" Calculating the fit parameters...", end="\r")
    fit_params = np.matmul(C_inv, g)
    # print(" Calculating the errors...", end="\r")
    eps = fit_params[0,0] * e[:,0] + fit_params[1, 0] * e[:,1] + fit_params[2, 0] * e[:,2] + fit_params[3, 0] * e[:,3] - abs_gammas_sq
    eps_abs_sq = eps * np.conjugate(eps)
    S_sq = np.dot(weights, eps_abs_sq)
    sigma_sq = np.empty([4,1])
    denominator = C[0,0] * C_inv[0,0] + C[1,1] * C_inv[1,1] + C[2,2] * C_inv[2,2] + C[3,3] * C_inv[3,3]
    for i in range(0, 4):
        sigma_sq[i] =  np.real(C_inv[i,i] * S_sq / denominator)
    return np.real(sigma_sq[:,0]), np.real(fit_params[:,0])


def make_C(e, P):
    shape = np.shape(e)
    C = np.empty([shape[1],shape[1]],dtype = 'complex_')
    for i in range(0, shape[1]):
        for j in range(0, shape[1]):
            projected_e_j = P*e[:,j]
            C[i,j] = np.matmul(np.matrix.conjugate(e[:,i]), projected_e_j)
    return C

def make_g(gammas, e, P):
    shape = np.shape(e)
    g = np.empty([shape[1],1],dtype = 'complex_')
    for i in range(0, shape[1]):
        projected_gammas = P*gammas
        g[i,0] = np.matmul(np.matrix.conjugate(e[:,i]), projected_gammas)
    return g

# Calculates the center and radius of a circle given 3 points in the complex plane
def circle_from_3_points(z1, z2, z3):
    x1 = np.real(z1);
    y1 = np.imag(z1);
    x2 = np.real(z2);
    y2 = np.imag(z2);
    x3 = np.real(z3);
    y3 = np.imag(z3);
    c = (x1-x2)**2 + (y1-y2)**2
    a = (x2-x3)**2 + (y2-y3)**2
    b = (x3-x1)**2 + (y3-y1)**2
    s = 2*(a*b + b*c + c*a) - (a*a + b*b + c*c)
    px = (a*(b+c-a)*x1 + b*(c+a-b)*x2 + c*(a+b-c)*x3) / s
    py = (a*(b+c-a)*y1 + b*(c+a-b)*y2 + c*(a+b-c)*y3) / s
    ar = a**0.5
    br = b**0.5
    cr = c**0.5
    r = ar*br*cr / ((ar+br+cr)*(-ar+br+cr)*(ar-br+cr)*(ar+br-cr))**0.5
    return (px + 1j * py), r

def calculate_t_intersection(params, z1, z2):
    z = z1-z2
    # z = z1
    # alpha = np.arctan2(np.imag(z),np.real(z))
    gamma_int = z1 - 2 * z
    t_int = np.real((gamma_int - params[1])/(params[0]-gamma_int*params[2]))
    return t_int, gamma_int

def tan_equations(v, Gamma_d, center_Gamma, radius_Gamma):
    r, x03, y03 = v
    z1 = x03 + y03 * 1j
    eq1 = np.absolute(z1 - center) - (r - radius_Gamma)
    eq2 = np.absolute(z1) - (1 - r)
    eq3 = np.absolute(Gamma_d - z1) - r
    return eq1, eq2, eq3

def touching_circle_center(Gamma_s, center, radius_tan):
    m = (np.real(Gamma_s) - np.real(center))/(np.imag(Gamma_s) - np.imag(center))
    y03_minus = np.imag(Gamma_s) - radius_tan/np.sqrt(m**2 + 1)
    x03_minus = np.real(Gamma_s) - m * (np.imag(Gamma_s) - y03_minus)

    y03_plus = np.imag(Gamma_s) + radius_tan/np.sqrt(m**2 + 1)
    x03_plus = np.real(Gamma_s) - m * (np.imag(Gamma_s) - y03_plus)

    if(np.absolute(x03_minus + y03_minus * 1j) < np.absolute(x03_plus + y03_plus * 1j)):
        center_tan = x03_minus + 1j * y03_minus
    else:
        center_tan = x03_plus + 1j * y03_plus
    return center_tan


def find_tangent_circle(Gamma_d, Gamma_l, center, radius):
    Gamma_s = Gamma_d
#    cosPhi1 = (np.abs(Gamma_s)**2 + 4*radius ** 2 - np.abs(Gamma_l) ** 2)/(4 * radius * np.abs(Gamma_s))
    cosPhi = np.real(Gamma_d * np.conj(Gamma_d-center)) / (np.abs(Gamma_d * (Gamma_d-center)))
    radius_tan = (1 - np.absolute(Gamma_s)**2)/(1 - np.absolute(Gamma_s) * cosPhi)/2.0
    center_tan = touching_circle_center(Gamma_s, center, radius_tan)
    return center_tan, radius_tan

def delta(freq, freq0):
    return freq / freq0 - np.true_divide(freq0,freq)

def S11dB_fit_func(freqs, freq0, Q0, beta, zero_level):
    deltas = delta(freqs, freq0);
    numerator = beta - 1 - 1j * Q0 * deltas
    numitor = beta + 1 + 1j * Q0 * deltas
    return 10*np.log10(np.absolute(np.true_divide(numerator,numitor))) + zero_level

def gamma_to_S11dB(gammas):
    return 10*np.log10(np.absolute(gammas))


def center_from_points_and_r(z1, z2, r):
    x1 = np.real(z1)
    y1 = np.imag(z1)
    x2 = np.real(z2)
    y2 = np.imag(z2)

    q = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2

    xx = (r ** 2 - (q / 2) ** 2) ** 0.5 * (y1 - y2) / q
    yy = (r ** 2 - (q / 2) ** 2) ** 0.5 * (x2 - x1) / q
    return (x3 + xx) + 1j * (y3 + yy)
    # return (x3 + xx, y3 + yy) + 1j* (x3 - xx, y3 - yy)

# Plots the Smith chart
def plot_step(gammas, Gamma_d, Gamma_l, center, radius, center_tan, radius_tan, measurement_type, method, freqs, f_L, fit_params, show_touching_circle = True):

    thetas = np.linspace(0, 2*np.pi, num = 1000)

    if method == 'kajfez':
            circle_zs = center + radius * np.exp(1j * thetas)
    elif method == 'qfit6':
            fit_freqs = np.linspace(min(freqs), max(freqs), num = 1000)
            circle_zs = qfit6_fit_function(freqs, fit_params, f_L)
    elif method == 'qfit7':
            fit_freqs = np.linspace(min(freqs), max(freqs), num = 1000)
            circle_zs = qfit7_fit_function(freqs, fit_params, f_L)
    elif method == 'qfit8':
            fit_freqs = np.linspace(min(freqs), max(freqs), num = 1000)
            circle_zs = qfit8_fit_function(freqs, fit_params, f_L)

    line_t = np.linspace(0,1,num = 1000)
    line1_zs = (1 - line_t) * Gamma_l + Gamma_d * line_t
    line2_zs = (1 - line_t) * Gamma_l + 0 * line_t
    circle_tan_zs = center_tan + radius_tan * np.exp(1j * thetas)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1, projection='smith')
    ax.set_aspect('equal', adjustable='box')
    plt.plot(line1_zs, datatype=SmithAxes.S_PARAMETER,  marker='None',color='k',linewidth=2, linestyle='-')
    plt.plot(line2_zs, datatype=SmithAxes.S_PARAMETER,  marker='None',color='k',linewidth=2, linestyle='-')
    plt.plot(gammas, datatype=SmithAxes.S_PARAMETER,  marker='o',color='k', linestyle='None',markersize=5, label = 'Data points')

    plt.plot(circle_zs, datatype=SmithAxes.S_PARAMETER,  marker='None', color='red',linewidth=2, linestyle='-', label = 'Fit circle')
    plt.plot([center], datatype=SmithAxes.S_PARAMETER,  marker='o',color='red',linewidth=2, linestyle='None',markersize=10, label="Center of fit circle")
    if (measurement_type == 'reflection' or  measurement_type == 'reflection_method1' or measurement_type == 'reflection_method2') and show_touching_circle:
        plt.plot(circle_tan_zs, datatype=SmithAxes.S_PARAMETER,  marker='None',color='forestgreen',linewidth=2, linestyle='-.', label = 'Touching Circle')
    # plt.plot([gamma_int], datatype=SmithAxes.S_PARAMETER,  marker='o',color='darkviolet',linewidth=2, linestyle='None',markersize=10, label="Center of fit circle")

    plt.plot([Gamma_d], datatype=SmithAxes.S_PARAMETER,  marker='o',color='forestgreen',linewidth=2, linestyle='None',markersize=10, label="$\Gamma_d$")
    plt.plot([Gamma_l], datatype=SmithAxes.S_PARAMETER,  marker='o',color='darkorange',linewidth=2, linestyle='None',markersize=10, label="$\Gamma_l$")
    ax.legend(bbox_to_anchor=(1.08, 1.08))

    lim = 1E-3
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])

    plt.show()
    plt.pause(0.001)

def plot_polar(gammas, Gamma_d, Gamma_l, center, radius, center_tan, radius_tan, measurement_type, method, freqs, f_L, fit_params, show_touching_circle = True):
    thetas = np.linspace(0, 2*np.pi, num = 1000)

    if method == 'kajfez':
            circle_zs = center + radius * np.exp(1j * thetas)
    elif method == 'qfit6':
            fit_freqs = np.linspace(min(freqs), max(freqs), num = 1000)
            circle_zs = qfit6_fit_function(freqs, fit_params, f_L)
    elif method == 'qfit7':
            fit_freqs = np.linspace(min(freqs), max(freqs), num = 1000)
            circle_zs = qfit7_fit_function(freqs, fit_params, f_L)
    elif method == 'qfit8':
            fit_freqs = np.linspace(min(freqs), max(freqs), num = 1000)
            circle_zs = qfit8_fit_function(freqs, fit_params, f_L)

    # circle_zs = center + radius * np.exp(1j * thetas)
    line_t = np.linspace(0,1,num = 1000)
    line1_zs = (1 - line_t) * Gamma_l + Gamma_d * line_t
    line2_zs = (1 - line_t) * Gamma_l + 0 * line_t
    circle_tan_zs = center_tan + radius_tan * np.exp(1j * thetas)
    circle_unit = 0+0j  + 1 * np.exp(1j * thetas)
    fig = plt.figure(figsize=(8, 8))
    # ax = plt.subplot(1, 1, 1, projection='smith')
    ax = plt.subplot(1, 1, 1)
    # rect = [0, 0, 1, 1]
    #
    # x_polar = fig.add_axes(rect, polar=True, frameon=False)

    ax.set_aspect('equal', adjustable='box')
    plt.plot(np.real(gammas), np.imag(gammas),  marker='o',color='k', linestyle='None',markersize=5, label = 'Data points')
    plt.plot(circle_zs.real, circle_zs.imag,  marker='None',color='red',linewidth=2, linestyle='-', label = 'Fit circle')
    plt.plot(circle_unit.real, circle_unit.imag,  marker='None',color='k',linewidth=2, linestyle='-', label = 'Unit Circle')
    plt.plot([center.real], [center.imag],  marker='o',color='red',linewidth=2, linestyle='None',markersize=10, label="Center of fit circle")
    plt.plot(line1_zs.real,line1_zs.imag,  marker='None',color='k',linewidth=2, linestyle='-')
    plt.plot(line2_zs.real,line2_zs.imag,  marker='None',color='k',linewidth=2, linestyle='-')
    if (measurement_type == 'reflection' or  measurement_type == 'reflection_method1' or measurement_type == 'reflection_method2') and show_touching_circle:
        plt.plot(circle_tan_zs.real,circle_tan_zs.imag,  marker='None',color='forestgreen',linewidth=2, linestyle='-.', label = 'Tangent Circle')
    # # plt.plot([gamma_int], datatype=SmithAxes.S_PARAMETER,  marker='o',color='darkviolet',linewidth=2, linestyle='None',markersize=10, label="Center of fit circle")
    plt.plot([Gamma_d.real],[Gamma_d.imag], marker='o',color='forestgreen',linewidth=2, linestyle='None',markersize=10, label="$\Gamma_d$")
    plt.plot([Gamma_l.real],[Gamma_l.imag],  marker='o',color='darkorange',linewidth=2, linestyle='None',markersize=10, label="$\Gamma_l$")
    ax.legend(loc = "upper right")
    #
    # lim = 1E-3
    # ax.set_xlim([-lim, lim])
    # ax.set_ylim([-lim, lim])

    plt.show()
    plt.pause(0.001)

def get_data_from_s1p(file, header=0, footer=0, every=1, bandwidth=0):
    try:
        network = rf.Network(file)
    except FileNotFoundError:
        print('File not found!')
        sys.exit(-1)
    except:
        print('Can\'t read this file! Make sure that the file has valid touchstone format headers!')
        sys.exit(-1)
    if footer == 0:
        freqs = network.f[header::every]
        gammas = network.s[header::every, 0, 0]
    else:
        freqs = network.f[header:-footer:every]
        gammas = network.s[header:-footer:every, 0, 0]

    if bandwidth != 0:
        abs_gammas = np.abs(gammas)
        min_freq = freqs[np.argmin(abs_gammas)]
        bool_array =  get_prune_range(freqs, min_freq, bandwidth)

        return freqs[bool_array], gammas[bool_array]
    else:
        return freqs, gammas


def get_prune_range(freqs, f0, bw):
    return  (freqs < f0 + bw) * (f0 - bw  < freqs)


def get_data_from_s2p(file, header = 0, footer = 0, every = 1, bandwidth = 0):
    try:
        network = rf.Network(file)
    except:
        print('Can\'t read the file! Make sure that the file exists and that it is has valid touchstone format headers!')
        sys.exit(-1)
    if footer == 0:
        freqs = network.f[header::every]
        S11 = network.s[header::every, 0, 0]
        S21 = network.s[header::every, 1, 0]
        S12 = network.s[header::every, 0, 1]
        S22 = network.s[header::every, 1, 1]
    else:
        freqs = network.f[header:-footer:every]
        S11 = network.s[header:-footer:every, 0, 0]
        S21 = network.s[header:-footer:every, 1, 0]
        S12 = network.s[header:-footer:every, 0, 1]
        S22 = network.s[header:-footer:every, 1, 1]

    if bandwidth != 0:
        abs_S21 = np.abs(S21)
        max_freq = freqs[np.argmax(abs_S21)]
        bool_array =  get_prune_range(freqs, max_freq, bandwidth)
        return freqs[bool_array], S11[bool_array], S21[bool_array], S12[bool_array], S22[bool_array]
    else:
        return freqs, S11, S21, S12, S22

        #def get_data_from_s1p(file, header, footer, every, comments, delimiter):
#    data = np.genfromtxt(open(file,'rt').readlines()[::every], comments = comments, delimiter = delimiter, skip_header = header, skip_footer = footer)
#    gammas = data[:, 1] + data[:, 2] * 1j
#    freqs = data[:, 0]
#    return freqs, gammas

#def get_data_from_s2p(file, header, footer, every, comments, delimiter):
#    data = np.genfromtxt(open(file,'rt').readlines()[::every], comments = comments, delimiter = delimiter, skip_header = header, skip_footer = footer)
#    S11 = data[:, 1] + data[:, 2] * 1j
#    S21 = data[:, 3] + data[:, 4] * 1j
#   S12 = data[:, 5] + data[:, 6] * 1j
#    S22 = data[:, 7] + data[:, 8] * 1j
#    freqs = data[:, 0]
#    return freqs, S11, S21, S12, S22
