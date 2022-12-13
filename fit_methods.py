import numpy as np
from matplotlib import pyplot as plt
import os

from utils import *

this_file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(this_file_path, 'MAT58_Scripts'))
from test_qfit6 import fit_qfit6
from test_qfit7 import fit_qfit7
from test_qfit8 import fit_qfit8
from scipy.optimize import curve_fit

from test_qfit6 import fit_qfit6
from test_qfit7 import fit_qfit7
from test_qfit8 import fit_qfit8


def fit_and_display_full_transmission(freqs, S11, S21, S12, S22,  n_steps = 3, loop_plan = 'fwfwfwc', scaling_factor = 1.0, method = 'kajfez', prune_last_step = False,\
 save_plots=False, out_folder='', out_name = 'output', out_format='png', hide_plots = False, no_smith = False, quiet = False):
 if method == 'kajfez':
     S11_dict = fit_and_display_kajfez(freqs, S11, measurement_type = 'reflection', lossless = True, n_steps=n_steps, prune_last_step = prune_last_step, quiet = True, hide_plots = True,)
     S12_dict = fit_and_display_kajfez(freqs, S12, measurement_type = 'transmission',lossless = True, n_steps=n_steps, prune_last_step = prune_last_step, quiet = True, hide_plots = True)
     S22_dict = fit_and_display_kajfez(freqs, S22, measurement_type = 'reflection', lossless = True, n_steps=n_steps, prune_last_step = prune_last_step, quiet = True, hide_plots = True)
 elif method == 'qfit6' or method == 'qfit7' or method == 'qfit8':
     # if method == 'qfit6': function = fit_qfit6
     # elif method == 'qfit7': function = fit_qfit7
     # elif method == 'qfit8': function = fit_qfit8
     # else:
     #     print("Invalid method!")
     #     return
     if not quiet: print("Fitting S11 data...")
     S11_dict = fit_and_display_qfit(freqs, S11, method, measurement_type = 'reflection_method1', scaling_factor = 'auto', loop_plan = loop_plan, quiet = False, hide_plots = True)
     if not quiet: print("Fitting S12 data...")
     S12_dict = fit_and_display_qfit(freqs, S12, method, measurement_type = 'transmission', scaling_factor = scaling_factor, loop_plan = loop_plan, quiet = False, hide_plots = True)
     if not quiet: print("Fitting S22 data...")
     S22_dict = fit_and_display_qfit(freqs, S22, method, measurement_type = 'reflection_method1', scaling_factor = 'auto', loop_plan = loop_plan, quiet = False, hide_plots = True)
 else:
     print("Invalid method!")
     sys.exit(-1)

 thetas = np.linspace(0, 2*np.pi, num = 1000)

 f_L = S12_dict["f_L"]
 den = (2.0 - S11_dict["diam"] - S22_dict["diam"])
 sigma_den = np.sqrt(S11_dict["sigma_diam"]**2 + S22_dict["sigma_diam"]**2 )

 beta1 = S11_dict["diam"] / den
 beta2 = S22_dict["diam"] / den
 Q_L = S12_dict["Q_L"]
 Q_0 = 2 * S12_dict["Q_L"] / den
 d1 = S11_dict["diam"]
 d2 = S22_dict["diam"]
 d12 = S12_dict["diam"]
 sigma_Q_L = S12_dict["sigma_Q_L"]
 sigma_beta1 = beta1 * np.sqrt((S11_dict["sigma_diam"] / S11_dict["diam"])**2 + (sigma_den / den)**2)
 sigma_beta2 = beta2 * np.sqrt((S22_dict["sigma_diam"] / S22_dict["diam"])**2 + (sigma_den / den)**2)
 sigma_Q_0 = Q_0 * np.sqrt((S12_dict["sigma_Q_L"] / S12_dict["Q_L"])**2 + (sigma_den / den)**2)

 if not quiet:
     if method == 'kajfez':
         print("Done")
         print(f" d1 = {d1:.6f}")
         print(f" d2 = {d2:.6f}")
         print(f" d12 = {d12:.6f}")
         print(f" Q_L = {Q_L:.6f} +/- {sigma_Q_L:.6f}")
         print(f" beta_1 = {beta1:.6f} +/- {sigma_beta1:.6f}")
         print(f" beta_2 = {beta2:.6f} +/- {sigma_beta2:.6f}")
         print(f" Q_0 = {Q_0:.6f} +/- {sigma_Q_0:.6f}")
         print(f" f_L = {f_L *1E-9:.8f} GHz")
         print()
     elif method == 'qfit6' or method == 'qfit7' or method == 'qfit8':
         print("Done")
         print(f" d1 = {d1:.6f}")
         print(f" d2 = {d2:.6f}")
         print(f" d12 = {d12:.6f}")
         print(f" Q_L = {Q_L:.6f}")
         print(f" beta_1 = {beta1:.6f}")
         print(f" beta_2 = {beta2:.6f}")
         print(f" Q_0 = {Q_0:.6f}")
         print(f" f_L = {f_L *1E-9:.8f} GHz")
         print()


 #fig = plt.figure(figsize=(8, 8))
 #ax = plt.subplot(1, 1, 1, projection='smith')
 #ax.set_aspect('equal', adjustable='box')
 #plt.plot(S11, datatype=SmithAxes.S_PARAMETER,  marker='o',color='green', linestyle='None',markersize=8, label = 'S11')
 #plt.plot(S12, datatype=SmithAxes.S_PARAMETER,  marker='o',color='darkcyan', linestyle='None',markersize=8, label = 'S12')
 #plt.plot(S22, datatype=SmithAxes.S_PARAMETER,  marker='o',color='orange', linestyle='None',markersize=8, label = 'S22')
 #plt.plot(circle_S11, datatype=SmithAxes.S_PARAMETER,  marker='None',color='red', linestyle='-', linewidth=1, label = 'Fit Circles')
 #plt.plot(circle_S12, datatype=SmithAxes.S_PARAMETER,  marker='None',color='red', linestyle='-', linewidth=1)
 #plt.plot(circle_S22, datatype=SmithAxes.S_PARAMETER,  marker='None',color='red', linestyle='-', linewidth=1)

 if method == 'kajfez':
          circle_S11 = S11_dict["center"] + 0.5*S11_dict["diam"] * np.exp(1j * thetas)
          circle_S12 = S12_dict["center"] + 0.5*S12_dict["diam"] * np.exp(1j * thetas)
          circle_S22 = S22_dict["center"] + 0.5*S22_dict["diam"] * np.exp(1j * thetas)
 elif method == 'qfit6':
         fit_freqs = np.linspace(min(freqs), max(freqs), num = 1000)
         circle_S11 = qfit6_fit_function(freqs, S11_dict["coeffs"], S11_dict["f_L"])
         circle_S12 = qfit6_fit_function(freqs, S12_dict["coeffs"], S12_dict["f_L"])
         circle_S22 = qfit6_fit_function(freqs, S22_dict["coeffs"], S22_dict["f_L"])
 elif method == 'qfit7':
         fit_freqs = np.linspace(min(freqs), max(freqs), num = 1000)
         circle_S11 = qfit7_fit_function(freqs, S11_dict["coeffs"], S11_dict["f_L"])
         circle_S12 = qfit7_fit_function(freqs, S12_dict["coeffs"], S12_dict["f_L"])
         circle_S22 = qfit7_fit_function(freqs, S22_dict["coeffs"], S22_dict["f_L"])
 elif method == 'qfit8':
         fit_freqs = np.linspace(min(freqs), max(freqs), num = 1000)
         circle_S11 = qfit8_fit_function(freqs, S11_dict["coeffs"], S11_dict["f_L"])
         circle_S12 = qfit8_fit_function(freqs, S12_dict["coeffs"], S12_dict["f_L"])
         circle_S22 = qfit8_fit_function(freqs, S22_dict["coeffs"], S22_dict["f_L"])

 fig_name = os.path.join(out_folder, f"smithfit_{out_name}.{out_format}")
 fig = plt.figure(figsize=(8, 8))
# ax = plt.subplot(1, 1, 1)
 ax = fig.subplots(subplot_kw={'projection': 'polar'})
 ax.plot(np.angle(S11), np.abs(S11),  marker='o',color='green', linestyle='None',markersize=8, label = 'S11')
 ax.plot(np.angle(S12), np.abs(S12), marker='o',color='steelblue', linestyle='None',markersize=8, label = 'S12')
 ax.plot(np.angle(S22), np.abs(S22),  marker='o',color='orange', linestyle='None',markersize=8, label = 'S22')
 ax.plot(np.angle(circle_S11), np.abs(circle_S11),  marker='None',color='red', linestyle='-', linewidth=2, label = 'Fit Circles')
 ax.plot(np.angle(circle_S12), np.abs(circle_S12), marker='None',color='red', linestyle='-', linewidth=2)
 ax.plot(np.angle(circle_S22), np.abs(circle_S22),  marker='None',color='red', linestyle='-', linewidth=2)
 ax.legend(bbox_to_anchor=(1.08, 1.08))
 ax.set_ylim([0,1])
 if(save_plots): plt.savefig(fig_name, dpi = 100,bbox_inches='tight')
 if not hide_plots: plt.show(block=True)

 results_dict = {
     "f_L": f_L,
     "Q_L": Q_L,
     "Q_0": Q_0,
     "d1":   d1,
     "d2":   d2,
     "d12":   d12,
     "beta_1": beta1,
     "sigma_beta_1": sigma_beta1,
     "sigma_beta_2": sigma_beta2,
     "beta_2": beta2,
     "sigma_Q_L": sigma_Q_L,
     "sigma_Q_0": sigma_Q_0,
    }

 return results_dict

def fit_and_display_scalar(freqs, gammas, scale = 1, n_steps = 3,\
 plot_every_step=True, save_plots=False, out_folder='', out_name = 'output', out_format='png', hide_plots = False, quiet = False):
    # head, out_name = os.path.split(file) #find file name

    plt.ion()
    # freqs, gammas = get_data_from_file(file, header, footer, every, comments, delimiter)
    # gammas = 60 * gammas
    freqs = freqs * scale
    gamma_abs_sq = np.real(gammas)*np.real(gammas) + np.imag(gammas) * np.imag(gammas)

    #First guess for f_L
    S11dBs = gamma_to_S11dB(gammas)

    f_L = freqs[np.argmin(S11dBs)]

    ts = 2 * (freqs - f_L)/f_L
    ps = 1.0 / (ts**4 * gamma_abs_sq + ts**4 + 1 + ts**2)
    # alpha = np.radians(50)
    # dist_from_center = 0.98
    # gammas = gamma_fit_function(ts, [-1000 * dist_from_center*(np.cos(alpha) + 1j* np.sin(alpha)), -dist_from_center*(np.cos(alpha) + 1j* np.sin(alpha))+(np.cos(alpha) + 1j* np.sin(alpha))+0.1, 1000j])
    # gamma_abs_sq = np.real(gammas)*np.real(gammas) + np.imag(gammas) * np.imag(gammas)
    # gamma_abs_sq = np.real(gammas)*np.real(gammas) + np.imag(gammas) * np.imag(gammas)

    for i in range(0,n_steps):
        if not quiet:
            print(f"Step {i+1}/{n_steps}")

        sigma_sq, params_Gamma = fit_gamma_scalar(ts, gamma_abs_sq, ps)
        gammas_fit = np.sqrt(np.true_divide(params_Gamma[1] * ts** 2 + params_Gamma[2] + params_Gamma[3] * ts, params_Gamma[0] * ts**2 + 1))
        S11dBs_fit = gamma_to_S11dB(gammas_fit)


        t_int = 0.5 * params_Gamma[3] / (params_Gamma[0] * params_Gamma[2] - params_Gamma[1])
        # Calculate the new frequency
        f_L = (1 + 0.5 * t_int)*f_L

        # Calculate the new t's
        ts = 2 * (freqs - f_L)/f_L

        # Calculate the new weights
        ps = 1.0 / (ts**4 * gamma_abs_sq * sigma_sq[0] + sigma_sq[1] * ts**4 + sigma_sq[2] + sigma_sq[3] * ts**2)

        # Calculate and show the quality factors and kappa
        Q_L = np.sqrt(params_Gamma[0])
        rho = np.sqrt(params_Gamma[1]/params_Gamma[0])
        d = rho + np.sqrt(params_Gamma[2])
        d2 = 1 + rho

        sigma_Q_L = np.sqrt(sigma_sq[2])
        sigma_d_sq = sigma_sq[0]/np.absolute(params_Gamma[2])**2 + sigma_sq[1] + sigma_sq[2] * np.absolute(params_Gamma[0]/params_Gamma[2]**2)**2

        kappa = 1.0 / (d2/d - 1)
        sigma_kappa =d2 * np.sqrt(sigma_d_sq) / (d2 - d)**2
        Q_0 = Q_L * (1.0 + kappa)
        sigma_Q_0 = np.sqrt(sigma_Q_L**2 * (1 + kappa)**2 + Q_L**2 * sigma_kappa ** 2)
        kappa_s = 2.0/d2-1

        if not quiet:
		#	print(f" d = {d:.6f}")
            print(f" d = {d:.6f}")

            print(f" Q_L = {Q_L:.6f} +/- {sigma_Q_L:.6f}")
            print(f" k = {kappa:.6f} +/- {sigma_kappa:.6f}")
            print(f" Q_0 = {Q_0:.6f} +/- {sigma_Q_0:.6f}")
            print(f" fL = {f_L*1E-9:.8f} GHz")
            print(f" ks = {kappa_s:.6f}")
            print()

        if (plot_every_step or i + 1 == n_steps) and not hide_plots:
            #gammas_fit = np.sqrt(np.true_divide(params_Gamma[1] * ts** 2 + params_Gamma[2] + params_Gamma[3] ** ts, params_Gamma[0] * ts**2 + 1))
            fig = plt.figure(figsize=(12, 8))
            plt.plot(freqs*1E-9, S11dBs, color = 'k', label = 'Data')
            plt.plot(freqs*1E-9, S11dBs_fit, color = 'r', label = 'Fit')
            plt.plot([f_L*1E-9, f_L*1E-9], [0, -6], color = 'g', label = 'Fit')
            plt.legend()
            plt.xlabel('Frequency [GHz]')
            plt.ylabel('S11 [dB]')
            plt.show(block=True)

            fig_name = os.path.join(out_folder, f"smithfit_{out_name}_step{i+1}.{out_format}")
            if(save_plots):
                plt.savefig(fig_name, dpi = 100)
            if(i + 1 == n_steps):
                plt.show(block=True)

    results_dict = {
        "coeffs": params_Gamma,
        "sigma_sq_coeffs": sigma_sq,
        "f_L": f_L,
        "Q_L": Q_L,
        "Q_0": Q_0,
        "diam":   d,
         "k":   kappa,
        "sigma_Q_L": sigma_Q_L,
        "sigma_diam": np.sqrt(sigma_d_sq),
        "sigma_k": sigma_kappa,
        "sigma_Q_0": sigma_Q_0,
        }
    return results_dict



def fit_and_display_kajfez(freqs, gammas, measurement_type = 'reflection', n_steps = 3, prune_last_step = False, lossless = False, \
 plot_every_step=False, save_plots=False, out_folder='', out_name = 'output', out_format='png', hide_plots = False, no_smith = False, quiet = False):
    # head, out_name = os.path.split(file) #find file name

    plt.ion()
    freqs =  freqs
    gamma_abs_sq = np.real(gammas)*np.real(gammas) + np.imag(gammas) * np.imag(gammas)

    #First guess for f_L
    S11dBs = gamma_to_S11dB(gammas)

    if measurement_type == 'reflection':
        f_L = freqs[np.argmin(S11dBs)]
    elif measurement_type == 'transmission':
        f_L = freqs[np.argmax(S11dBs)]

    ts = 2 * (freqs - f_L)/f_L
    ps = 1.0 / (1.0 + ts**2 * (1 + gamma_abs_sq))
    # alpha = np.radians(50)
    # dist_from_center = 0.98
    # gammas = gamma_fit_function(ts, [-1000 * dist_from_center*(np.cos(alpha) + 1j* np.sin(alpha)), -dist_from_center*(np.cos(alpha) + 1j* np.sin(alpha))+(np.cos(alpha) + 1j* np.sin(alpha))+0.1, 1000j])
    # gamma_abs_sq = np.real(gammas)*np.real(gammas) + np.imag(gammas) * np.imag(gammas)
    for i in range(0,n_steps):
        is_last_step = (i + 1 == n_steps)

        if not quiet:
            print(f"Step {i+1}/{n_steps}")

        if is_last_step and prune_last_step:
            bw = f_L/Q_L
            if not quiet: print(f"Last step: Removing data outside the bandwidth (BW = {bw*1E-6:.6f} MHZ)...")
            bool_array = get_prune_range(freqs, f_L, bw)
            ts = ts[bool_array]
            gammas=gammas[bool_array]
            ps = ps[bool_array]
            #
            # sigma_sq, params_Gamma = fit_gamma(ts[bool_array], gammas[bool_array], ps[bool_array])

        sigma_sq, params_Gamma = fit_gamma(ts, gammas, ps)
        Gamma_d = params_Gamma[0]/params_Gamma[2]
        Gamma_l = params_Gamma[1]
        center, radius = circle_from_3_points(Gamma_d, Gamma_l, gamma_fit_function(0.1, params_Gamma))

        t_int, Gamma_int = calculate_t_intersection(params_Gamma, Gamma_d, center)

        # Calculate the new weights
        if not is_last_step: ps = 1.0 / (ts**2 * sigma_sq[0] + sigma_sq[1] + ts**2 * gamma_abs_sq * sigma_sq[2])

        # Calculate the new frequency
        if not is_last_step: f_L = (1 + 0.5 * t_int)*f_L

        # Calculate the new t's
        if not is_last_step: ts = 2 * (freqs - f_L)/f_L

        # Get the circle tangent to both the unit circle and the fitted circle
        # center_tan, radius_tan = find_tangent_circle(Gamma_d, center, radius)

        if lossless:
            center_tan = 0 + 0j
            radius_tan = 1.0
        else:
            if measurement_type == 'reflection':
                center_tan, radius_tan = find_tangent_circle(Gamma_d, Gamma_l, center, radius)
                show_touching_circle = True
            elif measurement_type == 'transmission':
                center_tan, radius_tan = 0+0j, 2
        show_touching_circle = (measurement_type == 'reflection')

 #      radius_tan = 1
  #      center_tan = 0 + 0j
        # Calculate and show the quality factors and kappa
        Q_L = np.imag(params_Gamma[2])
        sigma_Q_L = np.sqrt(sigma_sq[2])
        d = np.absolute(Gamma_l - Gamma_d)
        sigma_d_sq = sigma_sq[0]/np.absolute(params_Gamma[2])**2 + sigma_sq[1] + sigma_sq[2] * np.absolute(params_Gamma[0]/params_Gamma[2]**2)**2

        if measurement_type == 'reflection':
            kappa = 1.0 / (2.0 * radius_tan/d - 1)
            sigma_kappa = 2.0 * radius_tan * np.sqrt(sigma_d_sq) / (2.0 * radius_tan - d)**2
            Q_0 = Q_L * (1.0 + kappa)
            sigma_Q_0 = np.sqrt(sigma_Q_L**2 * (1 + kappa)**2 + Q_L**2 * sigma_kappa ** 2)
            kappa_s = 1.0/radius_tan-1

        elif measurement_type == 'transmission':
            kappa = np.abs(Gamma_l) / (1 - np.abs(Gamma_l))
            Q_0 = Q_L / (1.0 - d)
            sigma_Q_0 = Q_0 * np.sqrt(sigma_Q_L**2 / Q_L**2 + sigma_d_sq / (1-d)**2)
        if not quiet:
		#	print(f" d = {d:.6f}")
            print(f" Diameter of the Q circle = {d:.6f}")
            if (show_touching_circle): print(f" Diameter of the touching circle = {(2.0*radius_tan):.6f}")
            print(f" Q_L = {Q_L:.6f} +/- {sigma_Q_L:.6f}")
            if measurement_type == 'reflection': print(f" k = {kappa:.6f} +/- {sigma_kappa:.6f}")
            print(f" Q_0 = {Q_0:.6f} +/- {sigma_Q_0:.6f}")
            print(f" fL = {f_L*1E-9:.8f} GHz")
          #  if measurement_type == 'reflection': print(f" ks = {kappa_s:.6f}")
            print()

        if (plot_every_step or is_last_step) and not hide_plots:
            if no_smith:
                plot_polar(gammas, Gamma_d, Gamma_l, center, radius, center_tan, radius_tan-0.00001, measurement_type, 'kajfez', freqs, f_L, params_Gamma, show_touching_circle = show_touching_circle)
            else:
                plot_step(gammas, Gamma_d, Gamma_l, center, radius, center_tan, radius_tan-0.00001, measurement_type, 'kajfez', freqs, f_L, params_Gamma, show_touching_circle = show_touching_circle)
            fig_name = os.path.join(out_folder, f"smithfit_{out_name}_step{i+1}.{out_format}")
            if(save_plots):
                plt.savefig(fig_name, dpi = 100,bbox_inches='tight')
            if(i + 1 == n_steps):
                plt.show(block=True)

    results_dict = {
        "coeffs": params_Gamma,
        "sigma_sq_coeffs": sigma_sq,
        "f_L": f_L,
        "Q_L": Q_L,
        "Q_0": Q_0,
        "diam":   d,
        "center": (Gamma_l + Gamma_d)/2.0,
        # "k":   kappa,
        "sigma_Q_L": sigma_Q_L,
        "sigma_diam": np.sqrt(sigma_d_sq),
        # "sigma_k": sigma_kappa,
        "sigma_Q_0": sigma_Q_0,
    }
    if measurement_type == 'reflection':
        results_dict["k"] = kappa
        results_dict["sigma_k"] = sigma_kappa
    return results_dict

def fit_and_display_qfit(freqs, gammas, method, measurement_type = 'reflection_method1', loop_plan = 'fwfwfwc', scaling_factor = 'Auto',\
 save_plots=False, out_folder='', out_format='png', out_name = 'output', hide_plots = False, no_smith = False, quiet = False):
    plt.ion()
    if method == 'qfit6': function = fit_qfit6
    elif method == 'qfit7': function = fit_qfit7
    elif method == 'qfit8': function = fit_qfit8
    results_dict = function(freqs, gammas, quiet = quiet, loop_plan = loop_plan, scaling_factor_A = scaling_factor, trmode = measurement_type)

    center = center_from_points_and_r(results_dict["Gamma_d"], results_dict["Gamma_l"], results_dict["diam"]/2.0)

    if measurement_type == 'reflection_method1' or measurement_type == 'reflection_method2':
        touching_center = touching_circle_center(results_dict["Gamma_d"], center, results_dict["touch_diam"]/2.0)
    elif measurement_type == 'transmission':
        touching_center, radius_tan = find_tangent_circle(results_dict["Gamma_l"], results_dict["Gamma_d"], center, results_dict["diam"]/2.0)
        results_dict["touch_diam"] = 2 * radius_tan

    show_touching_circle =  (measurement_type == 'reflection_method2')

    if not hide_plots:
        if no_smith:
             plot_polar(gammas, results_dict["Gamma_d"], results_dict["Gamma_l"], center, results_dict["diam"]/2.0, \
              touching_center, results_dict["touch_diam"]/2.0-0.00001, measurement_type, method, freqs, results_dict["f_L"], results_dict["coeffs"], show_touching_circle = show_touching_circle)
        else:
            plot_step(gammas, results_dict["Gamma_d"], results_dict["Gamma_l"], center, results_dict["diam"]/2.0, \
             touching_center, results_dict["touch_diam"]/2.0-0.00001, measurement_type, method, freqs, results_dict["f_L"], results_dict["coeffs"], show_touching_circle = show_touching_circle)

    results_dict["center"] = (results_dict["Gamma_d"] + results_dict["Gamma_l"]) / 2.0
    results_dict["sigma_diam"] = np.NaN
    results_dict["sigma_Q_L"] = np.NaN
    results_dict["sigma_Q_0"] = np.NaN
    if(save_plots):
        fig_name = os.path.join(out_folder, f"smithfit_{out_name}.{out_format}")
        plt.savefig(fig_name, dpi = 100)
    if not hide_plots:
        plt.show(block=True)
    return results_dict

def only_show_S11(freqs, gammas,\
 save_plots=False, out_folder='', out_format='png', out_name = 'output', hide_plots = False, quiet = False, no_smith = False, scalar = False):
    # head, out_name = os.path.split(file) #find file name
    plt.ion()
    # freqs, gammas = get_data_from_file(file, header, footer, every, comments, delimiter)
    fig_name = os.path.join(out_folder, f"smithfit_{out_name}.{out_format}")

    if no_smith:
        fig = plt.figure(figsize=(8, 8))
        thetas = np.linspace(0, 2*np.pi, num = 1000)
        circle_unit = 0+0j  + 1 * np.exp(1j * thetas)
        ax = plt.subplot(1, 1, 1)
        ax.set_aspect('equal', adjustable='box')
        plt.plot(gammas.real, gammas.imag,  marker='o',color='k', linestyle='None',markersize=7, label = 'Data points')
        plt.plot(circle_unit.real, circle_unit.imag,  marker='None',color='k',linewidth=2, linestyle='-', label = 'Unit Circle')
    elif scalar:
        fig = plt.figure(figsize=(12, 8))
        plt.plot(freqs*1E-9, 20*np.log10(np.abs(gammas)),  marker='None',color='black', linestyle='-', label = 'S11')
        plt.xlabel("Frequency [GHz]", fontsize=12)
        plt.ylabel("S-Parameter [dB]", fontsize=12)
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(1, 1, 1, projection='smith')
        ax.set_aspect('equal', adjustable='box')
        plt.plot(gammas, datatype=SmithAxes.S_PARAMETER,  marker='o',color='k', linestyle='None',markersize=7, label = 'Data points')

    if(save_plots): plt.savefig(fig_name, dpi = 100,bbox_inches='tight')
    plt.show(block=True)


def only_show_transmission(freqs, S11, S21, S12, S22,\
 save_plots=False, out_folder='', out_format='png', out_name = 'output', hide_plots = False, quiet = False, scalar = False):
    # head, out_name = os.path.split(file) #find file name
    plt.ion()
    # freqs, gammas = get_data_from_file(file, header, footer, every, comments, delimiter)
    fig_name = os.path.join(out_folder, f"smithfit_{out_name}.{out_format}")

    if scalar:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.subplots()
        ax.plot(freqs*1E-9, 20*np.log10(np.abs(S11)),  marker='None',color='green', linestyle='-', label = 'S11')
        ax.plot(freqs*1E-9, 20*np.log10(np.abs(S12)), marker='None',color='steelblue', linestyle='-', label = 'S12')
        ax.plot(freqs*1E-9, 20*np.log10(np.abs(S22)),  marker='None',color='orange', linestyle='-', label = 'S22')
        ax.legend()
        plt.xlabel("Frequency [GHz]", fontsize=12)
        plt.ylabel("S-Parameter [dB]", fontsize=12)
    else:
        fig = plt.figure(figsize=(8, 8))
        thetas = np.linspace(0, 2*np.pi, num = 1000)
        circle_unit = 0+0j  + 1 * np.exp(1j * thetas)
        ax = fig.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(np.angle(S11), np.abs(S11),  marker='o',color='green', linestyle='None',markersize=5, label = 'S11')
        ax.plot(np.angle(S12), np.abs(S12), marker='o',color='steelblue', linestyle='None',markersize=5, label = 'S12')
        ax.plot(np.angle(S22), np.abs(S22),  marker='o',color='orange', linestyle='None',markersize=5, label = 'S22')
        ax.legend(bbox_to_anchor=(1.08, 1.08))
        ax.set_ylim([0,1])

    if(save_plots): plt.savefig(fig_name, dpi = 100,bbox_inches='tight')

    # ax.plot(circle_unit.real, circle_unit.imag,  marker='None',color='k',linewidth=2, linestyle='-', label = 'Unit Circle')

    plt.show(block=True)
