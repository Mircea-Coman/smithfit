import numpy as np
from matplotlib import pyplot as plt
from utils import *
from fit_methods import fit_and_display_kajfez
from fit_methods import fit_and_display_qfit
import os

this_file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(this_file_path, 'MAT58_Scripts'))
from test_qfit6 import fit_qfit6
from test_qfit7 import fit_qfit7
from test_qfit8 import fit_qfit8

sys.path.append("smithplot")
from smithplot import SmithAxes


filename = "data/trans_short.s2p"
n_steps = 10

freqs, S11, S21, S12, S22 = get_data_from_s2p(filename, 9, 0, 1, '!', ' ')
# S11_dict = fit_and_display_kajfez(freqs, S11, measurement_type = 'reflection', n_steps=n_steps, quiet = True, hide_plots = True)
# S12_dict = fit_and_display_kajfez(freqs, S12, measurement_type = 'transmission', n_steps=n_steps, quiet = True, hide_plots = True)
# S22_dict = fit_and_display_kajfez(freqs, S22, measurement_type = 'reflection', n_steps=n_steps, quiet = True, hide_plots = True)

S11_dict = fit_and_display_qfit(freqs, S11, fit_qfit8, measurement_type = 'reflection', quiet = False, hide_plots = True)
S12_dict = fit_and_display_qfit(freqs, S12, fit_qfit8, measurement_type = 'transmission', quiet = False, hide_plots = True)
S22_dict = fit_and_display_qfit(freqs, S22, fit_qfit8, measurement_type = 'reflection', quiet = False, hide_plots = True)

thetas = np.linspace(0, 2*np.pi, num = 1000)
circle_S11 = S11_dict["center"] + 0.5*S11_dict["diam"] * np.exp(1j * thetas)
circle_S12 = S12_dict["center"] + 0.5*S12_dict["diam"] * np.exp(1j * thetas)
circle_S22 = S22_dict["center"] + 0.5*S22_dict["diam"] * np.exp(1j * thetas)

f_L = S12_dict["f_L"]
den = (2.0 - S11_dict["diam"] - S22_dict["diam"])
sigma_den = np.sqrt(S11_dict["sigma_diam"]**2 + S22_dict["sigma_diam"]**2 )
beta1 = S11_dict["diam"] / den
beta2 = S22_dict["diam"] / den
Q_L = S12_dict["Q_L"]
Q_0 = 2 * S12_dict["Q_L"] / den

sigma_Q_L = S12_dict["sigma_Q_L"]
sigma_beta1 = beta1 * np.sqrt((S11_dict["sigma_diam"] / S11_dict["diam"])**2 + (sigma_den / den)**2)
sigma_beta2 = beta2 * np.sqrt((S22_dict["sigma_diam"] / S22_dict["diam"])**2 + (sigma_den / den)**2)
sigma_Q_0 = Q_0 * np.sqrt((S12_dict["sigma_Q_L"] / S12_dict["Q_L"])**2 + (sigma_den / den)**2)

print(f" Q_L = {Q_L:.6f} +/- {sigma_Q_L:.6f}")
print(f" beta_1 = {beta1:.6f} +/- {sigma_beta1:.6f}")
print(f" beta_2 = {beta2:.6f} +/- {sigma_beta2:.6f}")
print(f" Q_0 = {Q_0:.6f} +/- {sigma_Q_0:.6f}")
print(f" f_L = {f_L *1E-9:.8f} GHz")
print()

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(1, 1, 1, projection='smith')
ax.set_aspect('equal', adjustable='box')
plt.plot(S11, datatype=SmithAxes.S_PARAMETER,  marker='o',color='green', linestyle='None',markersize=5, label = 'Data points')
plt.plot(S12, datatype=SmithAxes.S_PARAMETER,  marker='o',color='darkcyan', linestyle='None',markersize=5, label = 'Data points')
plt.plot(S22, datatype=SmithAxes.S_PARAMETER,  marker='o',color='orange', linestyle='None',markersize=5, label = 'Data points')
plt.plot(circle_S11, datatype=SmithAxes.S_PARAMETER,  marker='None',color='red', linestyle='-', linewidth=2, label = 'Data points')
plt.plot(circle_S12, datatype=SmithAxes.S_PARAMETER,  marker='None',color='red', linestyle='-', linewidth=2, label = 'Data points')
plt.plot(circle_S22, datatype=SmithAxes.S_PARAMETER,  marker='None',color='red', linestyle='-', linewidth=2, label = 'Data points')
plt.show(block=True)
