"""
    Test script for Q-factor fitting (pure Python version using numpy)
    ==================================================================

      Description
      -----------

         This software is from:
         "Q-factor Measurement by using a Vector Network Analyser",
         A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)

         Fits FL and QL to reflection (S11) data by using the NLQFIT7 algorithm.

         Requires script qfitmod7.py
         Test data is read from file Table6c27.txt (as used in Figure 16 and
         Table 6(c) of MAT 58).

         Tested with Python 2.7 and Python 3.8


      Creative Commons CC0 Public Domain Dedication
      ---------------------------------------------

         This software is released under the Creative Commons CC0
         Public Domain Dedication. You should have received a copy of
         the Legal Code along with this software.
         If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.


     Change log
     ----------

        :30-October-2021: Released under CC0.

"""
import numpy as np
from qfitmod7 import *
from math import pi


# fn = '/mnt/Mircea/Facultate/Master Thesis/data/old_data/loop_antenna_different_lenghts/0.66cm/loop_antenna_2.s1p';
def fit_qfit7(F, D, quiet=False, loop_plan = 'fwfwfwc', scaling_factor_A = 'Auto', trmode = 'reflection_method1'):
    # The convergence algorithm uses a number of steps set by loop_plan, a string of characters as follows:
    #   f - fit once without testing for convergence
    #   c - repeated fit, iterating until convergence is obtained
    #   w - re-calculate weighting factors on basis of previous fit
    #       Initially the weighting factors are all unity.
    #   The first character in loop_plan must not be w.
    # loop_plan ='fwfwfwc'
    if trmode == 'reflection_method1' or trmode == 'reflection_method2':
        measurement_type = 'reflection'
    elif trmode == 'transmission':
        measurement_type = 'transmission'

    Mg = np.absolute(D)
    if measurement_type == 'reflection':
        index_min_max = np.argmin(Mg)
    else:
        index_min_max = np.argmax(Mg)

    Tol = Mg[index_min_max]*1.0E-5
    Fseed = F[index_min_max]

    # Set Qseed: An order-of-magnitude estimate for Q-factor
    # mult = 5.0 # Not critical. A value of around 5.0 will work well for initial and optimised fits (Section 2.6).
    # Qseed = mult*Fseed/(F[-1]-F[0])

    # Set Qseed: An order-of-magnitude estimate for Q-factor
    mult = 5.0 # Not critical. A value of around 5.0 will work well for initial and optimised fits (Section 2.6).
    Qseed = mult*Fseed/(F[-1]-F[0])
    # Step 1: Initial unweighted fit --> solution vector
    N = len(F)
    sv = initialfit(F, D, N, Fseed, Qseed)
    # Initial solution (eqn. 17):
    #a = sv[0]+ j*sv[1]
    #b = sv[2] + j*sv[3]
    #QL = sv[4];
    # Step 2: Optimised weighted fit --> result vector
    mv, weighting_ratio, number_iterations, RMS_Error = optimisefit7(F, D, N, Fseed, sv, loop_plan, Tol, quiet)
    m1,m2,m3,m4,QL,FL, m7a = mv
    Flwst = F[0]
    coeffs = [m1, m2, m3, m4, QL, QL*Flwst/FL, m7a * Flwst]


    # Now calculate unloaded Q-factor and some other useful quantities.
    # print('Q-factor of uncoupled one-port resonator (unloaded Q-factor) by Method 1:')
    # print('Assumes attenuating uncalibrated line')
    # scaling_factor_A = 1.0
    # scaling_factor_A = 'Auto'

    if measurement_type == 'reflection' :
        ifail, p = GetUnloadedData(mv, scaling_factor_A, trmode, quiet)
        Qo, k, cal_diam, cal_touching_circle_diam, cal_gamma_V, cal_gamma_T = p #Gamma_V = Gamma_d from kajfez and Gamma_T = Gamma_L
    elif measurement_type == 'transmission':
        ifail, p = GetUnloadedData(mv, scaling_factor_A, trmode, quiet)
        Qo, cal_diam, cal_gamma_V, cal_gamma_T = p #Gamma_V = Gamma_d from kajfez and Gamma_T = Gamma_L


    if not quiet:
        print('\nOptimised solution:')
        print(' Q_L = %6.2f ' % QL)
        if measurement_type == 'reflection': print(' k = %6.5f ' % k)
        print(' Q_0 = %6.2f' % Qo)
        print(f' f_L = {FL[0]*1E-9} GHz')
        print(' Number of iterations = %i' % number_iterations)
        if weighting_ratio is not None: print(' Weighting ratio = %5.3f' % weighting_ratio)
        print(' RMS_Error = %10.8f' % RMS_Error)
        sqrt_eps = 1.3 # Average for and ptfe sections
        print(' Fitted length of uncalibrated line = ',-float(m7a)*2.99792458e2/(4.0*pi*1.3),'mm')
        print()

    if measurement_type == 'reflection':
        results_dict = {
            "coeffs": coeffs,
            "f_L": FL[0],
            "Q_L": QL[0],
            "Q_0": Qo[0],
            "k":   k,
            "diam": cal_diam,
            "touch_diam": cal_touching_circle_diam,
            "Gamma_d": cal_gamma_V,
            "Gamma_l": cal_gamma_T,
        }
    elif measurement_type == 'transmission':
        results_dict = {
            "coeffs": coeffs,
            "f_L": FL[0],
            "Q_L": QL[0],
            "Q_0": Qo[0],
            # "k":   k,
            "diam": cal_diam,
            # "touch_diam": cal_touching_circle_diam,
            "Gamma_d": cal_gamma_V,
            "Gamma_l": cal_gamma_T,
        }

    print()

    return results_dict


    # print('  Qcircle diameter = %6.4f' % cal_diam)
    # print('  Gamma_D = %10.8f +j* %10.8f (S11 detuned)' % (cal_gamma_V.real, cal_gamma_V.imag))
    # print('  |Gamma_D| = %10.8f' % abs(cal_gamma_V))
    # print('  Gamma_L = %10.8f +j* %10.8f (S11 tuned)' % (cal_gamma_T.real, cal_gamma_T.imag))


    # print('Q-factor of uncoupled one-port resonator (unloaded Q-factor) by Method 2:')
    # print ('Scaling factor A = 1.0 (assume no attenuation in uncalibrated line)')
    # scaling_factor_A = 1.0
    # trmode='reflection_method2'
    # ifail, p = GetUnloadedData(mv, scaling_factor_A, trmode, quiet)
    # Qo, cal_diam, cal_gamma_V, cal_gamma_T = p
    # print('  Qo = %6.2f  (expect 862)' % Qo)
    # print('  Qcircle diameter = %6.4f' % cal_diam)
    # print('  GammaV = %10.8f +j* %10.8f (S11 detuned)' % (cal_gamma_V.real, cal_gamma_V.imag))
    # print('  |GammaV| = %10.8f' % abs(cal_gamma_V))
    # print('  GammaT = %10.8f +j* %10.8f (S11 tuned)' % (cal_gamma_T.real, cal_gamma_T.imag))

# fn = '/mnt/Mircea/Facultate/Master Thesis/data/cookiebox_antenna_length/AutoSave8.s1p'
# F, D = get_data_from_file(fn, 13, 0, 1, '!', '')
# fit_qfit7(F, D)
