"""
    Test script for Q-factor fitting (pure Python version using numpy)
    ==================================================================

      Description
      -----------

         This software is from:
         "Q-factor Measurement by using a Vector Network Analyser",
         A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)

         Fits to transmission (S21) data for two-port resonator
         (with frequency-dependent leakage) by using the NLQFIT8 algorithm.

         Requires script qfitmod8.py
         Test data is read from file Figure23.txt (shown in Figure 23 of MAT 58)

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
from qfitmod8 import *
import cmath, math

def get_data_from_file(file, header, footer, every, comments, delimiter):
    data = np.genfromtxt(open(file,'rt').readlines()[::every], comments = comments, delimiter = delimiter, skip_header = header, skip_footer = footer)
    gammas = data[:, 1] + data[:, 2] * 1j
    freqs = data[:, 0]
    return freqs, gammas

def fit_qfit8(F, D, quiet = False, loop_plan = 'fwfwfwc', scaling_factor_A = 'Auto', trmode = 'reflection_method1'):
    N = len(F)
    # ncablelen = 1.2 # root_eps * cable length in metres
    # D = [D[i]*cmath.exp(complex(0.0,-2E9*math.pi*F[i]*ncablelen/2.99792458E8)) for i in range(N)]


    # The convergence algorithm uses a number of steps set by loop_plan, a string of characters as follows:
    #   f - fit once without testing for convergence
    #   c - repeated fit, iterating until convergence is obtained
    #   w - re-calculate weighting factors on basis of previous fit
    #       Initially the weighting factors are all unity.
    #   The first character in loop_plan must not be w.
    # loop_plan ='fwfwc'

    # Find peak in |S11| - this is used to give initial value of freq.
    # Tol is 1.0E-5 * |S21| at peak.

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
    mult = 2.0 # Not critical. A value of around 5.0 will work well for initial and optimised fits (Section 2.6).
    Qseed = mult*Fseed/(F[-1]-F[0])

    # Set Qseed: An order-of-magnitude estimate for Q-factor
    mult = 5.0 # Not critical. A value of around 5.0 will work well for initial and optimised fits (Section 2.6).
    Qseed = mult*Fseed/(F[-1]-F[0])
    if not quiet: print('Initial values for iteration:  Freq=',Fseed,' QL=',Qseed)

    # Step 1: Initial unweighted fit --> solution vector
    sv = initialfit(F, D, N, Fseed, Qseed)
    a1 = sv[0,0]+1j*sv[1,0];
    a2 = sv[2,0]+1j*sv[3,0];
    a3 = sv[4,0] * 1j

    if not quiet:
        print('Initial values for iteration:  Freq=',Fseed,' QL=',Qseed)
        print('Initial solution found');
        print(f' a = {a1:.6f}');
        print(f' b = {a2:.6f}');
        # print(f' a = {sv[0]+1j.*sv[1]:.3f}');
        print(f' QL = {np.imag(a3):.6f}');
        print('\nOptimising ....')

    # Step 2: Optimised weighted fit --> result vector
    mv, weighting_ratio, number_iterations, RMS_Error = optimisefit8(F, D, N, Fseed, sv, loop_plan, Tol, quiet)
    m1,m2,m3,m4,m8,m9,QL,FL = mv
    Flwst = F[0]
    coeffs = [m1, m2, m3, m4, QL, QL*Flwst/FL, m8, m9]

    if weighting_ratio is not None and not quiet: print('  Weighting ratio = %5.3f' % weighting_ratio)
    # print('  RMS_Error = %10.8f' % RMS_Error)
    # print()

    # Now calculate unloaded Q-factor and some other useful quantities.
    # Reciprocal of |S21| of a thru in place of resonator

    # scaling_factor_A = 'Auto'   # 1/|S21_thru|
    if measurement_type == 'reflection':
        # trmode='reflection_method1'
        ifail, p = GetUnloadedData(mv, scaling_factor_A, trmode, quiet)
        Qo, k, cal_diam, cal_touching_circle_diam, cal_gamma_V, cal_gamma_T = p #Gamma_V = Gamma_d from kajfez and Gamma_T = Gamma_L
    elif measurement_type == 'transmission':
        # trmode = 'transmission'
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
    return results_dict


# fn = '/mnt/Mircea/Facultate/Master Thesis/data/old_data/loop_antenna_different_lenghts/0.66cm/loop_antenna_2.s1p';
# F, D = get_data_from_file(fn, 13, 0, 1, '!', '')
# fit_qfit8(F,D)
