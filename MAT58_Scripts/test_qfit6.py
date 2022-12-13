"""
    Test script for Q-factor fitting (pure Python version using numpy)
    ==================================================================

      Description
      -----------

         This software is from:
         "Q-factor Measurement by using a Vector Network Analyser",
         A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)

         Fits FL and QL to transmission (S21) data for two-port resonator
         by using the NLQFIT6 algorithm.

         Requires script qfitmod6.py
         Test data is read from file Figure6b.txt (as used in Figure 6(b) of MAT 58).

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
from qfitmod6 import *


def fit_qfit6(F, D, quiet=False, loop_plan = 'fwfwfwc', scaling_factor_A = 'Auto', trmode = 'reflection_method1'):

    # The convergence algorithm uses a number of steps set by loop_plan, a string of characters as follows:
    #   f - fit once without testing for convergence
    #   c - repeated fit, iterating until convergence is obtained
    #   w - re-calculate weighting factors on basis of previous fit
    #       Initially the weighting factors are all unity.
    #   The first character in loop_plan must not be w.
    # loop_plan ='fwfwc'
    if trmode == 'reflection_method1' or trmode == 'reflection_method2':
        measurement_type = 'reflection'
    elif trmode == 'transmission':
        measurement_type = 'transmission'


    # Find peak in |S21| - this is used to give initial value of freq.
    # Tol is 1.0E-5 * |S21| at peak.
    Mg = np.absolute(D)
    if measurement_type == 'reflection':
        index_min_max = np.argmin(Mg)
    else:
        index_min_max = np.argmax(Mg)

    Tol = Mg[index_min_max]*1.0E-5
    Fseed = F[index_min_max]

    # Set Qseed: An order-of-magnitude estimate for Q-factor
    mult = 5.0 # Not critical. A value of around 5.0 will work well for initial and optimised fits (Section 2.6).
    Qseed = mult*Fseed/(F[-1]-F[0])

    if not quiet: print('Initial values for iteration:  Freq=',Fseed,' QL=',Qseed)

    # Step 1: Initial unweighted fit --> solution vector
    N = len(F)
    sv = initialfit(F, D, N, Fseed, Qseed)
    # print('Initial solution (eqn. 17):');
    # print('  a = %6.4f + j*%6.4f' % (sv[0],sv[1]));
    # print('  b = %6.4f + j*%6.4f' % (sv[2],sv[3]));
    # print('  QL = %3.3f' % sv[4]);
    #
    # print('\nOptimising ....')

    # Step 2: Optimised weighted fit --> result vector
    mv, weighting_ratio, number_iterations, RMS_Error = optimisefit6(F, D, N, Fseed, sv, loop_plan, Tol, quiet)
    m1,m2,m3,m4,QL,FL = mv
    Flwst = F[0]
    coeffs = [m1, m2, m3, m4, QL, QL*Flwst/FL]
    # Now calculate unloaded Q-factor and some other useful quantities.
    # Reciprocal of |S21| of a thru in place of resonator
    # scaling_factor_A = 1.0    # 1/|S21_thru|
    if measurement_type == 'reflection':
        # trmode='reflection_method2'
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
        sqrt_eps = 1.3 # Average for and ptfe sections
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
# # Tests
# assert abs(FL-3.987848)<0.000001
# assert abs(Qo-7546)<1.0
# assert abs(QL-7454)<1.0
