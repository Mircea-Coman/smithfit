"""
    qfitmod7.py
    ===========

      Description
      -----------

        This software is from:
        "Q-factor Measurement by using a Vector Network Analyser",
        A. P. Gregory, National Physical Laboratory Report MAT 58 (2021)

        Python implementation of the NLQFIT7 algorithm. The input parameter
        trmode specifies the resonance type (normally "reflection_method1"
        or "reflection_method2" - see MAT 58 Section 2.5.1).

        Uncalibrated line should be de-embedded (if it has significant
        length) from the S-parameter data before calling the functions
        in this module. The remaining length (phase) is fitted by optimisefit7.

        The user must supply string of characters 'loop_plan',
        which defines order of steps used by the fitting process.


      Creative Commons CC0 Public Domain Dedication
      ---------------------------------------------

         This software is released under the Creative Commons CC0
         Public Domain Dedication. You should have received a copy of
         the Legal Code along with this software.
         If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.


      System
      ------

        Compatible with Python 2.7 and 3.X
        Requires numpy 1.4 or later.


     Change log
     ----------

        :30-October-2021: Released under CC0.

"""

import numpy as np
import math, cmath


def angularwts(F, Fres, QL):
    """
        Calculate diagonal elements of weights matrix.
        ----------------------------------------------

          The weights are needed when F consists of equally-spaced
          frequency points (rather than points equally spaced around
          the Q-circle). MAT 58 eqn. (28).
    """

    F2 = np.array(F)
    ptmp = 2.0*QL*(F2-Fres)/Fres
    PV = 1.0/(ptmp**2 + 1.0);
    return PV


def initialfit(F, S, N, Fres, Qseed):
    """
        Initial linear fit, step (1)
        ----------------------------

          Input parameters:
            F     - List of frequencies
            S     - List of complex data S-parameter to be fitted.
            N     - Number of points
            Fres  - Estimated resonant frequency (not fitted)
            Qseed - Estimated QL (will be improved by fitting).
                    It is usually sufficient to supply QL = 1000.0

          Output parameters (MAT58 eqn. 17):
             sv = [a', a'', b', b'', QL]
    """

    N2 = 2*N;
    M = np.zeros([N2, 5])
    G = np.zeros(N2)[:, np.newaxis]

    for i in range(N):
        i2 = i+N
        t = 2.0*(F[i]/Fres - 1.0)
        y = 1.0/complex(1.0, Qseed*t)
        v = t*y
        v1 = y*S[i]
        G[i] = v1.real;    G[i2] = v1.imag
        v2 = v1*t
        M[i,:] = v.real, -v.imag, y.real, -y.imag, v2.imag
        M[i2,:] = v.imag, v.real, y.imag, y.real, -v2.real

    #X = M.transpose()
    #P = [1.0]*N2 # Weights if required
    #T = np.multiply(X,P)
    T = M.transpose() # unweighted
    C = np.dot(T,M)
    q = np.dot(T,G)
    sv = np.linalg.solve(C,q)
    return sv


def optimisefit7(F, S, N, Fseed, sv, loop_plan, Tol, quiet):
    """
       Iterative non-linear fit, NLQFIT7 Step (2)
       ------------------------------------------

         Optimised fit of Q-factor (QL) and resonant frequency (FL)
         by the gradient-descent method.

         Uses the results of the initial fit (sv) as the starting
         values for the iteration.

         Input parameters:
            F         - List of frequencies.
            S         - List of complex data S-parameter to be fitted.
            N         - Number of points.
            Fseed     - Estimated resonant frequency.
            sv        - Initial solution (numpy vector or a list) found with InitialFit
            loop_plan - String of characters which defines order of
                        steps used by the fitting process, e.g. 'fwfwc':
                         f - fit once without testing for convergence.
                         c - repeated fit, iterating until convergence is obtained.
                         w - re-calculate weighting factors on basis of previous fit.
            Tol       - Criterion for the convergence test.
                        Recommend using 1.0E-5 for reflection
                        or max(abs(Gamma))*1.0E-5 for transmission.
            quiet     - Boolean flag controlling output of information to
                        the console.

         Output (MAT58 eqn. 26)
               list of fitted parameters: [m1, m2, m4, m4, QL, FL, m7/Flwst]

    """

    if loop_plan[-1] == 'w': assert 0, 'Last item in loop_plan must not be w (weight calculation)'
    if loop_plan[0]  == 'w': assert 0, 'First item in loop_plan must not be w (weight calculation)'
    if loop_plan[-1] != 'c': print('Warning: Last item in loop_plan is not c so convergence not tested!')

    N2 = N*2;
    iterations = 0
    PV = np.ones(N)     # default weights vector
    PV2 = np.ones(N2)

    m1 = sv[1]/sv[4]  # a''/QL
    m2 = -sv[0]/sv[4]
    m3 = sv[2]-m1
    m4 = sv[3]-m2
    m5 = sv[4]
    Flwst = F[0]   # lowest freq. is a convenient normalisation factor.
    m6 = Flwst*m5/Fseed
    m7 = 0.0
    last_op = 'n'
    del sv
    weighting_ratio = None
    number_iterations = 0

    ## Loop through all of the operations specified in loop_plan
    for op in loop_plan:

        if op == 'w':          # Fr                       QL
            PV = angularwts(F, Flwst*float(m5)/float(m6), float(m5))
            weighting_ratio = max(PV)/min(PV)
            PV2 = np.concatenate((PV, PV))
            if not quiet: print('Op w, Calculate weights')
            last_op = 'n'
            continue

        if op == 'c': seek_convergence = True
        elif op == 'f': seek_convergence = False
        else: assert 0,'Unexpected character in loop_plan'

        TerminationConditionMet = False
        RMS_Error = None
        while not (TerminationConditionMet):
            number_iterations += 1
            M = np.zeros([N2,7])
            G = np.zeros(N2)[:, np.newaxis]
            c1 = complex(-m4,m3); c2 = complex(m1,m2); c3 = complex(m3,m4)
            for i in range(N):
                i2 = i+N
                y = 1.0/complex(1.0,2*(m6*F[i]/Flwst-m5))
                fdn = F[i]/Flwst - m5/m6
                pj = complex(0.0, m7*fdn)
                expm7 = cmath.exp(pj)
                ym = y*expm7
                u = c1*y*ym*2
                u2 = -u*F[i]/Flwst
                v = (c2 + y*c3)* expm7;
                u3 = v*fdn;
                M[i,:] = expm7.real, -expm7.imag, ym.real, -ym.imag, u.real, u2.real, -u3.imag
                M[i2,:] = expm7.imag, expm7.real, ym.imag, ym.real, u.imag, u2.imag, u3.real
                r = S[i] - v  # residual
                G[i] = r.real; G[i2] = r.imag

            X = M.transpose()
            T = np.multiply(X, PV2)
            C = np.dot(T,M)
            q = np.dot(T,G)
            dm = np.linalg.solve(C,q)
            m1 += dm[0]; m2 += dm[1]; m3 += dm[2]
            m4 += dm[3]; m5 += dm[4]; m6 += dm[5]
            m7 += dm[6]
            del G, X, T, C, dm
            iterations = iterations + 1
            if RMS_Error is not None: Last_RMS_Error = RMS_Error
            else: Last_RMS_Error = None

            SumNum = 0.0
            SumDen = 0.0
            for i in range(N):
                fdn = F[i]/Flwst - m5/m6;
                den = complex(1.0, 2*(m6*F[i]/Flwst-m5))
                pj = complex(0.0,m7*fdn)
                E = S[i] - (c2 + c3/den)*cmath.exp(pj)
                ip = PV[i]
                SumNum = SumNum + ip*(E.real*E.real + E.imag*E.imag)
                SumDen = SumDen + ip

            RMS_Error = math.sqrt(SumNum/SumDen);
            if not quiet:
                if last_op =='c':
                    print('      Iteration %i, RMS_Error %10.8f' % (iterations, RMS_Error))
                else:
                    print('Op %c, Iteration %i, RMS_Error %10.8f' % (op, iterations, RMS_Error))
            last_op = op

            if seek_convergence:
                if Last_RMS_Error is not None:
                    delta_S = abs(RMS_Error-Last_RMS_Error)
                    TerminationConditionMet = delta_S < Tol
            else:
                TerminationConditionMet = True

        # After last operation, we end up here ...
        if not quiet: print()
    return [m1,m2,m3,m4,m5,m5*Flwst/m6, m7/Flwst], weighting_ratio, number_iterations, RMS_Error
    # return [m1,m2,m3,m4 ,m5, m6, m7], weighting_ratio, number_iterations, RMS_Error


def QCircleData(A, m1, m2, m3, m4):
    """
      Use MAT 58 eqn. (31) to calculate calibrated diameter
      -----------------------------------------------------

      Also return calibrated gamma_V and gamma_T
    """
    aqratio = complex(m1,m2)
    b = complex(m1+m3, m2+m4)
    caldiam = abs(b - aqratio)*A
    calgamma_V =  complex(m1,m2)*A
    calgamma_T = b*A
    return caldiam, calgamma_V, calgamma_T


def GetUnloadedData(mv, scaling_factor_A, trmode, quiet):

    """
       Calculate unloaded Q-factor and various 'calibrated' quantities
       ---------------------------------------------------------------

       Input parameters:
         mv               - solution produced by OptimiseFit
         trmode           - 'transmission', 'reflection_method1',
                            'reflection_method2' or 'absorption'
         scaling_factor_A - scaling factor as defined in MAT 58.
                            For reflection_method1, can specify as 'AUTO'
                            to use the magnitude of the fitted detuned reflection coefficient (gammaV)
    """

    if type(scaling_factor_A) is str:
        if (scaling_factor_A.upper()=='AUTO'): auto_flag = True
        else:
            return 1, 'Illegal Scaling factor'
    else:
        # Also permit negative scaling factor to indicate 'Auto'
        if scaling_factor_A<0.0: auto_flag = True
        else: auto_flag = False

    m1,m2,m3,m4,m5,FL,m7_flwst = mv
    if trmode=='transmission':

        if auto_flag:
            return 1, 'Scaling factor must not be "Auto" for transmission case'
        cal_diam, cal_gamma_V, cal_gamma_T = QCircleData(scaling_factor_A, m1, m2, m3, m4)
        if cal_diam == 1.0:
            return 1, 'Divide by zero forestalled in calculation of Qo'
        Q0 = m5/(1.0-cal_diam)
        return 0, (Q0, cal_diam, cal_gamma_V, cal_gamma_T)

    elif trmode=='reflection_method1':
        if auto_flag:
            if not quiet: print('Supplied scaling_factor_A is "Auto", so using fitted data to estimate it')
            scaling_factor_A = 1.0/abs(complex(m1,m2)) # scale to gammaV if 'AUTO'
        cal_diam, cal_gamma_V, cal_gamma_T = QCircleData(scaling_factor_A, m1, m2, m3, m4)
        cal_touching_circle_diam = 2.0
        if not quiet: print('  Q-circle diam = %5.3f, touching_circle_diam = %5.3f' % (cal_diam, cal_touching_circle_diam))
        den = cal_touching_circle_diam/cal_diam - 1.0
        Q0 = m5*(1.0 + 1.0/den)
        return 0, (Q0, 1.0/den, cal_diam, cal_touching_circle_diam, cal_gamma_V, cal_gamma_T)


    elif trmode=='reflection_method2':
        if auto_flag: return 1, 'Scaling factor must not be "Auto" for Method 2'
        cal_diam, cal_gamma_V, cal_gamma_T = QCircleData(scaling_factor_A, m1, m2, m3, m4)
        gv = abs(cal_gamma_V)
        gv2 = gv*gv
        mb = abs(cal_gamma_T)
        cosphi = (gv2 + cal_diam*cal_diam - mb*mb)/(2.0*gv*cal_diam) # Cosine rule
        cal_touching_circle_diam = ((1.0-gv2)/(1.0-gv*cosphi))
        if not quiet: print('  Q-circle diam = %5.3f, touching_circle_diam = %5.3f' % (cal_diam, cal_touching_circle_diam))
        den = cal_touching_circle_diam/cal_diam - 1.0
        Q0 = m5*(1.0 + 1.0/den)
        return 0, (Q0, 1.0/den, cal_diam, cal_touching_circle_diam, cal_gamma_V, cal_gamma_T)


    elif trmode=='notch' or trmode=='absorption': # Absorption resonator measured by transmission
        if auto_flag:
            if not quiet: print('Notch/absorption Qo calculation: Supplied scaling_factor_A is "Auto", so using fitted data to estimate it')
            scaling_factor_A = 1.0/abs(complex(m1,m2)) # scale to gammaV if 'AUTO'
        cal_diam, cal_gamma_V, cal_gamma_T = QCircleData(scaling_factor_A, m1, m2, m3, m4)
        if not quiet: print('  Q-circle diam = %5.3f' % (cal_diam))
        if cal_diam == 1.0:
            return 1, 'Divide by zero forestalled in calculation of Qo'
        den = 1.0/cal_diam - 1.0 # Gao thesis (2008) 4.35 and 4.40
        Q0 = m5*(1.0 + 1.0/den)  # https://resolver.caltech.edu/CaltechETD:etd-06092008-235549
        # For this type of resonator, critical coupling occurs for cal_diam = 0.5.

    else:
        return 1, 'Unknown trmode'
