#!/usr/bin/env python3
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import brentq
from scipy.optimize import fsolve
import sys
import os
import argparse
from fit_methods import fit_and_display_kajfez
from fit_methods import fit_and_display_qfit
from fit_methods import fit_and_display_scalar
from fit_methods import only_show_S11
from fit_methods import fit_and_display_full_transmission
from fit_methods import only_show_transmission
from utils import *

this_file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(this_file_path, 'MAT58_Scripts'))
# sys.path.insert(0, '/mnt/Mircea/Facultate/Master Thesis/scripts/smithfit/MAT58_Scripts')
from test_qfit6 import fit_qfit6
from test_qfit7 import fit_qfit7
from test_qfit8 import fit_qfit8


if __name__ == "__main__":
    # file = "data/trans_short2.s2p"
    # header = 9
    # footer = 0
    # every = 1
    # comments = '!'
    # delimiter = ' '
    # measurement_type = 'reflection'


    # file = "data/Figure6b.txt"
    # header = 0
    # footer = 0
    # every = 1
    # comments = '%'
    # delimiter = ''
    # measurement_type = 'transmission'
    # scale = 1E9

    loop_plan = 'fwfwfwc'
    n_steps = 3
    plot_every_step = False
    save_plots = False
    out_folder=''
    out_format='png'
    method = 'kajfez'
    header = 0
    footer = 0
    every = 1
    only_show = False
    no_smith = False
    prune = False
    lossless = False
    scaling_factor = 'Auto'
    reflmode = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str, help="File Name")
    parser.add_argument("-method",  type=str, help="Method: kajfez, qfit6, qfit7 or qfit8. Default: kajfez", default=method)

    parser.add_argument("-steps", nargs = '?', type=int, help="Number of steps for the fit (KAJFEZ ONLY). Default: 3", default=n_steps)
    parser.add_argument("--plot-steps", action='store_true', help="Plot every fit iteration (Kajfez only)")
    parser.add_argument("--lossless", action='store_true', help="Consider the system losless and do not calculate the touching circle correction (Kajfez only)")
    parser.add_argument("--prune", action='store_true', help="Remove the data outside the bandwidth (BW=F_L/Q_L) in the last step (Kajfez only)")

    parser.add_argument("-reflmode",  type=int, help="QFIT Only. Valid values: 1 or 2. Mode 1: No touching circle, Mode 2: Use touching circle (scaling can't be auto). Default: 2", default=reflmode)
    parser.add_argument("-plan", type=str, help="Loop plan for the QFIT algorithm", default=loop_plan)
    parser.add_argument("-scaling", type=str, help="Scaling factor for the QFIT algorithm (Auto or float)", default=scaling_factor)

    parser.add_argument("-header", nargs = '?', metavar='N',  type=int, help="Remove the first N data points", default=header)
    parser.add_argument("-footer", nargs = '?', metavar='N',  type=int, help="Remove the last N data points", default=footer)
    parser.add_argument("-every", nargs = '?', metavar='N', type=int, help="Sample every N data points", default=every)
    parser.add_argument("-bandwidth", nargs = '?',type=float, help="Remove data outside of the bandwidth [MHz]", default=0)
    parser.add_argument("-ofolder",  type=str, help="Output folder", default=out_folder)
    parser.add_argument("-oformat",  type=str, help="Output format (.png, .jpg, .svg, .pdf etc.)", default=out_format)

    parser.add_argument("--save-plots", action='store_true', help="If present, the software saves the plots as image files")
    parser.add_argument("--polar", action='store_true', help="Polar plot")
    parser.add_argument("--no-fit", action='store_true', help="Do not fit, only show the data")
    parser.add_argument("--scalar", action='store_true', help="Show the data in scalar form (only if no-fit is active)")

    args = parser.parse_args()
    file = args.fname
    method = args.method
    n_steps = args.steps
    every = args.every
    bandwidth = args.bandwidth
    header = args.header
    footer = args.footer
    out_folder = args.ofolder
    out_format = args.oformat
    plot_every_step=args.plot_steps
    save_plots = args.save_plots
    reflmode = args.reflmode
    no_smith = args.polar
    prune = args.prune
    lossless = args.lossless
    loop_plan = args.plan
    scaling_factor = args.scaling
    no_fit = args.no_fit
    scalar = args.scalar

    extension = os.path.splitext(file)[1]
    # if extension == '.s2p':
    #     type = 'transmission_s2p'

  #  if args.unit == 'GHz':
#		scale = 1E9
 #   elif args.unit == 'MHz':
 #       scale = 1E6
 #   elif args.unit == 'kHz':
 #       scale = 1E3
 #   elif args.unit == 'Hz':
   #     scale = 1
    head, out_name = os.path.split(file) #find file name
    out_name = out_name.replace('.', '_')

    if extension == '.s2p':
        freqs, S11, S21, S12, S22 = get_data_from_s2p(file, header=header, footer = footer, every = every, bandwidth = bandwidth*1E6)

        if no_fit:
            only_show_transmission(freqs, S11, S21, S12, S22,\
                         save_plots=save_plots, out_folder=out_folder, out_format=out_format, out_name = out_name, scalar = scalar)

        else:
            print(f"Fitting TRANSMISSION data using the {method.upper()} mehtod...")
            if method == 'qfit6' or method == 'qfit7' or method == 'qfit8':
                if (scaling_factor.upper()=='AUTO'):
                    print('Auto scaling works only for reflection_method1! Changing scaling factor to 1.0')
                    scaling_factor = 1.0
                else:
                    try:
                      scaling_factor = float(scaling_factor)
                    except:
                      print("Invalid scaling factor! Accepted values are auto or a float value")
                      sys.exit(-1)
            fit_and_display_full_transmission(freqs, S11, S21, S12, S22,  n_steps = n_steps, method = method, loop_plan = loop_plan, scaling_factor = scaling_factor, prune_last_step=prune,\
            save_plots=save_plots, out_folder='', out_name = out_name, out_format=out_format, hide_plots = False, no_smith = False, quiet = False)
    else:
        freqs, gammas = get_data_from_s1p(file, header=header, footer=footer, every = every, bandwidth = bandwidth*1E6)
        if no_fit:
            only_show_S11(freqs, gammas,\
             save_plots=save_plots, out_folder=out_folder, out_format=out_format, out_name = out_name, no_smith = no_smith, scalar = scalar)
        else:
            print(f"Fitting REFLECTION data using the {method.upper()} mehtod...")
            if method == 'kajfez':
                fit_and_display_kajfez(freqs, gammas, out_name = out_name, measurement_type = 'reflection', prune_last_step=prune, lossless = lossless, \
                n_steps=n_steps, no_smith = no_smith, plot_every_step=plot_every_step, save_plots = save_plots,\
                 out_folder=out_folder, out_format=out_format)
            elif method == 'qfit6' or 'qfit7' or 'qfit8':
                 if reflmode == 1: trmode = 'reflection_method1'
                 elif reflmode == 2: trmode = 'reflection_method2'
                 else:
                     print("Invalid reflmode!")
                     sys.exit(-1)
                 if (scaling_factor.upper()=='AUTO'):
                     if(reflmode == 2):
                         print('Auto scaling works only for reflection_method1! Changing scaling factor to 1.0')
                         scaling_factor = 1.0
                 else:
                     try:
                       scaling_factor = float(scaling_factor)
                     except:
                       print("Invalid scaling factor! Accepted values are auto or a float value")
                       sys.exit(-1)

                 fit_and_display_qfit(freqs, gammas, method, loop_plan = loop_plan, measurement_type = trmode, scaling_factor = scaling_factor, out_name = out_name,\
                  out_folder=out_folder, no_smith = no_smith, out_format=out_format)
            elif method == 'scalar':
                fit_and_display_scalar(freqs, gammas, scale = scale, n_steps = n_steps,\
                    plot_every_step=False, save_plots=save_plots, out_folder=out_folder, out_name = out_name, out_format=out_format)
            else:
                print("Invalid method! The only valid methods are kajfez, qfit6, qfit7 and qfit8")
                sys.exit(-1)
