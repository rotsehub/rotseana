# originally created by [] via IDL
# ported to Python by Jacob Juvan
# last modified by Jacob Juvan: August 5, 2025

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
from scipy.interpolate import splrep, splev #legacy versions of cubic spline fitters, splrep is the only one that allows true periodic wrapping
from scipy.optimize import curve_fit
from tqdm import tqdm


def read_lightcurve(file_path):
    #read the dat file and return arrays for time, magnitude, and error
    time, mag, error = [], [], []
    with open(file_path) as f:
        for _ in range(3):
            f.readline()
        for line in f:
            try:
                t, m, dm = map(float, line.strip().split())
                if dm > 0:
                    time.append(t)
                    mag.append(m)
                    error.append(dm)
            except:
                continue
    return np.array(time), np.array(mag), np.array(error)


def safe_knots(x, n_intervals, k):
    #determine spline knot locations with n_intervals knots and degree k
    epsilon = 1e-12
    xmin, xmax = np.min(x), np.max(x)
    while n_intervals >= 1:
        if len(x) < (n_intervals + k):
            n_intervals -= 1
            continue
        percentiles = np.linspace(0, 100, n_intervals + 2)[1:-1]
        knots = np.percentile(x, percentiles)
        knots = knots[(knots > xmin + epsilon) & (knots < xmax - epsilon)]
        unique_knots = np.unique(knots)
        if len(unique_knots) >= k:
            return unique_knots
        else:
            n_intervals -= 1
    return None


def phase_fold(x, per, phi):
    #objective function for folding observational data
    return ((x - x.min()) / per + phi) % 1


def objective_parab(x, a, b, c):
    #parabola objective function for determining period uncertainty
    return a * (x - b)**2 + c


def fit_minimum(pers, chis, a, b, c):
    #find parabolic fit parameters
    p0 = [None] * 3
    p0[0] = a # width
    p0[1] = b # per shift
    p0[2] = c # chi shift
    params, pcov = curve_fit(objective_parab, pers, chis, p0)
    return params


def best_phase_chi2(x, y, dy, per, n_intervals, k=3, n_phase_bins=4):
    #determine the best phase_shift for a given period
    best_chi2 = np.inf
    best_phi = 0
    best_model = None
    best_ndof = None

    for i in range(n_phase_bins):
        phi = 0 * i / (n_phase_bins * n_intervals)
        phase = phase_fold(x, per, phi)
        idx = np.argsort(phase)
        x_folded, y_folded, dy_folded = phase[idx], y[idx], dy[idx]

        try:
            knots = safe_knots(x_folded, n_intervals, k)
            x_folded = np.append(x_folded, x_folded[0] + 1)
            y_folded = np.append(y_folded, y_folded[0])
            dy_folded = np.append(dy_folded, dy_folded[0])
            if knots is None:
                continue
            tck = splrep(x_folded, y_folded, w=1/dy_folded, k=k, t=knots, per=1) #per=1 means the spline will wrap (first and last points are continuous)
            y_fit = splev(x_folded, tck) #evaluate the spline at the given x coordinates
            n_params = len(knots) + k + 1
    
            residuals = (y_folded - y_fit) / dy_folded
            chi2 = np.sum(residuals**2)          
            ndof = len(x_folded) - n_params
    
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_phi = phi
                best_model = tck
                best_ndof = ndof
        except:
            continue

    return best_phi, best_chi2, best_model, best_ndof


def single_phase(file_name, max_per=1.0, min_per=0.0, n_intervals=8, save_plot=True, force_per=None, refine_per=None, dp=0.0001):
    #main function
    time, mag, error = read_lightcurve(file_name)

    if force_per is not None:
        #force a specific period number. Uncertainty will not be calculated
        results = []
        phi, chi2, tck, ndof = best_phase_chi2(time, mag, error, force_per, n_intervals)
        best_per = force_per
        best_phi = phi
        best_chi2 = chi2
        best_tck = tck
        best_ndof = ndof
        best_reduced_chi2 = best_chi2 / best_ndof if best_ndof > 0 else np.nan
        per_error = np.nan
        results.append((force_per, chi2, phi, tck, ndof))

    else:
        #run only if period is NOT forced
        if refine_per is not None:
            #refine a user-input period
            start_per = refine_per - 0.01
            stop_per = refine_per + 0.01
        else:
            start_per = min_per
            stop_per = max_per
    
        p_min = start_per
        p_max = stop_per
        d_per = 0.0001
    
        while d_per >= 0.000001:
            #try all periods between p_min and p_max inclusive with stepsize d_per
            trial_pers = np.arange(p_min, p_max + 0.000001, d_per)
            results = []
            for per in tqdm(trial_pers, ascii=True):
                if per != 0:
                    phi, chi2, tck, ndof = best_phase_chi2(time, mag, error, per, n_intervals)
                    if tck is not None:
                        results.append((per, chi2, phi, tck, ndof))
            results.sort(key=lambda t: t[1])
            top_per = results[0][0]
            results = np.array(results, dtype=object)
            p_min = top_per - d_per * 10
            p_max = top_per + d_per * 10
            d_per /= 10
        
        #display the best 15 periods
        top_results = results[:15]
        best_per, best_chi2, best_phi, best_tck, best_ndof = top_results[0]
        best_reduced_chi2 = best_chi2 / best_ndof if best_ndof > 0 else np.nan
    
        print("\nTop 15 Periods:")
        print(" #  har      chi^2       per          p1            p2            p3            p1-p3")
        #NOTE: Harmonic calculations are incorrect. These will need to be calculated properly.
        #Also, we would like to display the nearest periods around the best found rather than only the 15-best periods.
        for i, (per, chi2, phi, tck, ndof) in enumerate(top_results):
            phase = phase_fold(time, per, phi)
            folded = phase[np.argsort(phase)]
            power = splev(folded, tck) #evaluate each spline
            p1 = np.max(power)
            p2 = np.median(power)
            p3 = np.min(power)
            print(f"{i+1:2}  {2:3}  {chi2/ndof:12.6f}  {per:12.8f}  {p1:12.8f}  {p2:12.8f}  {p3:12.8f}  {p1-p3:12.8f}")
    
        fit_chis = []
        fit_pers = []
        
        trial_pers = np.arange(best_per - dp, best_per + dp + 0.0000001, 0.000001)
        for per in tqdm(trial_pers, ascii=True):
            #obtain periods and chi_squares to use for parabolic fitting in determining the period uncertainty
            if per != 0:
                phi, chi2, tck, ndof = best_phase_chi2(time, mag, error, per, n_intervals)
                if tck is not None:
                    if chi2/ndof <= best_reduced_chi2 + 3:
                        fit_chis.append(chi2/ndof)
                        fit_pers.append(per)

        
        try: 
            #attempt to fit a parabola to the periods vs chi-square plot
            init_coeffs = np.polyfit(fit_pers, fit_chis, 2)
            a, best_per, best_reduced_chi2 = fit_minimum(fit_pers, fit_chis, init_coeffs[0], best_per, best_reduced_chi2)
            
            z = symbols('z')
            parab_fit = a * (z - best_per)**2 + best_reduced_chi2
            chi_eq = Eq(parab_fit, best_reduced_chi2 + 1)
            per_window = solve(chi_eq, z)
            per_error = best_per - float(min(per_window))
            
            y_poly = []
            buffer = 2 * per_error * 0.05
            x_poly = np.linspace(best_per - per_error - buffer, best_per + per_error + buffer, 1000)
            for x_fit in x_poly:
                y_fit = a * (x_fit - best_per)**2 + best_reduced_chi2
                y_poly.append(y_fit)
            
        except:
            #return not-a-number if fail
            per_error = np.nan

       
    
        plt.figure(1, figsize=(6,4))
        plt.scatter(fit_pers, fit_chis, marker='.', label='Cubic Spline Chi-Squares', zorder=1)
        try:
            #attempt to plot chi-square vs period plot with parabolic fit and 1-sigma estimation lines
            plt.plot(x_poly, y_poly, color='tab:orange', label='Parabolic Fit', zorder=2)
            plt.scatter(best_per, best_reduced_chi2, marker='x', color='tab:orange', s=150, label='Parabola Minimum (Best Period)', zorder=3)
            left = [min(per_window), min(per_window)]
            right = [max(per_window), max(per_window)]
            plt.plot(per_window, [best_reduced_chi2, best_reduced_chi2], linestyle=':', color='tab:green', label='1-Sigma Period Window', zorder=2)
            plt.plot(left, [best_reduced_chi2, best_reduced_chi2+1], linestyle=':', color='tab:green', zorder=2)
            plt.plot(right, [best_reduced_chi2, best_reduced_chi2+1], linestyle=':', color='tab:green', zorder=2)
            plt.scatter(best_per - per_error, best_reduced_chi2+1, marker='+', color='tab:green', s=150, label='1-Sigma Period Values', zorder=3)
            plt.scatter(best_per + per_error, best_reduced_chi2+1, marker='+', color='tab:green', s=150, zorder=3)
        except:
            print('Period uncertainty could not be determined - poor parabolic fit.')
        plt.title(f'Chi-Square vs Period, Uncertainty Determination (dp={dp})')
        plt.legend()
        plt.xlabel('Period')
        plt.ylabel('Chi-Square')
        plt.show()

    print(f"\nBest Period plus Error: {best_per:.6f} +/- {per_error:.6f}")
    print(f"Chi-square: {best_chi2:.6f}")
    print(f"Reduced Chi-square: {best_chi2/best_ndof:.6f} (ndof={best_ndof})")
    print(f"Phase offset: {best_phi:.6f}")

    #write txt file
    outname = os.path.splitext(os.path.basename(file_name))[0] + ".txt"
    with open(outname, "w") as f:
        f.write(f"{len(time)}\n")
        f.write(f"{time.min():10.3f}{time.max():10.3f}{mag.min():11.5f}{mag.max():11.5f}\n")
        for xi, yi, dyi in zip(time, mag, error):
            f.write(f"{xi:8.3f}{yi:11.5f}{dyi:12.6f}\n")
        f.write(f"{2*len(time)}\n")
        f.write(f"{0.0:8.3f}{2.0:8.3f}{mag.min():11.5f}{mag.max():11.5f}\n")
        f.write(f"{best_per:10.6f}{n_intervals:10d}\n")
        phase = phase_fold(time, best_per, best_phi)
        for xi, yi, dyi in zip(phase, mag, error):
            f.write(f"{xi:9.5f}{yi:11.5f}{dyi:12.6f}\n")
            f.write(f"{xi+1.0:9.5f}{yi:11.5f}{dyi:12.6f}\n")
        f.write(f"{512}\n")
        for i in range(512):
            p = 2 * i / 511
            folded = (p - best_phi + 1) % 1
            val = splev(folded, best_tck)
            f.write(f"{p:9.5f}{val:14.5f}\n")

    if save_plot:
        #display phase plot with plotted cubic spline curve (saves as pdf)
        phi, chi2, tck, ndof = best_phase_chi2(time, mag, error, best_per, n_intervals)
        phase = phase_fold(time, best_per, phi)
        
        folded = phase[np.argsort(phase)]
        power = splev(folded, tck)
        p1 = np.max(power)
        p3 = np.min(power)
        amp = p1 - p3
        
        sorted_idx = np.argsort(phase)
        phase_sorted = list(phase[sorted_idx])
        y_sorted = list(mag[sorted_idx])
        dy_sorted = list(error[sorted_idx])
        
        for i in range(len(phase)):
            phase_sorted.append(phase_sorted[i] + 1)
            y_sorted.append(y_sorted[i])
            dy_sorted.append(dy_sorted[i])
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8))
        ax1.errorbar(time, mag, yerr=error, fmt='k.')
        ax1.invert_yaxis()
        ax1.set_title(f'PER: {best_per:.6f}±{per_error:.6f}   AMP: {amp:.3f}')
        ax1.set_xlabel('Time (MJD)')
        ax1.set_ylabel('Magnitude')
        ax2.errorbar(phase_sorted, y_sorted, yerr=dy_sorted, fmt='k.', zorder = 1)
        ax2.invert_yaxis()
        ax2.set_title(f'PER: {best_per:.6f}±{per_error:.6f}  AMP: {amp:.3f}')
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Magnitude')
        phase_grid = np.linspace(0, 1, 500)
        plt.plot(phase_grid, splev(phase_grid, tck), 'r-', label='Spline Fit', zorder=2)
        plt.plot(phase_grid+1, splev(phase_grid, tck), 'r-', label='Spline Fit', zorder=2)
        plt.tight_layout()
        plotname = os.path.splitext(os.path.basename(file_name))[0] + ".pdf"
        plt.savefig(plotname)
        plt.show()

    return {
        "period": best_per,
        "chi2": best_chi2,
        "reduced_chi2": best_reduced_chi2,
        "ndof": best_ndof,
        "phi": best_phi,
        "spline": best_tck,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Light curve data file (.dat)")
    parser.add_argument("--max_per", "-up", type=float, default=1.0, help="Maximum period to scan")
    parser.add_argument("--min_per", "-lp", type=float, default=0.0, help="Minimum period to scan")
    parser.add_argument("--n_intervals", "-n", type=int, default=8, help="Number of cubic splines to fit with")
    parser.add_argument("--dp", "-dp", type=float, default=0.0001, help="Distance from best period to fit parabola for 1-sigma estimation")
    parser.add_argument("--no_plot", "-xplot", action="store_true", help="Display single_phase plot")
    parser.add_argument("--force_per", "-fper", type=float, help="Force a specific period without uncertainty calculation")
    parser.add_argument("--refine_per", "-rper", type=float, help="Refine period around a specific initial value")
    args = parser.parse_args()

    single_phase(
        args.file,
        max_per=args.max_per,
        min_per=args.min_per,
        n_intervals=args.n_intervals,
        save_plot=not args.no_plot,
        force_per=args.force_per,
        refine_per=args.refine_per,
        dp=args.dp
    )
