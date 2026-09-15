#!/usr/bin/env python3
"""
ROTSE-I calibration-only version.
--Vaisakh
This version intentionally preserves ORIGINAL calibration logic of lccal_2.1.py as closely as possible, while removing the unconex/filtering machinery.

Preserved calibration logic:
    - target identification by RA/Dec
    - reference-star search within radius
    - reference-star cuts used by the original calibration code
    - reference-star baseline using the original avmag() flux-average
    - exact target/reference epoch matching
    - per-epoch reference deltas
    - SIMPLE (unweighted) mean reference correction
    - target magnitude + correction

Removed:
    - R1_unconex
    - R3_unconex
    - pair averaging
    - target filtering
    - limiting-magnitude filtering of target observations
    - test-star calibration
    - nightly selection
    - plotting/filtering functionality

This file is intended as a control/baseline against which the modified
calibration-only version can be compared.
"""

import math
import argparse
import glob
import os
import sys
import numpy as np
from scipy.io import readsav
from astropy.io import fits


# ----------------------------------------------------------------------
# Original input functions
# ----------------------------------------------------------------------

def read_fits_file(file, fits_index=1):
    try:
        hdus = fits.open(file, memmap=True)
        hdus_ext = hdus[fits_index]
        match = hdus_ext.data
    except Exception as e:
        raise Exception("cannot read fits data from file: %s" % (file,)) from e
    return match, 'ROTSE3'


def read_match_file(file, *args, **kwargs):
    try:
        match = readsav(file)['match']
    except Exception as e:
        raise Exception("cannot read match data from file: %s" % (file,)) from e
    return match, 'ROTSE1'


def read_data_file(file, fits_index=1, tmpdir='/tmp'):
    if not os.path.isfile(file):
        raise Exception("file not found: %s" % (file,))
    file_ext = file.rpartition('.')[2]

    if file_ext == 'fit':
        match, rotse = read_fits_file(file, fits_index)
    else:
        match, rotse = read_match_file(file)

    return match, rotse


def get_matchstructs(match_structures):
    cwd = os.getcwd()
    os.chdir(match_structures)

    temp_matchs = []
    fits_files = glob.glob("*.fit")
    dats = glob.glob("*.dat")
    datcs = glob.glob("*.datc")

    for fit in fits_files:
        temp_matchs.append(fit)
    for dat in dats:
        temp_matchs.append(dat)
    for datc in datcs:
        temp_matchs.append(datc)

    return temp_matchs, cwd


def get_data(refra, refdec, match):
    match_file = None

    if isinstance(match, str):
        match_file = match
        match, tele = read_data_file(match_file)

    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]

    cond = np.logical_and.reduce(
        (
            np.abs(match_ra - refra) < 0.001,
            np.abs(match_dec - refdec) < 0.001
        )
    )

    goodobj = np.where(cond)
    objid = goodobj[0]

    match_m_lim = match['STAT'][0]['M_LIM']
    match_exptime = match.field('EXPTIME')[0]
    match_merr = match.field('MERR')[0][objid][0]
    match_m = match.field('M')[0][objid][0]
    match_jd = match.field('JD')[0]

    curve = []

    for q in range(len(match_jd)):
        epoch = match_jd[q]
        mag = match_m[q]
        magerr = match_merr[q]
        exptime = match_exptime[q] / 86400
        m_lim = match_m_lim[q]
        curve.append((epoch, mag, magerr, exptime, m_lim))

    return curve


def find_target(vra, vdec, temp_matchs):
    matchs = []
    target_lc = []

    for match in temp_matchs:
        try:
            lc = get_data(vra, vdec, match)

            for i in lc:
                target_lc.append(i)

            print(f"Target found in {match}")
            matchs.append(match)

        except IndexError:
            print(
                f"Cannot find target in {match}; "
                "this match structure was removed from the list"
            )

    return matchs, target_lc


def getobjids(inmatch, refra, refdec, radius):
    match_file = None

    if isinstance(inmatch, str):
        match_file = inmatch
        match, tele = read_data_file(match_file)
    else:
        match = inmatch

    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]

    cond = np.logical_and.reduce(
        (
            np.abs(match_ra - refra) <= radius,
            np.abs(match_dec - refdec) <= radius
        )
    )

    objects = list(np.where(cond)[0])
    goodobj = []

    for x in objects:
        coords = getcoords(inmatch, x)

        if math.sqrt(
            (coords[0] - refra) ** 2 +
            (coords[1] - refdec) ** 2
        ) <= radius:
            goodobj.append(x)

    return goodobj


def getcoords(match, objid):
    if isinstance(match, str):
        match, tele = read_data_file(match)

    match_ra = match.field('RA')[0]
    match_dec = match.field('DEC')[0]

    return [list(match_ra)[objid], list(match_dec)[objid]]


# ----------------------------------------------------------------------
# Original photometric baseline
# ----------------------------------------------------------------------

def mag2flux(in_mag):
    return float(3.636 * 10 ** (-float(in_mag) / 2.5))


def flux2mag(in_flux):
    return float(-2.5 * math.log10(float(in_flux) / 3.636))


def avmag(data):
    """
    ORIGINAL calibration baseline:
    average the fluxes, then convert the average flux back to magnitude.
    """
    fluxs = []

    for row in data:
        fluxs.append(mag2flux(row[1]))

    avflux = math.fsum(fluxs) / len(fluxs)
    return flux2mag(avflux)


# ----------------------------------------------------------------------
# Original calibration logic
# ----------------------------------------------------------------------

def find_refstars(matchs, ra, dec, radius, target_lc,
                  requested_refstars=5,
                  max_mean_error=0.06,
                  decent_epochs_input=0.9,
                  use_avmag=True):
    """
    Preserve the original reference-star selection logic, except that the
    dead --avmag switch is made explicit here.

    Original cuts retained:
        1. not the target
        2. mean photometric error <= max_mean_error
        3. exact same epoch list as target
        4. >= decent_epochs fraction of good observations
        5. original magnitude-window criterion
    """

    def cuts(package):
        cand_coords = package[0]
        lightcurve = package[1]
        per_match = package[2]
        good_obs = package[3]

        allowed_diff = 0.001

        # Original target exclusion logic.
        if (
            not ra - allowed_diff <= cand_coords[0] <= ra + allowed_diff
            and
            not dec - allowed_diff <= cand_coords[1] <= dec + allowed_diff
        ):
            is_not_target = True
        else:
            is_not_target = False

        if not is_not_target:
            return False

        if max_mean_error is not False:
            if (
                math.fsum([obs[2] for obs in good_obs]) /
                len(good_obs)
            ) > max_mean_error:
                return False

        target_epochs = [obs[0] for obs in target_lc]
        candidate_epochs = [obs[0] for obs in lightcurve]

        if target_epochs != candidate_epochs:
            return False

        av_m_lim = (
            math.fsum([obs[4] for obs in lightcurve]) /
            len(lightcurve)
        )

        if use_avmag:
            candidate_avmag = avmag(lightcurve)

            if not (
                av_m_lim - 4 <= candidate_avmag <= av_m_lim
            ):
                return False

        if (
            len(
                [
                    obs[1]
                    for obs in good_obs
                    if obs[4] - 4 <= obs[1] <= obs[4]
                ]
            ) / len(lightcurve)
            < decent_epochs_input
        ):
            return False

        return True

    refstars = []
    surroundstars = getobjids(matchs[0], ra, dec, radius)
    test_candidates = []

    for star in surroundstars:
        try:
            coords = getcoords(matchs[0], star)

            full_lightcurve = []
            lightcurves_per_match = []

            for match in matchs:
                match_lightcurve = sorted(
                    get_data(coords[0], coords[1], match),
                    key=lambda x: x[0]
                )

                lightcurves_per_match.append(match_lightcurve)
                full_lightcurve.extend(match_lightcurve)

            good_lightcurve = [
                obs for obs in full_lightcurve
                if 0 < obs[1] < 99
            ]

            package = [
                coords,
                full_lightcurve,
                lightcurves_per_match,
                good_lightcurve
            ]

            if cuts(package):
                refstars.append(
                    [coords, avmag(good_lightcurve), good_lightcurve]
                )

            else:
                allowed_diff = 0.001

                if (
                    not ra - allowed_diff <= coords[0] <= ra + allowed_diff
                    and
                    not dec - allowed_diff <= coords[1] <= dec + allowed_diff
                ):
                    test_candidates.append(
                        [coords, good_lightcurve, lightcurves_per_match]
                    )

        except IndexError:
            pass

    # Original script ultimately chooses the nearest requested references.
    refstars.sort(
        key=lambda x:
        math.sqrt(
            (x[0][0] - ra) ** 2 +
            (x[0][1] - dec) ** 2
        )
    )

    return refstars[:requested_refstars]


def get_corrections(refstars, target_lc):
    """
    ORIGINAL calibration:
        correction = simple arithmetic mean of reference deltas.

    No error weighting, no outlier rejection, no robust estimator.
    """
    corrections = []

    target_epochs = [obs[0] for obs in target_lc if 0 < obs[1] < 99]

    for epoch in target_epochs:
        diffs = []
        ref_errors = []

        for star in refstars:
            trumag = star[1]
            lightcurve = star[2]

            for obs in lightcurve:
                if obs[0] == epoch:
                    diffs.append(trumag - obs[1])
                    ref_errors.append(obs[2])

        if len(diffs) == len(refstars):
            correction = math.fsum(diffs) / len(diffs)
            calibration_error = (
                math.sqrt(math.fsum(err ** 2 for err in ref_errors))
                / len(ref_errors)
            )
            corrections.append([epoch, correction, calibration_error])

    return corrections


def calibrate_target(corrections, target_lc):
    """
    ORIGINAL calibration:
        calibrated magnitude = target magnitude + correction

    Original target magnitude uncertainty is retained unchanged.
    """
    calibrated = []

    target_good = [
        obs for obs in target_lc
        if 0 < obs[1] < 99
    ]

    for obs in target_good:
        for correction in corrections:
            if obs[0] == correction[0]:
                calibrated.append(
                    [
                        obs[0],
                        obs[1] + correction[1],
                        math.sqrt(obs[2] ** 2 + correction[2] ** 2),
                        obs[3],
                        obs[4],
                    ]
                )

    return calibrated


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ROTSE-I original calibration logic, filtering removed"
    )

    parser.add_argument("match_structures")
    parser.add_argument("vra", type=float)
    parser.add_argument("vdec", type=float)
    parser.add_argument("--requested-refstars", "-ref", type=int, default=5)
    parser.add_argument("--radius", "-r", type=float, default=0.1)
    parser.add_argument("--max-mean-error", "-e", type=float, default=0.06)
    parser.add_argument("--decent-epochs", "-d", type=float, default=0.9)
    parser.add_argument("--out", default="calibrated_lightcurve_oldlogic.dat")

    args = parser.parse_args()

    files, cwd = get_matchstructs(args.match_structures)

    if not files:
        raise RuntimeError("No match structures found")

    matchs, target_lc = find_target(
        args.vra,
        args.vdec,
        files
    )

    if not target_lc:
        raise RuntimeError("Target not found")

    refs = find_refstars(
        matchs,
        args.vra,
        args.vdec,
        args.radius,
        target_lc,
        requested_refstars=args.requested_refstars,
        max_mean_error=args.max_mean_error,
        decent_epochs_input=args.decent_epochs,
        use_avmag=True,
    )

    if len(refs) < args.requested_refstars:
        print(
            f"WARNING: requested {args.requested_refstars} references, "
            f"but only {len(refs)} passed the original calibration cuts."
        )

    print(f"Using {len(refs)} reference stars.")

    for i, ref in enumerate(refs, 1):
        print(
            f"REF{i}: RA={ref[0][0]}, Dec={ref[0][1]}, "
            f"baseline={ref[1]:.4f}, N={len(ref[2])}"
        )

    corrections = get_corrections(refs, target_lc)
    calibrated = calibrate_target(corrections, target_lc)

    os.chdir(cwd)

    with open(args.out, "w") as f:
        f.write(
            "# MJD calibrated_mag calibrated_mag_err "
            "exptime_days limiting_mag zeropoint_correction\n"
        )

        correction_dict = {c[0]: c[1] for c in corrections}

        for obs in calibrated:
            f.write(
                f"{obs[0]:.8f} "
                f"{obs[1]:.6f} "
                f"{obs[2]:.6f} "
                f"{obs[3]:.8f} "
                f"{obs[4]:.6f} "
                f"{correction_dict[obs[0]]:.6f}\n"
            )

    print(f"Target observations: {len(target_lc)}")
    print(f"Calibrated observations: {len(calibrated)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()

