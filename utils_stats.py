import itertools
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import agasc
import numpy as np
import numpy.typing as npt
import requests
import Ska.ftp
from astropy.io import ascii
from astropy.table import Table, vstack
from cxotime import CxoTime
from scipy.stats import binom
import matplotlib.pyplot as plt
import matplotlib.patches as patches


TWIKI_URL_ROOT = "https://occweb.cfa.harvard.edu/twiki/pub/"
SKA = Path(os.environ["SKA"])


@dataclass
class ValsBins:
    vals: tuple

    @cached_property
    def bins(self):
        return get_bins(self.vals)


@dataclass
class AcqBins:
    mag: ValsBins
    t_ccd: ValsBins
    halfwidth: ValsBins = field(
        default_factory=lambda: ValsBins((60, 80, 100, 120, 140, 160))
    )

    @cached_property
    def shape(self):
        return (
            len(self.mag.vals),
            len(self.t_ccd.vals),
            len(self.halfwidth.vals),
        )


def flatten_pea_test_data(dat):
    """
    Take PEA test set table with 8 samples per row and flatten into
    a single table with a new "slot" column.
    """
    out_names = OrderedDict()
    for in_name in dat.colnames:
        m = re.match(r"(\w+)_(\d)(_\w+)?$", in_name)
        if m:
            start, col_slot, end = m.groups()
            out_name = start + (end or "")  # Unique output name
            if col_slot == "0":
                out_names[out_name] = []
            out_names[out_name].append(in_name)
        else:
            out_name = in_name
            out_names[out_name] = [in_name] * 8

    out_tables = []
    for slot in range(8):
        names_for_slot = [out_names[name][slot] for name in out_names]
        out_table = Table(dat[names_for_slot], names=list(out_names))
        out_table["slot"] = slot
        out_tables.append(out_table)

    out = vstack(out_tables)
    return out


def read_twiki_csv(filename, web="Aspect", auth=None):
    if auth is None:
        auth = Ska.ftp.parse_netrc()["occweb"]
        auth = (auth["login"], auth["password"])  # auth for requests

    url = TWIKI_URL_ROOT + web + "/" + filename
    r = requests.get(url, auth=auth)
    if r.status_code != 200:
        raise ValueError(f"query of URL {url} failed: status code {r.status_code}")

    dat = ascii.read(r.text)
    return dat


def get_acq_stats_data(start="2019-07-01"):
    """
    Get acq stats data after ``start`` in a standard format.

    By default this returns data since 2019-07-01 which is when the MAXMAG change was
    put in place.
    """
    acq_file = SKA / "data" / "acq_stats" / "acq_stats.h5"
    acqs = Table.read(acq_file)

    # Remap and select columns. This includes
    names = {
        "tstart": "guide_tstart",
        "obsid": "obsid",
        "obc_id": "acqid",
        "halfwidth": "halfw",
        "mag_aca": "mag_aca",
        "mag_obs": "mag_obs",
        "known_bad": "known_bad",
        "color": "color1",
        "img_func": "img_func",
        "ion_rad": "ion_rad",
        "sat_pix": "sat_pix",
        "agasc_id": "agasc_id",
        "t_ccd": "ccd_temp",
        "slot": "slot",
        # Compatability with SWATS and ASVT naming
        "star_mag": "mag_aca",
        "search_success": "acqid",
        "search_box_hw": "halfw",
        "ccd_temp": "ccd_temp",
        "ObsID": "obsid",
    }
    acqs = Table({new_name: acqs[name] for new_name, name in names.items()})

    # Apply start filter
    ok1 = acqs["tstart"] > CxoTime(start).secs
    ok2 = ~acqs["ion_rad"].astype(bool) & ~acqs["sat_pix"].astype(bool)
    ok3 = ~np.isclose(acqs["color"], 1.5)
    acqs = acqs[ok1 & ok2 & ok3]

    stars = agasc.get_stars(acqs["agasc_id"], agasc_file="agasc*")
    acqs["mag_aca"] = stars["MAG_ACA"]
    acqs["star_mag"] = stars["MAG_ACA"]
    acqs["mag_catid"] = stars["MAG_CATID"]
    acqs["color"] = stars["COLOR1"]

    # Coerce uint8 columns (which are all actually bool) to bool
    for col in acqs.itercols():
        if col.dtype.type is np.uint8:
            col.dtype = bool

    # Add year and quarter columns for convenience
    year_q0 = 1999.0 + 31.0 / 365.25  # Jan 31 approximately
    acqs["year"] = CxoTime(acqs["tstart"], format="cxcsec").decimalyear.astype("f4")
    acqs["quarter"] = (np.trunc((acqs["year"] - year_q0) * 4)).astype("f4")

    # Remove known bad stars
    bad_stars = agasc.get_supplement_table("bad")
    bad = np.isin(acqs["agasc_id"], bad_stars["agasc_id"])
    bad |= acqs["known_bad"]
    acqs = acqs[~bad]

    del acqs["known_bad"]

    return acqs


def get_bins(vals: npt.ArrayLike) -> np.ndarray:
    """Get bins for a list of values that are sorted and unique"""
    vals = np.asarray(vals)
    centers = (vals[1:] + vals[:-1]) / 2
    bins = np.concatenate(
        [
            [vals[0] - (vals[1] - vals[0]) / 2],
            centers,
            [vals[-1] + (vals[-1] - vals[-2]) / 2],
        ]
    )
    return bins


def get_vals_and_bins(vals):
    """Get unique sorted vals and bins from a list of values that may have duplicates"""
    out_vals = np.array(sorted(set(vals)))
    out_val_centers = (out_vals[1:] + out_vals[:-1]) / 2
    out_val_bins = np.concatenate(
        [
            [out_vals[0] - (out_vals[1] - out_vals[0]) / 2],
            out_val_centers,
            [out_vals[-1] + (out_vals[-1] - out_vals[-2]) / 2],
        ]
    )
    return out_vals, out_val_bins


def get_samples_successes(
    dat,
    acq_bins,
    mag_name="star_mag",
    t_ccd_name="ccd_temp",
    halfwidth_name="search_box_hw",
    obc_id_name="search_success",
):
    """
    Aggregate binned number of samples and successes for ASVT data.

    Take the table of acquisition samples and return two 3-d arrays (mag, t_ccd,
    halfwidth):

    - n_samp: number of samples in each bin
    - n_succ: number of successes in each bin
    """
    mag_bins = acq_bins.mag.bins
    t_ccd_bins = acq_bins.t_ccd.bins
    halfwidth_bins = acq_bins.halfwidth.bins

    zeros = np.zeros(shape=acq_bins.shape, dtype=int)
    n_samp = zeros.copy()
    n_succ = zeros.copy()

    for ii, mag0, mag1 in zip(itertools.count(), mag_bins[:-1], mag_bins[1:]):
        ok0 = (dat[mag_name] >= mag0) & (dat[mag_name] < mag1)
        for jj, t_ccd0, t_ccd1 in zip(
            itertools.count(), t_ccd_bins[:-1], t_ccd_bins[1:]
        ):
            ok1 = (dat[t_ccd_name] >= t_ccd0) & (dat[t_ccd_name] < t_ccd1)
            for kk, halfwidth0, halfwidth1 in zip(
                itertools.count(), halfwidth_bins[:-1], halfwidth_bins[1:]
            ):
                ok2 = (dat[halfwidth_name] >= halfwidth0) & (
                    dat[halfwidth_name] < halfwidth1
                )
                ok = ok0 & ok1 & ok2
                n_samp[ii, jj, kk] = np.count_nonzero(ok)
                n_succ[ii, jj, kk] = np.count_nonzero(dat[obc_id_name][ok])

    return n_samp, n_succ


def as_summary_table(arr, acq_bins: AcqBins, fmt=None, add_mag=True):
    """Turn one of the summary 6x6 arrays into a readable table"""
    mag_vals = acq_bins.mag.vals
    t_ccd_vals = acq_bins.t_ccd.vals

    t = Table()
    if add_mag:
        t["mag"] = [str(val) for val in mag_vals]
    else:
        t["|"] = ["|"] * len(mag_vals)
    names = [f"{t_ccd:.1f}" for t_ccd in t_ccd_vals]
    for jj, name in enumerate(names):
        t[name] = arr[:, jj]
        if fmt:
            t[name].info.format = fmt
    return t


def pformat_sampling(
    dat,
    acq_bins: AcqBins,
    add_mag=True,
):
    lines = []
    n_samp_all, _ = get_samples_successes(dat, acq_bins=acq_bins)
    for kk, halfw in enumerate((60, 80, 100, 120, 140, 160)):
        n_samp = n_samp_all[:, :, kk]
        lines.append(f"Halfwidth {halfw}:")
        lines.extend(as_summary_table(n_samp, acq_bins, add_mag=add_mag).pformat_all())
        lines.append("")
    return lines


def calc_diff_pmf(p, pmf1, pmf2):
    dp = p[1] - p[0]  # assume uniform grid
    pmf1 = pmf1 / np.sum(pmf1)
    pmf2 = pmf2 / np.sum(pmf2)

    i0 = int(1 / dp)
    n_out = 2 * i0 + 1
    x = (np.arange(n_out) - i0) * dp
    out = np.zeros(n_out)
    p2 = p
    for i1, p1 in enumerate(p):
        d_pmf12 = pmf1[i1] * pmf2
        i_out = np.round((p1 - p2) / dp).astype(int) + i0
        out[i_out] += d_pmf12

    return x, np.cumsum(out)


def plot_diff_pmf(k1, n1, k2, n2, title="", l1="", l2="", axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax0 = axes[0]
    ax1 = axes[1]
    dp = 0.001
    p = np.arange(0.0 + dp / 2, 1.0, dp)
    pmf1 = binom.pmf(k1, n1, p)
    pmf2 = binom.pmf(k2, n2, p)
    dp, cdf = calc_diff_pmf(p, pmf1, pmf2)

    ax0.plot(p, pmf1, label=f"k={k1} n={n1} {l1}")
    ax0.plot(p, pmf2, label=f"k={k2} n={n2} {l2}")
    ax0.grid(True)
    if title:
        ax0.set_title(title)
    ax0.set_xlabel("p")
    ax0.legend(loc="best")

    ax1.plot(dp, cdf)
    ax1.grid(True)
    ax1.set_title("CDF of difference")
    i10, i90 = np.searchsorted(cdf, [0.1, 0.9])
    p10, p90 = dp[[i10, i90]]
    patch = patches.Rectangle((p10, 0.1), p90 - p10, 0.8, fc="r", alpha=0.2, ec="k")
    ax1.add_patch(patch)
    ax1.set_xlim(-0.5, 0.5)
    ax1.axvline(0, color="r", ls="--")
