import os
import re
from collections import OrderedDict
from pathlib import Path

import agasc
import numpy as np
import requests
import Ska.ftp
from astropy.io import ascii
from astropy.table import Table, vstack
from cxotime import CxoTime

TWIKI_URL_ROOT = "https://occweb.cfa.harvard.edu/twiki/pub/"
SKA = Path(os.environ["SKA"])


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

    # Remap and select columns
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
    }
    acqs = Table({new_name: acqs[name] for new_name, name in names.items()})

    # Apply start filter
    ok1 = acqs["tstart"] > CxoTime(start).secs
    ok2 = ~acqs['ion_rad'].astype(bool) & ~acqs['sat_pix'].astype(bool)
    acqs = acqs[ok1 & ok2]

    stars = agasc.get_stars(acqs['agasc_id'], agasc_file="agasc*")
    acqs['mag_aca'] = stars['MAG_ACA']
    acqs['mag_catid'] = stars['MAG_CATID']
    acqs["color"] = stars["COLOR1"]
    acqs = acqs[~np.isclose(acqs["color"], 1.5)]

    # Coerce uint8 columns (which are all actually bool) to bool
    for col in acqs.itercols():
        if col.dtype.type is np.uint8:
            col.dtype = bool

    # Add year and quarter columns for convenience
    year_q0 = 1999.0 + 31.0 / 365.25  # Jan 31 approximately
    acqs["year"] = CxoTime(acqs["tstart"], format="cxcsec").decimalyear.astype("f4")
    acqs["quarter"] = (np.trunc((acqs["year"] - year_q0) * 4)).astype("f4")

    # # Get latest mag estimates from the AGASC with supplement
    # mags_supp = agasc.get_supplement_table('mags')
    # mags_supp = dict(zip(mags_supp['agasc_id'], mags_supp['mag_aca']))
    # acqs["mag_aca"] = [
    #     mags_supp.get(agasc_id, mag_aca)
    #     for agasc_id, mag_aca in zip(acqs["agasc_id"], acqs["mag_aca"])
    # ]

    # Remove known bad stars
    bad_stars = agasc.get_supplement_table('bad')
    bad = np.isin(acqs['agasc_id'], bad_stars['agasc_id']) | acqs['known_bad']
    acqs = acqs[~bad]

    del acqs["known_bad"]

    return acqs
