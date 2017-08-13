import re
from collections import OrderedDict

from astropy.table import Table, vstack
import Ska.ftp
import requests
from astropy.io import ascii


TWIKI_URL_ROOT = 'https://occweb.cfa.harvard.edu/twiki/pub/'


def flatten_pea_test_data(dat):
    """
    Take PEA test set table with 8 samples per row and flatten into
    a single table with a new "slot" column.
    """
    out_names = OrderedDict()
    for in_name in dat.colnames:
        m = re.match(r'(\w+)_(\d)(_\w+)?$', in_name)
        if m:
            start, col_slot, end = m.groups()
            out_name = start + (end or '')  # Unique output name
            if col_slot == '0':
                out_names[out_name] = []
            out_names[out_name].append(in_name)
        else:
            out_name = in_name
            out_names[out_name] = [in_name] * 8

    out_tables = []
    for slot in range(8):
        names_for_slot = [out_names[name][slot] for name in out_names]
        out_table = Table(dat[names_for_slot], names=list(out_names))
        out_table['slot'] = slot
        out_tables.append(out_table)

    out = vstack(out_tables)
    return out


def read_twiki_csv(filename, web='Aspect', auth=None):
    if auth is None:
        auth = Ska.ftp.parse_netrc()['occweb']
        auth = (auth['login'], auth['password'])  # auth for requests

    url = TWIKI_URL_ROOT + web + '/' + filename
    r = requests.get(url, auth=auth)
    if r.status_code != 200:
        raise ValueError(f'query of URL {url} failed: status code {r.status_code}')

    dat = ascii.read(r.text)
    return dat
