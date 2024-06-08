##################################################################################
#
#   open_basti: opens BaSTI isochrone files and joins them
#
#   Contains functions:
#
#             - open_one_basti_file
#             - open_basti
#
##################################################################################

import numpy as np
from os.path import join
from os import listdir
from astropy.table import Table, vstack

def open_one_basti_file(path, met="FEHm010"):
    """
    Opens one BASTI model with Gaia EDR3 abs magnitudes.
    Returns:
        astropy table
    """
    t = Table.read(path, format="ascii")
    # rename the columns
    t['col1'].name = 'logage'
    t['col2'].name = 'm_ini'
    t['col3'].name = 'logl'
    t['col4'].name = 'logteff'
    t['col5'].name = 'G'
    t['col6'].name = 'G_BP'
    t['col7'].name = 'G_RP'
    # add a metallicity column
    if met[3] == "m":
        t["m_h"] = - np.ones(len(t)) * 0.01*np.float32(met[5:])
    elif met[3] == "p":
        t["m_h"] = + np.ones(len(t)) * 0.01*np.float32(met[5:])
    return t
    
def open_basti(basti_dir="./isochrones/BaSTI/"):
    """
    Opens all locally available BASTI models with Gaia EDR3 abs magnitudes and stacks them.
    Returns:
        astropy table
    """
    # get directories of the metallicity-separated files:
    met_folders = listdir(basti_dir)
    # iterate through them
    for ii in np.arange(len(met_folders)):
        # List all files within a folder
        files_ii = listdir( join(basti_dir, met_folders[ii]) )
        # read the files iteratively with astropy.table
        for jj in np.arange(len(files_ii)):
            path_ii_jj  = join(basti_dir, met_folders[ii], files_ii[jj])
            table_ii_jj = open_one_basti_file( path_ii_jj, met=met_folders[ii] )
            if ii == 0 and jj == 0:
                tab = table_ii_jj
            else:
                tab = vstack([tab, table_ii_jj])
    return tab
            

