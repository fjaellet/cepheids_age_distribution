# MIT License
# 2022-2024 F. Anders (ICCUB)

import numpy as np

from astropy.table import Table, Column, MaskedColumn
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from scipy.interpolate import interp1d, interp2d
import astropy.table

import abj2016

def calc_geometric_distance(parallax, parallax_error, prior="exponential"):
    """
    Calculates simple Bayesian distances from parallax measurements
    à la Astraatmadja & Bailer-Jones 2016

    See abj2016.py for details.
    """
    distpdf = abj2016.distpdf(parallax, parallax_error, prior=prior)
    return distpdf.modedist

def calc_galcen_coords(l, b, dist, Rsun = 8.2):
    """
    Calculates Galactocentric coordinates from Galactic (l, b) 
    and distance.
    """
    X   = np.cos(b * np.pi / 180.) * np.cos(l * np.pi / 180.) * dist - Rsun
    Y   = np.cos(b * np.pi / 180.) * np.sin(l * np.pi / 180.) * dist
    R   = np.sqrt( X**2. + Y**2. )
    Phi = np.arctan2(Y, X) * 180. / np.pi + 180.
    Z   = np.sin(b * np.pi / 180.) * dist
    return X, Y, Z, R, Phi

def interpolate_anderson_a123(mass, met, crossing=1):
    """
    Interpolate the ages & delta ages of the instability strip
    from Tables A1, A2, & A3 of Anderson+2016 
    
    Input:
        mass   - float (or float array)
        met    - float (or float array)
    Output:
        tuple(logage, delta_logage) - (time of entry into the instability strip,
                                       time spent inside the IS)
    """
    # for omega_ini = 0.5
    masses_1x = np.array([2., 2.5, 3., 4., 5., 7., 9., 12., 15.])
    masses_2x = np.array([5., 7., 9.])
    masses_3x = np.array([5., 7., 9.])
    ages_1x = np.array([9.108, 8.833, 8.614, 8.283, 8.046, 7.715, 7.503, 7.273, 7.140])
    ages_2x = np.array([8.085, 7.747, 7.528])
    ages_3x = np.array([8.094, 7.765, 7.546])
    d_ages_1x = np.array([7.188, 6.068, 5.44, 4.854, 4.401, 3.789, 3.461, 3.404, 4.694])
    d_ages_2x = np.array([5.878, 4.718, 3.981])
    d_ages_3x = np.array([6.409, 4.9, 4.268])
    if crossing==1:
        return np.interp(mass, masses_1x, ages_1x), np.interp(mass, masses_1x, d_ages_1x)
    elif crossing==2:
        return np.interp(mass, masses_2x, ages_2x), np.interp(mass, masses_2x, d_ages_2x)
    elif crossing==3:
        return np.interp(mass, masses_3x, ages_3x), np.interp(mass, masses_3x, d_ages_3x)

def interpolate_anderson_a2(mass, crossing=1):
    """
    Interpolate the ages & delta ages of the instability strip
    from Table A2 of Anderson+2016
    
    """
    # for omega_ini = 0.5
    masses_1x = np.array([2., 2.5, 3., 4., 5., 7., 9., 12., 15.])
    masses_2x = np.array([4., 5., 7., 9.])
    masses_3x = np.array([4., 5., 7., 9.])
    ages_1x = np.array([9.017, 8.746, 8.537, 8.229, 8.005, 7.696, 7.495, 7.306, 7.165])
    ages_2x = np.array([8.289, 8.053, 7.723, 7.520])
    ages_3x = np.array([8.293, 8.062, 7.744, 7.536])
    d_ages_1x = np.array([6.666, 5.743, 5.23, 4.674, 4.238, 3.683, 3.495, 4.601, 4.5])
    d_ages_2x = np.array([6.003, 6.286, 4.520, 3.93])
    d_ages_3x = np.array([5.817, 5.821, 4.605, 4.353])
    if crossing==1:
        return np.interp(mass, masses_1x, ages_1x), np.interp(mass, masses_1x, d_ages_1x)
    elif crossing==2:
        return np.interp(mass, masses_2x, ages_2x), np.interp(mass, masses_2x, d_ages_2x)
    elif crossing==3:
        return np.interp(mass, masses_3x, ages_3x), np.interp(mass, masses_3x, d_ages_3x)

def interpolate_anderson_a3(mass, crossing=1):
    """
    Interpolate the ages & delta ages of the instability strip
    from Table A3 of Anderson+2016
    
    """
    # for omega_ini = 0.5
    masses_1x = np.array([2., 2.5, 3., 4., 5., 7., 9., 12., 15.])
    masses_2x = np.array([3., 4., 5., 7.])
    masses_3x = np.array([3., 4., 5., 7.])
    ages_1x = np.array([8.935, 8.671, 8.473, 8.181, 7.971, 7.677, 7.524, 7.302, 7.166])
    ages_2x = np.array([8.533, 8.206, 7.983, 7.682])
    ages_3x = np.array([8.543, 8.242, 8.027, 7.722])
    d_ages_1x = np.array([6.194, 5.482, 5.197, 4.551, 4.184, 3.977, 3.774, 3.794, 4.101])
    d_ages_2x = np.array([6.849, 6.286, 4.520, 3.93])
    d_ages_3x = np.array([5.817, 5.821, 4.605, 4.353])
    if crossing==1:
        return np.interp(mass, masses_1x, ages_1x), np.interp(mass, masses_1x, d_ages_1x)
    elif crossing==2:
        return np.interp(mass, masses_2x, ages_2x), np.interp(mass, masses_2x, d_ages_2x)
    elif crossing==3:
        return np.interp(mass, masses_3x, ages_3x), np.interp(mass, masses_3x, d_ages_3x)


def get_cepheidages(period, feh, usefeh=True, mode="fundamental"):
    """
    Compute cepheid ages using the recipe described in Skowron+2019 & Dekany+2019 
    """
    if usefeh:
        # transform [Fe/H] to absolute metallicity 
        Z_proxy = 10.**(feh - 1.77)
    else:
        # assume that all Cepheids have solar abundances
        Z_proxy = 0.014
        
    # get the tabulated relation from the Geneva models given in Anderson+2016, Tab. 4
    if mode=="fundamental":
        Zvals     = np.array([0.014, 0.006, 0.002])
        alphavals = np.array([-.592, -.665, -.84])
        betavals  = np.array([8.476, 8.628, 8.794])
    elif mode=="firstovertone":
        Zvals     = np.array([0.014, 0.006, 0.002])
        alphavals = np.array([-.633, -.825, -.961])
        betavals  = np.array([8.406, 8.651, 8.768])

    # quadratic interpolation between the 3 Z values
    alphaZ = interp1d(Zvals, alphavals, kind="linear", fill_value="extrapolate")
    betaZ  = interp1d(Zvals, betavals,  kind="linear", fill_value="extrapolate")
    # Period - age relation
    logt  = alphaZ(np.array(np.maximum(-0.4, Z_proxy))) * np.log10(period) + \
            betaZ(np.array(np.maximum(-0.4, Z_proxy)))
    # But: Only use ages for [Fe/H] > -0.4
    #table["logt"][ (table["FeH_proxy"]<-0.4) | (table["logt"]<0) ] = np.nan
    return logt
    
def get_cepheiddistances_ripepi2019(period, G, RP):
    # PW relation parameters for MW DCEP_F stars with unknown [Fe/H] (Ripepi+2019, Table 3, 1st line):
    alpha = -2.701 
    beta  = -3.32
    W_A = alpha + beta * np.log10(period)
    # Wesenheit definition in the Gaia passbands according to Clementini+2019 (Eq. 5):
    W   = G - 0.08193 -2.98056 * (G-RP) - 0.21906*(G-RP)**2. - 0.6378*(G-RP)**3.
    distance = 10.**( 0.2* (W-W_A) -2 ) # in kpc
    return distance
    
def get_cepheiddistances_ripepi2022(period, G, BP, RP, feh):
    # PW relation parameters for MW DCEP_F stars with known [Fe/H] (Ripepi+2022, Table 2, 3rd line):
    alpha = -5.948 
    beta  = -3.165
    gamma = -0.725
    # W = α + β(log10 P − 1.0) + γ[Fe/H]
    W_A    = alpha + beta * (np.log10(period) - 1) + gamma * feh
    # Wesenheit definition in the Gaia passbands according to Ripepi+2022 (Eq. 2):
    W   = G - 1.9 * (BP-RP) + 0.01
    distance = 10.**( 0.2* (W-W_A) -2 ) # in kpc
    return distance
    
def get_cepheiddistances_ripepi2022_no_feh(period, G, BP, RP):
    # PW relation parameters for MW DCEP_F stars with unknown [Fe/H] (Ripepi+2022, Table 2, 1st line):
    alpha_no = -6.023 
    beta_no  = -3.301
    # W = α + β(log10 P − 1.0)
    W_A_no = alpha_no + beta_no * (np.log10(period) - 1)
    # Wesenheit definition in the Gaia passbands according to Ripepi+2022 (Eq. 2):
    W   = G - 1.9 * (BP-RP) + 0.01
    distance = 10.**( 0.2* (W-W_A_no) -2 ) # in kpc
    return distance
    
####### OBSOLETE #############

def pietrukowicz2021_OLD(ages=True, fundamental=True, distance=True, **kwargs):
    """
    Get the cepheid compilation of Pietrukowicz+2021 (Acta Astronomica)
    """
    # Data from https://www.astrouw.edu.pl/ogle/ogle4/OCVS/allGalCep.listID
    pietru = Table.read("./data/Pietrukowicz2021_x_GaiaEDR3.fits")
    if fundamental:
        # Use only fundamental-mode Cepheids:
        pietru = pietru[(pietru["Mode"] == "F     ") | (pietru["Mode"] == "F1O   ")]
    # Compute distance
    pietru["distance"] = get_cepheiddistances(pietru["Period"], pietru["phot_g_mean_mag_corrected"],
                                              pietru["phot_rp_mean_mag"])
    if distance:
        pietru = pietru[(pietru["distance"] > 0.01)]
    pietru["X"] = np.cos(pietru["b"] * np.pi / 180.) * np.cos(pietru["l"] * np.pi / 180.) * pietru["distance"] - 8.2
    pietru["Y"] = np.cos(pietru["b"] * np.pi / 180.) * np.sin(pietru["l"] * np.pi / 180.) * pietru["distance"]
    pietru["R"]   = np.sqrt( (pietru["X"])**2. + (pietru["Y"])**2. )
    pietru["Phi"] = np.arctan2(pietru["Y"], pietru["X"]) * 180. / np.pi + 180.
    pietru["Z"] = np.sin(pietru["b"] * np.pi / 180.) * pietru["distance"]
    if ages:
        # Determine log age with the method of Dekany+2019
        pietru_f  = get_cepheidages(pietru[(pietru["Mode"] == "F     ") | (pietru["Mode"] == "F1O   ")],  mode="fundamental", **kwargs)
        pietru_1o = get_cepheidages(pietru[(pietru["Mode"] == "1O    ") | (pietru["Mode"] == "1O2O  ")], mode="firstovertone", **kwargs)
        pietru = astropy.table.vstack([pietru_f, pietru_1o])
    print(len(pietru), "objects from Pietrukowicz+2021")
    return pietru

