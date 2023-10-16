from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import ligo.skymap.io
from astropy.io import fits
from astropy.table import Table, vstack, hstack, Column
import astropy.units as u
from astropy.coordinates import SkyCoord
import ligo.skymap.plot
from scipy.stats import norm
import pandas as pd
import os
import argparse
# import ligo.skymap.io
from ligo.skymap.io import read_sky_map
from ligo.skymap.plot import marker
from ligo.skymap import io
from ligo.skymap.distance import (
    parameters_to_marginal_moments, principal_axes, volume_render, marginal_pdf)
import scipy.stats
import astropy_healpix as ah
from ligo.skymap.tool.matplotlib import figure_parser
from ligo.skymap.tool import ArgumentParser, FileType
import ligo.skymap.tool


with fits.open('/home/growth/emgw_skymaps/config_tile_array/GIT_Observable_catalog.fits') as hdu:
    nedTable = Table(hdu[1].data)
cat = nedTable['objname', '_RAJ2000', '_DEJ2000', 'Dist', 'Mstar']
# cat.rename_column('DistMpc', 'Dist')
# cat.rename_column('ra', '_RAJ2000')
# cat.rename_column('dec', '_DEJ2000')

# inj_file = 'bayestar.fits'
curpath = os.path.abspath('.')
# areaPath = os.path.join(curpath, 'areaLoc')
csvPath = os.path.join(curpath, 'csvGal')


def getGalaxiesbyArea(cat):
    cat.sort('dP_dA_norm', reverse=True)
    cum_sort = np.cumsum(cat['dP_dA_norm'])
    cumnorm_sort = cum_sort/np.max(cum_sort)
    cat['P_A'] = cumnorm_sort

    # ID galaxies inside the 90% prob by volume
    icutarea90 = np.where(cat['P_A'] <= 0.95)
    clucutarea90 = cat[icutarea90]

    # generate astropy coordinate object for this sample
    clucutarea90coord = SkyCoord(ra=np.radians(
        clucutarea90['_RAJ2000'])*u.rad, dec=np.radians(clucutarea90['_DEJ2000'])*u.rad)

    return clucutarea90


def get_90_Loc_Vol(catFin, prob, distmu, distsigma, distnorm):
    # def getGalaxiesIn90Vol(catFin, prob):
    clucoord = SkyCoord(ra=catFin['_RAJ2000'] *
                        u.deg, dec=catFin['_DEJ2000']*u.deg)

    # make astropy coordinate object of CLU galaxies
    # load in healpix map
    npix = len(prob)
    nside = hp.npix2nside(npix)
    pixarea = hp.nside2pixarea(nside)

    # get coord of max prob density by area for plotting purposes
    ipix_max = np.argmax(prob)
    # print("Max prob pix", ipix_max)
    theta_max, phi_max = hp.pix2ang(nside, ipix_max)

    # calc hp index for each galaxy and populate CLU Table with the values
    theta = 0.5 * np.pi - clucoord.dec.to('rad').value
    phi = clucoord.ra.to('rad').value
    # print("phi", len(phi))
    # ipix = hp.ang2pix(nside, clucoord.dec, clucoord.ra, lonlat=True)
    ipix = hp.ang2pix(nside, theta, phi)

    dp_dV = prob[ipix] * distnorm[ipix] * \
        norm(distmu[ipix], distsigma[ipix]).pdf(
            np.array(catFin['Dist'].tolist()))/pixarea
    # print("dp_dV len", dp_dV)
    catFin['dP_dV'] = dp_dV

    # use normalized cumulative dist function to calculate Volume P-value for each galaxy
    catFin.sort('dP_dV', reverse=True)
    # clu.reverse()
    cum_sort = np.cumsum(catFin['dP_dV'])
    cumnorm_sort = cum_sort/np.max(cum_sort)
    catFin['P_N'] = cumnorm_sort

    # ID galaxies inside the 90% prob by volume
    icut90 = np.where(catFin['P_N'] <= 0.9)
    clucut90 = catFin[icut90]

    # generate an astropy coordinate object for this subset

    return catFin, clucut90, np.size(clucut90)

# This function fetches the nuber of galaxies within each pixel by grouping the galaxies which lie in same healpix pixel
# It also aggregates the total Mass/Probability*Mass based on the value of param_sort within  a healpix pixel


def getCatTileIds(catMass, param_sort):

    catMass_grouped = catMass.group_by('pixValue')
    prob_sum = catMass_grouped[param_sort].groups.aggregate(np.sum)
    prob_rev = (-prob_sum).argsort()[:len(prob_sum)]

    stacked_catMass = catMass_grouped.groups[0]

    for i, group in enumerate(catMass_grouped.groups):

        index = np.where(prob_rev == i)[0]

        group['TileId'] = index+1
        # group['TileId'] = prob_rev[i]+1

        # if i==1:
        #     print(group['Mass_prob'])
        if i > 0:
            stacked_catMass = vstack([stacked_catMass, group])
        else:
            stacked_catMass = group

    # stacked_catMass.sort('prob_mass', reverse=True)

    return stacked_catMass

# This function calculates the total mass/ probability*mass within a Telescope FOV Tile
# It also calculates the maximum mass/prob*mass in a tile


def get_probabilities(ra, dec, df_dict, nside_128, radius=0.35*u.deg):
    """
    Compute the probabilities covered in a grid of ra, dec with radius
    given a healpix skymap

    Pass the legal RA and Dec lists to this function

    Note: this radius is such that for the default sky grid (nside=128),
    fields overlap such that the area of the sky becomes 1.83 * 4pi steradians.
    As a result, the sum of probabilities will be 1.83
    """
    # fact : int, optional
    # Only used when inclusive=True. The overlapping test will be done at
    # the resolution fact*nside. For NESTED ordering, fact must be a power of 2, less than 2**30,
    # else it can be any positive integer. Default: 4.
    fact = 1
    # nside_skymap = hp.npix2nside(len(skymap))
    # pixValues = summed_df_mass['pixValue'].value
    pix_values = np.array(list(df_dict.keys()))
    mass_Values = np.array(list(df_dict.values()))
    tile_area = np.pi * radius.to(u.deg).value ** 2
    pixel_area = hp.nside2pixarea(nside_128, degrees=True)
    fov_probabilities = []
    max_prob_pix_arr = []
    max_prob_arr = []
    vecs = hp.ang2vec(ra, dec, lonlat=True)
    pixels = []
    disc_center = []
    for i in range(len(ra)):
        sel_pix = hp.query_disc(nside_128, vecs[i], radius.to(
            u.rad).value, inclusive=True, fact=fact)
        # pixels[i] = sel_pix
        pixels.append(sel_pix)
        disc_center.append(tuple((ra[i], dec[i])))
        sum_mass = 0
        max_prob_pix = sel_pix[0]
        max_prob = -99
        for j in sel_pix:
            if j in df_dict:
                # print(df_dict[j])
                sum_mass += df_dict[j]
                if df_dict[j] > max_prob:
                    max_prob_pix = j
                    max_prob = df_dict[j]
        fov_probabilities.append(sum_mass *
                                 tile_area / pixel_area / len(sel_pix))
        max_prob_pix_arr.append(max_prob_pix)
        if max_prob == -99:
            max_prob_arr.append(0)
        else:
            max_prob_arr.append(df_dict[max_prob_pix])
    return fov_probabilities, pixels, disc_center, max_prob_pix_arr, max_prob_arr


def get_ProbMass_Sorted(catMass, key):
    catMass['prob_mass'] = catMass['Mstar']*catMass[key]
    catMass.sort('prob_mass', reverse=True)
    cumSumProbMass = np.cumsum(catMass['prob_mass'])
    normCumSumProbMass = cumSumProbMass/np.max(cumSumProbMass)
    catMass['cum_prob_mass'] = normCumSumProbMass
    catMass['prob_mass'] = catMass['prob_mass']/np.max(cumSumProbMass)
    return catMass

# 'tiles_GIT_0.71_0.59.csv'


curpath = os.path.abspath('.')
eventPath = os.path.join(curpath, 'emgw_skymaps/processed_skymaps')
utilityPath = os.path.join(curpath, 'emgw_skymaps/config_tile_array')
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--distcut", help="If we need a distance cut")
parser.add_argument("--skymap", help="The input skymap")
parser.add_argument("--tiles", help="The Input Tiles file",
                    default='home/growth/emgw_skymaps/config_tile_array/tiles_GIT_0.71_0.59.csv')
args = parser.parse_args()
distcut = args.distcut
inj_file = args.skymap
tilings = args.tiles
# print(os.path.join(eventPath, inj_file))

# (prob, distmu, distsigma, distnorm), metadata = read_sky_map(os.path.join(areaPath, inj_file), distances=True)
if distcut == 'True':
    (prob, distmu, distsigma, distnorm), metadata = read_sky_map(
        os.path.join(eventPath, inj_file), distances=True)
else:
    prob = hp.read_map(inj_file)
# print((prob[0]))
npix = len(prob)
nside = hp.npix2nside(npix)
pixarea = hp.nside2pixarea(nside)
# print(pixarea)
prob_128 = hp.pixelfunc.ud_grade(prob, nside_out=1024)
# print(len(prob_128))
npix_128 = len(prob_128)
nside_128 = hp.npix2nside(npix_128)
# print(nside_128)
pixarea_128 = hp.nside2pixarea(nside_128, degrees=True)
# print(pixarea_128)
# ipix = hp.ang2pix(nside, cat['_RAJ2000'].value,
#                   cat['_DEJ2000'].value, lonlat=True)
ipix = hp.ang2pix(nside, cat['_RAJ2000'],
                  cat['_DEJ2000'], lonlat=True)
cat['pixValue'] = ipix
dp_dA = prob[ipix]/pixarea
cat['dP_dA'] = dp_dA
cat['dP_dA_norm'] = cat['dP_dA']/np.sum(cat['dP_dA'])
# cat90cut = getGalaxiesbyArea(cat)
if distcut == 'True':
    catFin, cat90cut, len_list = get_90_Loc_Vol(
        cat, prob, distmu, distsigma, distnorm)
else:
    cat90cut = getGalaxiesbyArea(cat)
cat90cut['Mstar'] = [0 if np.isnan(x) else x for x in cat90cut['Mstar']]
totMass = np.sum(cat90cut['Mstar'])
cat90cut['Mass_prob'] = cat90cut['Mstar']/totMass
cat90cut.sort('Mstar', reverse=True)
catMass = cat90cut[0:200]
# print(catMass[0:2])
catMass = getCatTileIds(catMass, 'Mass_prob')
catMass.sort('Mstar', reverse=True)
df_mass = catMass.to_pandas()
if distcut == 'True':
    catProbMass = get_ProbMass_Sorted(cat90cut, 'dP_dV')
    catProbMass = catProbMass[0:200]
    catProbMass = getCatTileIds(catProbMass, 'prob_mass')
    catProbMass.sort('prob_mass', reverse=True)
    df_prob_mass = catProbMass.to_pandas()
else:
    catProbMass = get_ProbMass_Sorted(cat90cut, 'dP_dA')
    catProbMass = getCatTileIds(catProbMass, 'prob_mass')
    catProbMass.sort('prob_mass', reverse=True)
    df_prob_mass = catProbMass.to_pandas()


df_needed_mass = df_mass[['pixValue', 'Mass_prob']]
df_needed_prob_mass = df_prob_mass[['pixValue', 'prob_mass']]

summed_df_mass = df_needed_mass.groupby(
    'pixValue')['Mass_prob'].sum().reset_index()

# Print the resulting DataFrame
summed_df_mass.to_csv('testGrouped.csv')
(raList, decList) = hp.pix2ang(nside_128, summed_df_mass['pixValue'])

summed_df_prob_mass = df_needed_prob_mass.groupby(
    'pixValue')['prob_mass'].sum().reset_index()

# Print the resulting DataFrame
summed_df_prob_mass.to_csv('testGrouped_prob_mass.csv')
(raList, decList) = hp.pix2ang(nside_128, summed_df_mass['pixValue'])

df_dict = dict(zip(summed_df_mass['pixValue'], summed_df_mass['Mass_prob']))
df_dict_prob_mass = dict(
    zip(summed_df_prob_mass['pixValue'], summed_df_prob_mass['prob_mass']))

# df_tiles = pd.read_csv(os.path.join(utilityPath, tilings))
df_tiles = pd.read_csv(
    '/home/growth/emgw_skymaps/config_tile_array/tiles_GIT_0.71_0.59.csv')
raList = df_tiles['RA_Center'].values
decList = df_tiles['DEC_Center'].values

prob_by_Mass, pixels, disc_center, max_prob_pix_arr, max_prob_arr = get_probabilities(
    raList, decList, df_dict, nside_128, radius=0.35*u.deg)

prob_by_Mass_prob_mass, pixels_prob_mass, disc_center_prob_mass, max_prob_pix_arr_prob_mass, max_prob_arr_prob_mass = get_probabilities(
    raList, decList, df_dict_prob_mass, nside_128, radius=0.35*u.deg)

(ra, dec) = hp.pix2ang(nside_128, max_prob_pix_arr, lonlat=True)
# ra_max, dec_max = hp.pix2ang(nside, ipix_max, lonlat=True)

ra_list = ra.tolist()
dec_list = dec.tolist()


list_of_tuples = list(zip(prob_by_Mass, disc_center,
                      ra_list, dec_list, max_prob_arr))
df_2 = pd.DataFrame(list_of_tuples, columns=[
                    'prob_by_Mass', 'disc_center', 'max_prob_pos_ra', 'max_prob_pos_dec', 'max_prob_arr'])

list_of_tuples = list(zip(prob_by_Mass_prob_mass, disc_center_prob_mass,
                      ra_list, dec_list, max_prob_arr_prob_mass))
df_2_prob_mass = pd.DataFrame(list_of_tuples, columns=[
                              'm*p', 'disc_center', 'max_prob_pos_ra', 'max_prob_pos_dec', 'max_m*p'])

# df.to_csv('final_GIT_FOV_Coverage.csv')
df_2.to_csv('final_GIT_FOV_Coverage2.csv')

df_2_prob_mass.to_csv('final_GIT_FOV_Coverage2_prob_mass.csv')
