import astropy
import astropy.coordinates as coo
from astropy.time import Time
from astropy.table import Table, Column
import astropy.units as u
import astroplan
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import healpy as hp
from matplotlib.backends.backend_pdf import PdfPages
from astroplan import Observer, FixedTarget
from astropy.coordinates import SkyCoord
import datetime
import argparse
import re
import os
from ligo.skymap.io import read_sky_map


def make_grid(nside=128):
    """
    Create a grid of tiles on the entire sky, and return RA, Dec
    Make it a table, so that later rise and set times can be included in the same thing.

    For GIT we select a healpix grid, with inter-pixel separation 0.46 degrees (nside = 128)
    """
    npix = hp.nside2npix(nside)
    pixels = np.arange(npix)
    ra, dec = hp.pix2ang(nside, pixels, lonlat=True)
    return ra * u.deg, dec * u.deg


def grid2targets(ra, dec):
    """
    Take RA and Dec (astropy quantities) and return a list of targets usable by astroplan
    """
    target_list = []
    for i in range(len(ra)):
        target_list.append(astroplan.FixedTarget(coord=coo.SkyCoord(ra[i], dec[i]),
                                                 name="tile_{}".format(i)))
    return target_list


def get_probabilities(skymap, ra, dec, radius=0.35*u.deg):
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
    nside_skymap = hp.npix2nside(len(skymap))
    tile_area = np.pi * radius.to(u.deg).value ** 2
    pixel_area = hp.nside2pixarea(nside_skymap, degrees=True)
    probabilities = np.zeros(len(ra))
    vecs = hp.ang2vec(ra.to(u.deg).value, dec.to(u.deg).value, lonlat=True)
    for i in range(len(ra)):
        sel_pix = hp.query_disc(nside_skymap, vecs[i], radius.to(
            u.rad).value, inclusive=True, fact=fact)
        probabilities[i] = np.sum(skymap[sel_pix]) * \
            tile_area / pixel_area / len(sel_pix)
    return probabilities


def get_obs_times(observatory=astroplan.Observer.at_site("iao"),
                  exptime=5*u.min, start_time=None, end_time=None):
    """
    Make a list of observing times
    """
    if start_time is None:
        start_time = observatory.twilight_evening_civil(Time.now(), 'next')
    if end_time is None:
        end_time = observatory.twilight_morning_civil(start_time, 'next')
    return np.arange(start_time, end_time, exptime)


def get_top_tiles(ra, dec, probabilities, frac=0.90):
    """
    probabilities may not add up to 1
    return indices of tiles that add up to frac of total
    """
    sortorder = np.argsort(probabilities)
    p_cum = np.cumsum(probabilities[sortorder]) / np.sum(probabilities)
    startind = np.where(p_cum > 1 - frac)[0][0]
    top_tiles = sortorder[startind:]
    return top_tiles


def get_rs_indices(toptiles, legalra, legaldec, obs_times, nside_skymap=128, radius=0.35*np.pi/180):
    obs_times_float = np.array([i.jd for i in obs_times])
    obs_times_object = Time(obs_times_float, format='jd')
    iao = Observer.at_site("iao")
    aa = iao.altaz(obs_times_object)
    topra, topdec = legalra[toptiles], legaldec[toptiles]
    tile_coords = SkyCoord(ra=topra, dec=topdec)
    rtimes = []
    stimes = []
    aa = iao.altaz(obs_times_object)
    for i in range(len(toptiles)):
        tile = toptiles[i]
        tra, tdec = legalra[tile], legaldec[tile]
        tile_coords = SkyCoord(ra=tra, dec=tdec)
        target_altaz = tile_coords.transform_to(aa)
        try:
            st = np.where(target_altaz.alt.value > 20)[0][-1]
            rt = np.where(target_altaz.alt.value > 20)[0][0]
            if (st == len(obs_times)-1):
                stimes.append(0)
                rtimes.append(rt)
            else:
                stimes.append(st)
                rtimes.append(len(obs_times))
        except:
            stimes.append(0)
            try:
                rtimes.append(np.where(target_altaz.alt.value > 20)[0][0])
            except:
                rtimes.append(len(obs_times))

    return np.clip(rtimes, 0, 125), np.clip(stimes, 0, 125)


'''
def get_sear_schedule(filename, tileno, legalra, legaldec, nside_skymap=128, radius=0.35*np.pi/180):
    skymap = hp.read_map(filename)
    probabilities = get_probabilities(skymap, legalra, legaldec)
    toptiles = get_top_tiles(legalra, legaldec, probabilities, frac=0.95)

    obs_times = get_obs_times()

    # iao = Observer.at_site("iao")
    R = np.zeros((len(legalra), len(obs_times)))
    S = np.zeros((len(legalra), len(obs_times)))
    # d = obs_times_float[1] - obs_times_float[0]

    topra = legalra[toptiles]
    topdec = legaldec[toptiles]

    rise_times, set_times = get_rs_indices(
        toptiles, legalra, legaldec, obs_times)

    for i in range(len(toptiles)):
        tile = toptiles[i]
        p = probabilities[tile]
        # print(i)
        S[tile][:set_times[i]] = p
        R[tile][rise_times[i]:] = p

    print('Done')

    obs_schedule = [0]*len(obs_times)
    set_schedule = [0]*len(obs_times)
    rise_schedule = [0]*len(obs_times)

    original_skymap = skymap

    for i in range(-1, -len(obs_times)-1, -1):
        ns = S[:, i].argmax()
        set_schedule[i] = ns
        vecs = hp.pix2vec(nside_skymap, ns)
        # sel_pix = hp.query_disc(nside_skymap, vecs, radius, inclusive=True, fact=1)
        S[ns, :] = 0

    for i in range(len(obs_times)):
        nr = R[:, i].argmax()
        rise_schedule[i] = nr
        vecs = hp.pix2vec(nside_skymap, nr)
        # sel_pix = hp.query_disc(nside_skymap, vecs, radius, inclusive=True, fact=1)
        R[nr, :] = 0

    for i in range(len(obs_times)):
        if probabilities[set_schedule[i]] >= probabilities[rise_schedule[i]]:
            obs_schedule[i] = set_schedule[i]
        else:
            obs_schedule[i] = rise_schedule[i]

    return np.array(obs_schedule), probabilities[obs_schedule]
'''


def get_sear_schedule(filename, tileno, legalra, legaldec, nside_skymap=128, radius=0.35*np.pi/180):
    skym = read_sky_map(filename, moc=False)
    skymap = skym[0]
    probabilities = get_probabilities(skymap, legalra, legaldec)
    toptiles = get_top_tiles(legalra, legaldec, probabilities, frac=0.95)
    obs_times = get_obs_times()
    iao = Observer.at_site("iao")
    if Time.now() > iao.twilight_evening_civil(Time.now(), 'nearest'):
        obs_times = get_obs_times(start_time=Time.now())
    iao = Observer.at_site("iao")
    R = np.zeros((len(legalra), len(obs_times)))
    S = np.zeros((len(legalra), len(obs_times)))
    # d = obs_times_float[1] - obs_times_float[0]
    topra = legalra[toptiles]
    topdec = legaldec[toptiles]
    comesup = np.where(topdec > -33*u.deg)
    topra = topra[comesup]
    topdec = topdec[comesup]
    rise_times, set_times = get_rs_indices(
        toptiles, legalra, legaldec, obs_times)
    for i in range(len(toptiles)):
        tile = toptiles[i]
        p = probabilities[tile]
        # print(i)
        S[tile][:set_times[i]] = p
        R[tile][rise_times[i]:] = p
    print('Done')
    obs_schedule = [0]*len(obs_times)
    set_schedule = [0]*len(obs_times)
    rise_schedule = [0]*len(obs_times)
    original_skymap = skymap
    for i in range(-1, -len(obs_times)-1, -1):
        ns = S[:, i].argmax()
        set_schedule[i] = ns
        vecs = hp.pix2vec(nside_skymap, ns)
        # sel_pix = hp.query_disc(nside_skymap, vecs, radius, inclusive=True, fact=1)
        S[ns, :] = 0
    for i in range(len(obs_times)):
        nr = R[:, i].argmax()
        rise_schedule[i] = nr
        vecs = hp.pix2vec(nside_skymap, nr)
        # sel_pix = hp.query_disc(nside_skymap, vecs, radius, inclusive=True, fact=1)
        R[nr, :] = 0
    set_schedule = np.array(set_schedule)

    riseset = []
    for i in range(len(obs_times)):
        if probabilities[set_schedule[i]] >= probabilities[rise_schedule[i]]:
            obs_schedule[i] = set_schedule[i]
            riseset.append(0)
        else:
            riseset.append(1)
            obs_schedule[i] = rise_schedule[i]
    time = obs_times[0] - 2*u.hour
    iao = Observer.at_site("iao")
    final_targets = SkyCoord(
        ra=legalra[obs_schedule], dec=legaldec[obs_schedule])
    return np.array(obs_schedule), probabilities[obs_schedule], np.array(riseset)


def get_enar_schedule(filename, tileno, legalra, legaldec, nside_skymap=128, radius=0.35*np.pi/180):
    skymap = hp.read_map(filename)
    probabilities = get_probabilities(skymap, legalra, legaldec)
    toptiles = get_top_tiles(legalra, legaldec, probabilities, frac=0.95)

    obs_times = get_obs_times()

    iao = Observer.at_site("iao")
    R = np.zeros((len(legalra), len(obs_times)))
    S = np.zeros((len(legalra), len(obs_times)))
    # d = obs_times_float[1] - obs_times_float[0]

    topra = legalra[toptiles]
    topdec = legaldec[toptiles]

    rise_times, set_times = get_rs_indices(
        toptiles, legalra, legaldec, obs_times)

    for i in range(len(toptiles)):
        tile = toptiles[i]
        p = probabilities[tile]
        # print(i)
        S[tile][:set_times[i]] = p
        R[tile][rise_times[i]:] = p

    print('Done')

    obs_schedule = [0]*len(obs_times)
    set_schedule = [0]*len(obs_times)
    rise_schedule = [0]*len(obs_times)

    original_skymap = skymap

    for i in range(-1, -len(obs_times)-1, -1):
        ns = S[:, i].argmax()
        set_schedule[i] = ns
        vecs = hp.pix2vec(nside_skymap, ns)
        # sel_pix = hp.query_disc(nside_skymap, vecs, radius, inclusive=True, fact=1)
        S[ns, :] = 0

    for i in range(len(obs_times)):
        nr = R[:, i].argmax()
        rise_schedule[i] = nr
        vecs = hp.pix2vec(nside_skymap, nr)
        # sel_pix = hp.query_disc(nside_skymap, vecs, radius, inclusive=True, fact=1)
        R[nr, :] = 0

    set_schedule = np.array(set_schedule)
    prob_sorted_tiles = set_schedule[np.argsort(probabilities[set_schedule])]
    finalized_slots = []

    for i in range(1, len(prob_sorted_tiles)):
        # print(np.where(toptiles == prob_sorted_tiles[i])[0])
        last_seen = np.array(set_times)[np.where(
            set_schedule == prob_sorted_tiles[i])[0][0]]

        while last_seen in finalized_slots and last_seen > 0:
            last_seen -= 1

        if last_seen not in finalized_slots:
            current_ind = np.where(set_schedule == prob_sorted_tiles[1])[0][0]
            set_schedule[last_seen], set_schedule[current_ind] = set_schedule[current_ind], set_schedule[last_seen]
            finalized_slots.append(last_seen)

    for i in range(len(obs_times)):
        if probabilities[set_schedule[i]] >= probabilities[rise_schedule[i]]:
            obs_schedule[i] = set_schedule[i]
        else:
            obs_schedule[i] = rise_schedule[i]

    return np.array(obs_schedule)[1:], probabilities[obs_schedule][1:]


def save_sear_table(infile, config_tile_file, outfile):
    """
    Filename should include extension.
    """
    # sum_prob = 0
    obs_times = get_obs_times()
    f = np.loadtxt(config_tile_file, delimiter=',', skiprows=1)
    tileno = f[:, 1]
    lra = f[:, 2]*u.deg
    ldec = f[:, 3]*u.deg
    sear = get_sear_schedule(infile, tileno, lra, ldec)

    times, tiles, ra, dec, probabilities, riseset = obs_times, sear[0], lra[sear[0]
                                                                            ], ldec[sear[0]], sear[1], sear[2]

    final_table = Table(names=("mjdobs", "obstime", "ra", "dec", "prob", "tile_id"),
                        dtype=(np.float32, 'S23', np.float32, np.float32, np.float32, np.float32))
    mid = np.where(riseset == 0)[0][-1]
    tot = len(tiles)*(len(tiles)+1)
    for count in range(len(tiles)):
        this_tile = tiles[count]
        this_time = times[count]
        if riseset[count] == 0:
            this_prob = (len(tiles)-count)/tot

        else:
            this_prob = (count-mid)/tot
        # sum_prob += this_prob
        final_table.add_row((this_time.mjd, this_time.isot,
                             ra[count], dec[count],
                             this_prob, this_tile))
    # final_table.sort('mjdobs')
    # final_table.add_column(
    #    Column(data=np.arange(len(final_table)), name='num'), 0)

    final_table.write(outfile, overwrite=True)
    print('Probability covered = ',np.sum(probabilities))


def save_enar_table(infile, config_tile_file, outfile):
    """
    Filename should include extension.
    """
    # sum_prob = 0

    obs_times = get_obs_times()
    f = np.loadtxt(config_tile_file, delimiter=',', skiprows=1)
    tileno = f[:, 1]
    lra = f[:, 2]*u.deg
    ldec = f[:, 3]*u.deg
    enar = get_enar_schedule(infile, tileno, lra, ldec)

    times, tiles, ra, dec, probabilities = obs_times, enar[0], lra[enar[0]
                                                                   ], ldec[enar[0]], enar[1]

    final_table = Table(names=("mjdobs", "obstime", "ra", "dec", "prob", "tile_id"),
                        dtype=(np.float32, 'S23', np.float32, np.float32, np.float32, np.float32))
    for count in range(len(tiles)):
        this_tile = tiles[count]
        this_time = times[count]
        this_prob = probabilities[count]
        # sum_prob += this_prob
        final_table.add_row((this_time.mjd, this_time.isot,
                             ra[count], dec[count],
                             this_prob, this_tile))
    # final_table.sort('mjdobs')
    # final_table.add_column(
    #    Column(data=np.arange(len(final_table)), name='num'), 0)

    final_table.write(outfile, overwrite=True)
    print('Probability covered = ',np.sum(probabilities))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-input_file", help="Input file with Directory", default=None)
    parser.add_argument(
        "-ouput_dir", help="To store Output scheduled fits file with Directory", default='/home/growth/emgw_skymaps/')
    parser.add_argument(
        "-config_tile_file", help="path+file of tile array for GIT", default='/home/growth/emgw_skymaps/config_tile_array/tiles_GIT7.csv')
    parser.add_argument(
        "-type", help="scheduling type : put s-setting_array or e-enhanced_array", default='s')
    args = parser.parse_args()

    file = args.input_file
    output_name = file + "_scheduled.csv"
    print(output_name)
    if args.type == 's':
        save_sear_table(file, args.config_tile_file, outfile=output_name)
    else:
        save_enar_table(file, args.config_tile_file, outfile=output_name)
