from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import ligo.skymap.io
from astropy.io import fits
import seaborn as sns
from astropy.table import Table, vstack, hstack, Column
import astropy.units as u
from astropy.coordinates import SkyCoord
import ligo.skymap.plot
from scipy.stats import norm
import pandas as pd
import os
#import ligo.skymap.io
from ligo.skymap.io import read_sky_map
from ligo.skymap.plot import marker
from ligo.skymap import io
from ligo.skymap.distance import (parameters_to_marginal_moments, principal_axes, volume_render, marginal_pdf)
from matplotlib import gridspec
from matplotlib import transforms
from ligo.skymap.tool.matplotlib import figure_parser
from ligo.skymap.tool import ArgumentParser, FileType
from tqdm import tqdm
import ligo.skymap.tool
import argparse


with fits.open('GIT_Observable_catalog.fits') as hdu:
     nedTable = Table(hdu[1].data)
cat = nedTable['objname', '_RAJ2000', '_DEJ2000', 'Dist', 'Mstar', 'SFR_W4', 'SFR_hybrid']
# cat.rename_column('DistMpc', 'Dist')
# cat.rename_column('ra', '_RAJ2000')
# cat.rename_column('dec', '_DEJ2000')


# with fits.open('NEDGWF_D1000_Sept2021_BetaV3.fits') as hdu:
#     nedTable = Table(hdu[1].data)
# cat = nedTable['objname', 'ra', 'dec', 'DistMpc', 'Mstar', 'SFR_W4', 'SFR_hybrid']
# cat.rename_column('DistMpc', 'Dist')
# cat.rename_column('ra', '_RAJ2000')
# cat.rename_column('dec', '_DEJ2000')


def mark_event_center(prob, nside, filename, savename, plot_needed):
    ipix_max = np.argmax(prob)
    ra_max, dec_max = hp.pix2ang(nside, ipix_max, lonlat=True)
    #print(dec_max*u.deg)
    center = SkyCoord(ra=ra_max*u.deg,dec=dec_max*u.deg)
    #print('Coordinates (RA,Dec) = %s' %(center))
    if plot_needed=='yes':
        ax = plt.axes(
        [0.05, 0.05, 0.9, 0.9],
        projection='astro globe',
        center=center)

        ax.grid()
        #print(filename)
        #print(nside)
        #ax.imshow_hpx(filename, cmap='cylon')
        ax.plot(
            center.ra.deg, center.dec.deg,
            transform=ax.get_transform('world'),
            marker=ligo.skymap.plot.reticle(inner=0,outer=1),
            markersize=10,
            markeredgewidth=2)
        fig = plt.gcf()
        plt.savefig('event-center'+savename+'.png')
        plt.clf()

    return center


def get_90_Loc_Area(ipix, prob, pixarea, distmu, distsigma, distnorm, center, filename, catNew):
    dp_dA=prob[ipix]/pixarea
    catNew['dP_dA']=dp_dA
    
    #probability along radial distance
    dp_dr=cat['Dist'].value**2 * distnorm[ipix] * norm(distmu[ipix],distsigma[ipix]).pdf(cat['Dist'].value)
    catNew['dp_dr'] = dp_dr

    catNew.sort('dP_dA', reverse=True)
    cum_sort=np.cumsum(cat['dP_dA'])
    cumnorm_sort=cum_sort/np.max(cum_sort)
    catNew['P_A']=cumnorm_sort

    icutarea90=np.where(catNew['P_A'] <= 0.9)
    clucutarea90=cat[icutarea90]


    print('# of galaxies in 90%% Area = %i' %(np.size(icutarea90)))

    
    #icutarea90=np.where(catNew['P_A'] <= 0.01)
    #clucutarea90=cat[icutarea90]

    return clucutarea90, np.size(icutarea90)

def plotArea(catNew, clucutarea90, center, prob, savename, numArea):
    #ID galaxies inside the 90% prob by volume    

    #generate astropy coordinate object for this sample
    clucutarea90coord=SkyCoord(ra=np.radians(clucutarea90['_RAJ2000'])*u.rad,dec=np.radians(clucutarea90['_DEJ2000'])*u.rad)

    #sort the galaxies by P-value and print out top 20
    clucutarea90['_RAJ2000','_DEJ2000','dP_dA','P_A'][0:20].pprint(max_lines=22)

    # Plot galaxies in 90% Area 
    ax = plt.axes(
    [0.05, 0.05, 0.9, 0.9],
    projection='astro globe',
    center=center)
    plt.text(0.9, 0.5, "fTotal galaxies in 1%% localization area is:{numArea}", fontsize=25)

#Zoomed-in inset window to better view the locations of the galaxies.
    ax_inset = plt.axes(
        [0.59, 0.3, 0.4, 0.4],
        projection='astro zoom',
        center=center,
        radius=15*u.deg)
    for key in ['ra', 'dec']:
        ax_inset.coords[key].set_ticklabel_visible(False)
        ax_inset.coords[key].set_ticks_visible(False)
    ax.grid()
    ax.mark_inset_axes(ax_inset)
    ax.connect_inset_axes(ax_inset, 'upper left')
    ax.connect_inset_axes(ax_inset, 'lower left')
    ax_inset.scalebar((0.1, 0.1), 5 * u.deg).label()
    ax_inset.compass(0.9, 0.1, 0.2)
    
    # (prob, distmu, distsigma, distnorm), metadata = read_sky_map(filename, distances=True)
    # npix = len(prob)
    # print(npix)
    # nside = hp.npix2nside(npix)
    # print("sadasdasdasdasdasd",nside)


    #print(read_sky_map(filename))
    ax.imshow_hpx(prob, cmap='cylon')
    ax_inset.imshow_hpx(prob, cmap='cylon')

    for coord in clucutarea90coord:
        ax_inset.plot(
        coord.ra.deg, coord.dec.deg,
        transform=ax_inset.get_transform('world'),
        marker=ligo.skymap.plot.reticle(inner=0,outer=1),
        markersize=10,
        markeredgewidth=1)
    plt.title("Top 1%% localization by Area", fontsize = 25)
    plt.savefig('savedata/area'+savename+'.png')
    plt.clf()
    

def getCumMass(catMass, needed_mass_frac):
    catMass.sort('Mstar', reverse=True)
    cumsumMass = np.cumsum(catMass['Mstar'])
    normCumSum = cumsumMass/np.max(cumsumMass)
    catMass['cumMass'] = normCumSum
    m_frac1 = np.argmax(catMass['cumMass'] >= needed_mass_frac[0])
    #m_thirty = np.argmax(catMass['cumMass'] >= 0.3)
    m_frac2 = np.argmax(catMass['cumMass'] >= needed_mass_frac[1])
    #print("No of galaxies:", m_fifty)
    #print(catMass[0:10])
    return catMass, m_frac1, m_frac2

#Only calculate a cummulative Probabilty of top n galaxies
#different from P_N as there we calculated Cummulative Prob of all the galaxies in > 90% localization region
def getCumProb(catMass):
    catMass.sort('dp_dV', reverse = True)
    cumsumProb = np.cumsum(catMass['dP_dV'])
    normCumSumProb = cumsumProb/np.max(cumsumProb)
    catMass['cumProb'] = normCumSumProb
    return catMass

#calculate the parameter prob*mass and get the table sorted by this parameter 
def getProbMassSorted(catMass, key):
    catMass['prob_mass'] = catMass['Mstar']*catMass[key]
    catMass.sort('prob_mass', reverse = True)
    cumSumProbMass = np.cumsum(catMass['prob_mass'])
    normCumSumProbMass = cumSumProbMass/np.max(cumSumProbMass)
    catMass['cum_prob_mass'] = normCumSumProbMass
    return catMass

def plotCumMass(catMass, savename, key):
    fig,ax = plt.subplots(figsize=(9,9))
    x = np.linspace(1,len(catMass),len(catMass))
    ax.plot(x, catMass['cumMass'], label='mass fraction', color ='blue', alpha=0.4, linewidth=3)
    y_fifty = 0.5
    #x_fifty = catMass['cumMass'][np.argmin(catMass['cumMass'] >= y_fifty)]
    x_fifty = np.interp(y_fifty, catMass['cumMass'], x)
    ax.text(int(x_fifty), y_fifty, f'{int(x_fifty)}', ha='center', va='bottom', fontsize=10)
    ax.fill_between(x, catMass['cumMass'], where=catMass['cumMass'] < 0.5, color='blue', alpha=0.2)
    ax.set_xlabel('No. of Galaxies', fontsize=14)
    ax.set_ylabel('Fraction of Mass Covered', fontsize=14)
    catMass = getProbMassSorted(catMass, key)
    ax.plot(x, catMass['cum_prob_mass'], label='prob mass fraction', color ='red', alpha=0.4, linewidth=3)
    #x_fifty = catMass['cumMass'][np.argmin(catMass['cumMass'] >= y_fifty)]
    x_fifty_prob_mass = np.interp(y_fifty, catMass['cum_prob_mass'], x)
    ax.text(int(x_fifty_prob_mass), y_fifty, f'{int(x_fifty_prob_mass)}', ha='center', va='bottom', fontsize=10)
    ax.fill_between(x, catMass['cum_prob_mass'], where=catMass['cum_prob_mass'] < 0.5, color='red', alpha=0.2)
    ax.legend(loc = 'upper left', fontsize=12)
    plt.savefig('savedata/cumMass'+savename+'.png')
    plt.clf()
    return x_fifty 

def plotCumProb(catMass, savename):
    fig,ax = plt.subplots(figsize=(9,9))
    x = np.linspace(1,len(catMass),len(catMass))
    #ax.plot(x, catMass['cumMass'])
    a, = ax.plot(x, catMass['cumProb'], label='cumm probability', color ='blue', alpha=0.4)
    catMass = getProbMassSorted(catMass)
    axTwin = ax.twinx()
    b, = axTwin.plot(x, catMass['cum_prob_mass'], label='cumm prob weighted mass', color='red', alpha=0.4)
    #axTwin.set_xlabel('No. of Galaxies')
    ax.set_xlabel('No. of Galaxies covered', fontsize=12)
    ax.set_ylabel("Cummulative Probability Covered", fontsize=12)
    axTwin.set_ylabel('Cummulative p_dV*m Covesred')
    p = [a,b]
    ax.legend(p, [p_.get_label() for p_ in p],
           loc= 'upper left', fontsize= '12')
    #labels = [l.get_label() for l in p]
    #ax.legend(p, labels, loc=0)
    #ax.legend(loc = 'upper left', fontsize=8)
    plt.savefig('savedata/cumMass'+savename+'.png')
    plt.clf()

def get_90_Loc_Vol(catFin, prob, distmu, distsigma, distnorm, filename):
    #def getGalaxiesIn90Vol(catFin, prob):
    #load in CLU catalog
    clucoord=SkyCoord(ra=catFin['_RAJ2000']*u.deg,dec=catFin['_DEJ2000']*u.deg)
    nclu=np.size(catFin)

    #make astropy coordinate object of CLU galaxies
    #clucoord=SkyCoord(ra=catFin['_RAJ2000']*u.deg,dec=catFin['_DEJ2000']*u.deg)
    #print("SkyCoord object length", len(clucoord))

    #load in healpix map
    #prob,distmu,distsigma,distnorm=hp.read_map('data/GW170817_prelim.fits.gz',field=[0,1,2,3],dtype=('f8','f8','f8','f8'))
    npix=len(prob)
    nside=hp.npix2nside(npix)
    pixarea=hp.nside2pixarea(nside)

    #get coord of max prob density by area for plotting purposes
    ipix_max = np.argmax(prob)
    #print("Max prob pix", ipix_max)
    theta_max, phi_max = hp.pix2ang(nside, ipix_max)
    ra_max = np.rad2deg(phi_max)
    dec_max = np.rad2deg(0.5 * np.pi - theta_max)
    center = SkyCoord(ra=ra_max*u.deg,dec=dec_max*u.deg)
    #print(center)

    #calc hp index for each galaxy and populate CLU Table with the values
    theta=0.5 * np.pi - clucoord.dec.to('rad').value
    phi=clucoord.ra.to('rad').value
    #print("phi", len(phi))
    ipix=hp.ang2pix(nside,theta,phi) # ipix = hp.ang2pix(nside, clucoord.dec, clucoord.ra, lonlat=True)
    #print(type(ipix))
    #print("ipix", len(ipix))
    # print("prob ipix", prob[ipix])
    # print("prob*distnorm",prob[ipix] * distnorm[ipix])
    # print("cat dist",np.array(catFin['Dist'].tolist()))
    # print("pdf", norm(distmu[ipix],distsigma[ipix]).pdf(np.array(catFin['Dist'].tolist())))
    #calc probability density per volume for each galaxy
    dp_dV=prob[ipix] * distnorm[ipix] * norm(distmu[ipix],distsigma[ipix]).pdf(np.array(catFin['Dist'].tolist()))/pixarea
    #print("dp_dV len", dp_dV) 
    catFin['dP_dV']=dp_dV

    #use normalized cumulative dist function to calculate Volume P-value for each galaxy
    catFin.sort('dP_dV', reverse=True)
    #clu.reverse()
    cum_sort=np.cumsum(catFin['dP_dV'])
    cumnorm_sort=cum_sort/np.max(cum_sort)
    catFin['P_N']=cumnorm_sort

    #ID galaxies inside the 90% prob by volume
    icut90=np.where(catFin['P_N'] <= 0.9)
    clucut90=catFin[icut90]

    #generate an astropy coordinate object for this subset
    #clucut90coord=SkyCoord(ra=np.radians(clucut90['_RAJ2000'])*u.rad,dec=np.radians(clucut90['_DEJ2000'])*u.rad)
    number = len(clucut90)

    #print('# of galaxies in 90%% volume = %i' %(np.size(clucut90)))
    
    return catFin, clucut90, np.size(clucut90)

def get_90_Loc_Vol_Mass(catFin, prob, distmu, distsigma, distnorm, filename):
    #def getGalaxiesIn90Vol(catFin, prob):
    #load in CLU catalog
    clucoord=SkyCoord(ra=catFin['_RAJ2000']*u.deg,dec=catFin['_DEJ2000']*u.deg)
    nclu=np.size(catFin)

    #make astropy coordinate object of CLU galaxies
    #clucoord=SkyCoord(ra=catFin['_RAJ2000']*u.deg,dec=catFin['_DEJ2000']*u.deg)
    #print("SkyCoord object length", len(clucoord))

    #load in healpix map
    #prob,distmu,distsigma,distnorm=hp.read_map('data/GW170817_prelim.fits.gz',field=[0,1,2,3],dtype=('f8','f8','f8','f8'))
    npix=len(prob)
    nside=hp.npix2nside(npix)
    pixarea=hp.nside2pixarea(nside)

    #get coord of max prob density by area for plotting purposes
    ipix_max = np.argmax(prob)
    #print("Max prob pix", ipix_max)
    theta_max, phi_max = hp.pix2ang(nside, ipix_max)
    ra_max = np.rad2deg(phi_max)
    dec_max = np.rad2deg(0.5 * np.pi - theta_max)
    center = SkyCoord(ra=ra_max*u.deg,dec=dec_max*u.deg)
    #print(center)

    #calc hp index for each galaxy and populate CLU Table with the values
    theta=0.5 * np.pi - clucoord.dec.to('rad').value
    phi=clucoord.ra.to('rad').value
    #print("phi", len(phi))
    ipix=hp.ang2pix(nside,theta,phi) # ipix = hp.ang2pix(nside, clucoord.dec, clucoord.ra, lonlat=True)
    #print(type(ipix))
    #print("ipix", len(ipix))
    # print("prob ipix", prob[ipix])
    # print("prob*distnorm",prob[ipix] * distnorm[ipix])
    # print("cat dist",np.array(catFin['Dist'].tolist()))
    # print("pdf", norm(distmu[ipix],distsigma[ipix]).pdf(np.array(catFin['Dist'].tolist())))
    #calc probability density per volume for each galaxy
    dp_dV=prob[ipix] * distnorm[ipix] * norm(distmu[ipix],distsigma[ipix]).pdf(np.array(catFin['Dist'].tolist()))/pixarea
    #print("dp_dV len", dp_dV) 
    catFin['dP_dV']=dp_dV

    #use normalized cumulative dist function to calculate Volume P-value for each galaxy
    catFin.sort('dP_dV', reverse=True)
    #clu.reverse()
    cum_sort=np.cumsum(catFin['dP_dV'])
    cumnorm_sort=cum_sort/np.max(cum_sort)
    catFin['P_N']=cumnorm_sort

    #ID galaxies inside the 90% prob by volume
    icut90=np.where(catFin['P_N'] <= 0.9)
    clucut90=catFin[icut90]

    #generate an astropy coordinate object for this subset
    #clucut90coord=SkyCoord(ra=np.radians(clucut90['_RAJ2000'])*u.rad,dec=np.radians(clucut90['_DEJ2000'])*u.rad)
    number = len(clucut90)

    #print('# of galaxies in 90%% volume = %i' %(np.size(clucut90)))
    
    return catFin, clucut90, np.size(clucut90)


def plotVol(clucut90, center, prob, savename, numVol, galToShow):
    
    catLast = clucut90[0:100]
    catLast.sort('Mstar',reverse=True)
    #catLast = clucut90[0:galToShow]
    #print(len(catLast))
    catCord=SkyCoord(ra=np.radians(catLast['_RAJ2000'])*u.rad,dec=np.radians(catLast['_DEJ2000'])*u.rad)
    
    #Plot the galaxies in 90% localization region by Volume
    ax = plt.axes(
    [0.05, 0.05, 0.9, 0.9],
    projection='astro globe',
    center=center)
    #plt.text(0.9, 0.5, "(Only shown upto"+str(galToShow)+") Total galaxies within 90% localization volume is: "+str(numVol), fontsize=25)

    # ax_inset = plt.axes(
    #     [0.59, 0.3, 0.4, 0.4],
    #     projection='astro zoom',
    #     center=center,
    #     radius=100*u.deg)

    for key in ['ra', 'dec']:
        #ax_inset.coords[key].set_ticklabel_visible(False)
        #ax_inset.coords[key].set_ticks_visible(False)
        ax.coords[key].set_ticklabel_visible(False)
        ax.coords[key].set_ticks_visible(False)
    ax.grid()
    #ax.mark_inset_axes(ax_inset)
    #ax.connect_inset_axes(ax_inset, 'upper left')
    #ax.connect_inset_axes(ax_inset, 'lower left')
    #ax_inset.scalebar((0.1, 0.1), 5 * u.deg).label()
    #ax_inset.compass(0.9, 0.1, 0.2)

    ax.imshow_hpx(prob, cmap='cylon')
    ax.grid()
    #ax_inset.imshow_hpx(prob, cmap='cylon')
    for coord in catCord:
        ax.plot(
        coord.ra.deg, coord.dec.deg,
        transform=ax.get_transform('world'),
        marker=ligo.skymap.plot.reticle(inner=0,outer=1),
        markersize=10,
        markeredgewidth=1)
        #c4993=SkyCoord.from_name('NGC 4993')
        #ax.text(c4993.ra.deg+10.5, c4993.dec.deg,'NGC 4993',transform=ax_inset.get_transform('world'),fontdict={'size':10,'color':'black','weight':'normal'})
        # ax_inset.plot(
        # coord.ra.deg, coord.dec.deg,
        # transform=ax_inset.get_transform('world'),
        # marker=ligo.skymap.plot.reticle(inner=0,outer=1),
        # markersize=10,
        # markeredgewidth=1)
        # c4993=SkyCoord.from_name('NGC 4993')
        # ax_inset.text(c4993.ra.deg+10.5, c4993.dec.deg,'NGC 4993',transform=ax_inset.get_transform('world'),fontdict={'size':10,'color':'black','weight':'normal'})
    #where is NGC4993? hint: use ax_inset.text()
    #c4993=SkyCoord.from_name('NGC 4993')
    #ax_inset.text(c4993.ra.deg+10.5, c4993.dec.deg,'NGC 4993',transform=ax_inset.get_transform('world'),fontdict={'size':10,'color':'black','weight':'normal'})

    #plt.show()
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=True,         # ticks along the top edge are off
    labelbottom=True)
    #plt.title("Some galaxies in the localisation volume", fontsize = 25)
    plt.savefig('savedata/volume'+savename+'.png')
    plt.clf()

    return catLast

def plot_Projections(nside, prob, mu, sigma, norm, catLast, metadata, filename, savename, max_dp_dr):
    #min_distance = np.min(catLast['Dist'])
    #max_distance = np.max(catLast['Dist'])
    max_distance = None
    projection = 0
    figure_width = 3.5
    #contour = 90
    radecdist = []
    annotate =False
    neededcols = np.array(catLast['_RAJ2000','_DEJ2000','Dist'])
    radecdist = tuple(map(tuple, neededcols))
    progress = tqdm()
    progress.set_description('Starting up')
    progress.set_description('Loading FITS file')
    npix = len(prob)
    #print(radecdist)
    #nside = ah.npix_to_nside(npix)
    
    progress.set_description('Preparing projection')

    prob2, mu2, sigma2, norm2 = prob, mu, sigma, norm

    if max_distance is None:
        mean, std = parameters_to_marginal_moments(prob2, mu2, sigma2)
        #print(mean.shape)
        max_distance = mean + 2.5 * std
    else:
        max_distance = max_distance
    rot = np.ascontiguousarray(principal_axes(prob2, mu2, sigma2))

    # if opts.chain:
    #     chain = io.read_samples(opts.chain.name)
    #     chain = np.dot(rot.T, (hp.ang2vec(
    #         0.5 * np.pi - chain['dec'], chain['ra']) *
    #         np.atleast_2d(chain['dist']).T).T)

    fig = plt.figure(frameon=False, figsize=(10,10))
    n = 1 if projection else 2
    gs = gridspec.GridSpec(
        n, n, left=0.01, right=0.99, bottom=0.01, top=0.99,
        wspace=0.05, hspace=0.05)

    imgwidth = int(100 * figure_width / n)
    s = np.linspace(-max_distance, max_distance, imgwidth)
    xx, yy = np.meshgrid(s, s)

    truth_marker = marker.reticle( 
        inner=0.5 * np.sqrt(2), outer=1.5 * np.sqrt(2), angle=45)

    for iface, (axis0, axis1, (sp0, sp1)) in enumerate((
            (1, 0, [0, 0]),
            (0, 2, [1, 1]),
            (1, 2, [1, 0]),)):

        if projection and projection != iface + 1:
            continue

        progress.set_description('Plotting projection {0}'.format(iface + 1))

        # Marginalize onto the given face
        density = volume_render(
            xx.ravel(), yy.ravel(), max_distance, axis0, axis1, rot, False,
            prob, mu, sigma, norm).reshape(xx.shape)
        
        #print(density.shape)
        #print(type(density))

        # Plot heat map
        ax = fig.add_subplot(
            gs[0, 0] if projection else gs[sp0, sp1], aspect=1)
        ax.imshow(
            density, origin='lower',
            extent=[-max_distance, max_distance, -max_distance, max_distance])

        # Add contours if requested
        # if contour:
        #     flattened_density = density.ravel()
        #     print(flattened_density)
        #     indices = np.argsort(flattened_density)[::-1]
        #     print(len(indices))
        #     cumsum = np.empty_like(flattened_density)
        #     cs = np.cumsum(flattened_density[indices])
        #     cumsum[indices] = cs / cs[-1] * 100
        #     cumsum = np.reshape(cumsum, density.shape)
        #     u, v = np.meshgrid(s, s)
        #     contourset = ax.contour(
        #         u, v, cumsum, levels=contour, linewidths=0.5)

        # Mark locations
        #distlist = []
        ax._get_lines.get_next_color()  # skip default color
        #print("$$$$$$$$$$$$$$$$Here's your length$$$$$$$$$$$$$")
        #print(len(radecdist))
        for ra, dec, dist in radecdist:
            theta = 0.5 * np.pi - np.deg2rad(dec)
            phi = np.deg2rad(ra)
            xyz = np.dot(rot.T, hp.ang2vec(theta, phi) * dist)
            #distlist.append(dist)
            ax.plot(
                xyz[axis0], xyz[axis1], marker=truth_marker,
                markerfacecolor='none', markeredgewidth=1)
            


        ax.set_xticks([])
        ax.set_yticks([])

        # Set axis limits
        ax.set_xlim([-max_distance, max_distance])
        ax.set_ylim([-max_distance, max_distance])

        # Mark origin (Earth)
        progress.set_description('Marking Earth')
        ax.plot(
            [0], [0], marker=marker.earth, markersize=5,
            markerfacecolor='none', markeredgecolor='black',
            markeredgewidth=0.75)

        if iface == 2:
            ax.invert_xaxis()

    if not projection:
        # Add scale bar, 1/4 width of the plot
        ax.plot(
            [0.0625, 0.3125], [0.0625, 0.0625],
            color='black', linewidth=1, transform=ax.transAxes)
        ax.text(
            0.0625, 0.0625,
            '{0:d} Mpc'.format(int(np.round(0.5 * max_distance))),
            fontsize=8, transform=ax.transAxes, verticalalignment='bottom')

        # Create marginal distance plot.
        progress.set_description('Plotting distance')
        gs1 = gridspec.GridSpecFromSubplotSpec(5, 5, gs[0, 1])
        ax = fig.add_subplot(gs1[1:-1, 1:-1])

        # Plot marginal distance distribution, integrated over the whole sky.
        d = np.linspace(0, max_distance)
        #print("LOOK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(np.sum(marginal_pdf(d, prob, mu, sigma, norm)))
        ax.fill_between(d, marginal_pdf(d, prob, mu, sigma, norm)/1e-6,
                        alpha=0.5, color=ax._get_lines.get_next_color())
        
        # ax.hist(dist, 10, density = 0, 
        #                     color ='green', 
        #                     alpha = 0.7)
        #ax.vlines(x=np.array(catLast['Dist']), ymin=0, ymax=ax.get_ylim())

        # Scale axes
        ax.set_xticks([0, max_distance])
        ax.set_xticklabels(
            ['0', "{0:d}\nMpc".format(int(np.round(max_distance)))],
            fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(0, max_distance)
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_title("Marginal Distance")

        if annotate:
            text = []
            try:
                objid = metadata['objid']
            except KeyError:
                pass
            else:
                text.append('event ID: {}'.format(objid))
            try:
                distmean = metadata['distmean']
                diststd = metadata['diststd']
            except KeyError:
                pass
            else:
                text.append('distance: {}±{} Mpc'.format(
                            int(np.round(distmean)), int(np.round(diststd))))
                #text.append('distance:', int(np.round(distmean)))
                
            ax.text(0, 1, '\n'.join(text), transform=ax.transAxes, fontsize=7,
                    ha='left', va='bottom', clip_on=False)
    # plt.tick_params(
    # axis='x',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    # labelbottom=False)        
    #plt.title("Projection")
    #plt.show()
    #plt.gcf()
    plt.savefig('savedata/projection'+savename+'.png')
    plt.clf()
        
    progress.set_description('Saving')
    return fig

def plotMassHist(catMass, savename):
    massVal = [(mass/1e10) for mass in catMass['Mstar'] if mass !=0]
    #x_m = np.linspace(np.min(massVal)/1e10, np.max(massVal)/1e10, len(massVal))
    #print("$$$$Len$$$$",len(x_m))
    fig, ax = plt.subplots(figsize=(8,8))
    print("Min value:", np.min(massVal), "Max Value:", np.max(massVal))
    ax.hist(massVal, bins=1000, range=(np.min(massVal), np.max(massVal)+10))
    ax.set_xlabel("Mass(kg/1e12)")
    ax.set_ylabel("# of Galaxies")
    #ax.set_title("Some galaxies in the localisation volume", fontsize = 25)
    plt.savefig('savedata/massHist'+savename+'.png')

def plotProbHist(catMass, savename):
    #x_m = np.linspace(np.min(massVal/1e10, np.max(massVal/1e10), len(massVal)))
    probNorm = catMass['dP_dV']/np.max(catMass['dP_dV'])
    fig, ax = plt.subplots(figsize=(8,8))
    ax.hist(probNorm, bins=20)
    ax.set_xlabel("Normalized Volume Probabilty")
    ax.set_ylabel("# of Galaxies")
    #ax.set_title("Some galaxies in the localisation volume", fontsize = 25)
    plt.savefig('savedata/probHist'+savename+'.png')

curpath = os.path.abspath('.')
#data = os.path.join(curpath, 'O4/bns_astro/allsky/')
data = os.path.join(curpath, 'Events')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--distcut", help = "If we need a distance cut (values: yes/no)")
parser.add_argument("--plot", help = "If we need plots (values: yes/no)")
args = parser.parse_args()
distcut = args.distcut
event_plot_needed = args.plot


injections = [f for f in os.listdir(data) if f.endswith('.fits')]
total_data = 3
galToShow = 50
top_by_Vol = []
allEvents = []
metadata = {}
metadata['distmean']=0
metadata['diststd']=0
metadata['objid']=0
#event_plot_needed = True
needed_mass_frac = [0.1, 0.5]
i=0
#url = 'https://emfollow.docs.ligo.org/userguide/_static/bayestar.fits.gz'
#inj_file = astropy.utils.data.download_file(url)
for i, inj_file in enumerate(injections[0:2]):
    print("Processing Event #",i)
    print("Injection file",inj_file)
    #hdul = fits.open(os.path.join(data, inj_file))
    #header = hdul[0].header
    #nside = header['NSIDE']
    #print(nside)
    event_dict = {}
    catNew = cat
    #print(fitsData)
    (prob, distmu, distsigma, distnorm), metadata = read_sky_map(os.path.join(data, inj_file), distances=True)
    #prob, distmu, distsigma, distnorm = hp.read_map(inj_file, field=[0,1,2,3])
    plt.clf()
    distmean, diststd = ligo.skymap.distance.parameters_to_marginal_moments(prob, distmu, distsigma)
    print(distmean)
    #savename = inj_file.split('.')[0]
    savename = str(i) + 'new'
    if event_plot_needed=='yes':
        hp.mollview(prob, hold=True)
        fig = plt.gcf()
        plt.savefig('savedata/mollview'+savename+'.png')
        plt.clf()
    npix = len(prob)
    print(npix)
    nside = hp.npix2nside(npix)
    print(nside)
    pixarea = hp.nside2pixarea(nside)
    pixarea_deg = hp.nside2pixarea(nside, degrees=True)
    print(pixarea_deg)
    prob_sorted = np.sort(prob)[::-1]
    cumProb = np.cumsum(prob_sorted)
    print("ProbLen90:", len(np.where(cumProb<0.9)))
    area_loc = len(np.array(np.where(cumProb<0.90))[0])*pixarea_deg
    print("Area:", area_loc)

    center = mark_event_center(prob, nside, os.path.join(data, inj_file), savename, event_plot_needed)
    #probability along radial distance
    if event_plot_needed=='yes':
        print("Marking Event Center")
        
    #Convert all catalog Angles to pixels to query 90% localization area 
    ipix = hp.ang2pix(nside, catNew['_RAJ2000'].value, catNew['_DEJ2000'].value, lonlat=True)
    #print("Getting Galaxies in 1 %% localization by area")
    clucutarea90, numArea = get_90_Loc_Area(ipix, prob, pixarea, distmu, distsigma, distnorm, center, os.path.join(data, inj_file), catNew)
    max_dp_dr = np.max(catNew['dp_dr'])
    #plotArea(catNew, clucutarea90, center, prob, savename, numArea)


    #print("^^^^^^^^^^^^^dist norm^^^^^^^^^^^^^^^", np.sum(cat['dp_dr']))

    #catFin = cat
    #print("Getting Galaxies in 90 localization by volume")
    catFin, clucut90, numVol =  get_90_Loc_Vol(catNew, prob, distmu, distsigma, distnorm, os.path.join(data, inj_file))
    #print("^^^^^^^^^^^^",len(clucut90),"^^^^^^^^^^^^^^")
    if event_plot_needed=='yes':
        #catLast = plotVol(clucut90, center, prob, savename, numVol, galToShow)
        catLast = plotVol(clucut90, center, prob, savename, numVol, galToShow)

    clucut90['Mstar'] = [0 if np.isnan(x) else x for x in clucut90['Mstar']]

    non_zero_mass = np.count_nonzero(clucut90['Mstar'])

    if non_zero_mass > 0: 
        
        if event_plot_needed=='yes':
            print("Getting Projections")
            fig = plot_Projections(nside, prob, distmu, distsigma, distnorm, catLast, metadata, os.path.join(data, inj_file), savename, max_dp_dr)
        else:
            clucut90 = getProbMassSorted(clucut90, 'dP_dV')
            if distcut=='no':
                clucut90 = getProbMassSorted(clucutarea90, 'dP_dA')

        catMass, num_m_frac1, num_m_frac2 = getCumMass(clucut90, needed_mass_frac)
        

        #event_dict['m_frac1'] = catMass[0:num_m_frac1]
        #event_dict['m_frac2'] = catMass[0:num_m_frac2]
        event_dict['m_frac1'] = len(catMass[0:num_m_frac1])
        event_dict['m_frac2'] = len(catMass[0:num_m_frac2])
        event_dict['total_gal'] = len(catMass)
        pixarea_deg = hp.nside2pixarea(nside, degrees=True)
        area_loc = len(prob[np.cumsum(prob)>0.1])*pixarea_deg
        #print("90%% localization Area: ",area_loc,"sq deg")
        event_dict["loc_area"] = area_loc
        event_dict["avg_dist"] = distmean

        allEvents.append(event_dict)

        #catMass = getCumProb(catLast)
        if event_plot_needed=='yes':
            if distcut=='no': 
                x_fifty = plotCumMass(catMass, savename, 'dP_dA')
            else:
                x_fifty = plotCumMass(catMass, savename, 'dP_dV')

        #plotMassHist(catMass, savename)

        #plotProbHist(catMass[0:galToShow], savename)
        #plotCumProb(catMass, savename)
        top_by_Vol.append(catMass)
        
        if event_plot_needed=='yes':
            mollIMg = plt.imread("savedata/mollview"+savename+".png")
            #areaImg = plt.imread("savedata/area"+savename+".png")
            volImg = plt.imread("savedata/volume"+savename+".png")
            projImg = plt.imread("savedata/projection"+savename+".png")
            cumMass = plt.imread("savedata/cumMass"+savename+".png")
            #massHist = plt.imread("savedata/massHist"+savename+".png")
            #probHist = plt.imread("savedata/probHist"+savename+".png")
        
            figi, axi = plt.subplots(nrows=2, ncols=2, figsize=(20,20))
            axi[0][0].imshow(mollIMg)
            axi[0][0].set_title("Mollweide View", fontsize=15)
            axi[0][1].imshow(volImg, cmap='cylon')
            axi[0][1].set_title("Top {} galaxies by mass in 90% credible volume".format(galToShow), fontsize=15)
            
            axi[1][0].imshow(projImg)
            axi[1][0].set_title("Projection Maps and Marginal Distance", fontsize=15) 
            axi[1][1].imshow(cumMass)
            axi[1][1].set_title("Cummulative Probability Coverage", fontsize=15)
            axi[1][1].set(yticklabels=[])  # remove the tick labels
            axi[1][1].set(xticklabels=[])
            axi[1][1].tick_params(left=False)
            axi[1][1].tick_params(top=False)


            for i in range(2):
                for j in range(2):
                    axi[i][j].get_xaxis().set_visible(False)
                    axi[i][j].get_yaxis().set_visible(False) 
            #plt.axis('off')
            plt.savefig('savedata/combined/combined'+savename+'.png')

    
prob_mass_sorted = catMass.sort('prob_mass', reverse=True)
catMass_pandas = catMass.to_pandas()
catMass_pandas.to_csv('csvGal_probMass.csv')
catMass.sort('Mstar', reverse=True)
catMass_pandas = catMass.to_pandas()
catMass_pandas.to_csv('csvGal_Mass.csv')
    