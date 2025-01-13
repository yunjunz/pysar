############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Yidi Wang, Jan 2025                #
############################################################
# Recommend import:
#   from mintpy.geometry.dem import calc_shadow, calc_layover, calc_local_slope
#   from mintpy.geometry import dem


import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from mintpy.objects.progress import progressBar


def calc_shadow(dem, reso, los_inc_angle, los_az_angle, plot=False, print_msg=True):
    """Calculate shadow mask

    Parameters: dem           - 2D np.ndarray in int/float, DEM in meters
                reso          - int/float, DEM resolution in meters
                los_inc_angle - float, incidence angle of the LOS vector from target to sensor in degrees
                los_az_angle  - float, azimuth   angle of the LOS vector from target to sensor in degrees
                                measured from the north with anti-clockwise as positve
    Returns:    shadow        - 2D np.ndarray in bool, flag matrix with 1 for shadow and 0 for others
    """
    no_data_value = -9999

    ## rorate DEM to radar coordinates for fast row-wise calcualtion
    dem_rot = simple_geo2rdr(dem, los_az_angle, look_direction='right', interp_order=1)

    ## calculate shadow in radar coordinates
    los_inc_angle_tan = np.tan(np.deg2rad(los_inc_angle))

    # initiate outputs
    shadow_rot = np.zeros(dem_rot.shape, dtype=np.bool_)

    # loop row by row
    num_row = dem_rot.shape[0]
    prog_bar = progressBar(maxValue=num_row, print_msg=print_msg)
    for y in range(num_row):
        prog_bar.update(y+1, every=1, suffix=f'row {y+1}')
        dem_y = dem_rot[y,:]
        if not np.all(np.isnan(dem_y)):
            # max number of pixels to search based on the maximum height difference
            num_search = int(np.ceil((np.nanmax(dem_y) - np.nanmin(dem_y)) * los_inc_angle_tan) / reso)

            # calculate shadow pixel by pixel
            for x in range(dem_y.size):
                if not np.isnan(dem_y[x]):
                    num_search_x = min(x, num_search)
                    dem_2compare = dem_y[x-num_search_x:x] - dem_y[x]
                    # convert nan to a extreamly small value
                    dem_2compare[np.isnan(dem_2compare)] = no_data_value

                    # the maximum allowed height along the num_search line for a shadow-free imaging
                    hgt_los = np.linspace(reso*num_search_x, reso, num_search_x) / los_inc_angle_tan
                    if np.any(dem_2compare >= hgt_los):
                        shadow_rot[y, x] = 1
    prog_bar.close()

    ## rotate back shadow for output
    shadow = simple_rdr2geo(shadow_rot, dem.shape, los_az_angle, look_direction='right', interp_order=0)

    ## plot
    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[10, 8])
        titles = ['DEM (geo-coord)', 'DEM (radar-coord)', 'shadow (geo-coord)', 'shadow (radar-coord)']
        for ax, data, title in zip(axs.flatten(), [dem, dem_rot, shadow, shadow_rot], titles):
            im = ax.imshow(data, interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(title)
        fig.tight_layout()
        plt.show()

    return shadow


def calc_layover(dem, reso, los_inc_angle, los_az_angle, plot=False, print_msg=True):
    """Calculate shadow mask

    Parameters: dem           - 2D np.ndarray in int/float, DEM in meters
                reso          - int/float, DEM resolution in meters
                los_inc_angle - float, incidence angle of the LOS vector from target to sensor in degrees
                los_az_angle  - float, azimuth   angle of the LOS vector from target to sensor in degrees
                                measured from the north with anti-clockwise as positve
    Returns:    layover       - 2D np.ndarray in bool, flag matrix with 1 for layover and 0 for others
    """
    no_data_value = -9999

    ## rorate DEM to radar coordinates for fast row-wise calcualtion
    dem_rot = simple_geo2rdr(dem, los_az_angle, look_direction='right', interp_order=1)

    ## calculate shadow in radar coordinates
    los_inc_angle_tan = np.tan(np.deg2rad(los_inc_angle))

    # initiate outputs
    layover_rot = np.zeros(dem_rot.shape, dtype=np.bool_)

    # loop row by row
    num_row = dem_rot.shape[0]
    prog_bar = progressBar(maxValue=num_row, print_msg=print_msg)
    for y in range(num_row):
        prog_bar.update(y+1, every=1, suffix=f'row {y+1}')
        dem_y = dem_rot[y,:]
        if not np.all(np.isnan(dem_y)):
            # max number of pixels to search based on the maximum height difference
            num_search = int(np.ceil((np.nanmax(dem_y) - np.nanmin(dem_y)) / los_inc_angle_tan) / reso)
            num_pixel = dem_y.size

            # calculate shadow pixel by pixel
            for x in range(num_pixel):
                if not np.isnan(dem_y[x]):
                    num_search_x = min(num_pixel-x, num_search)
                    dem_2compare = dem_y[x:x+num_search_x] - dem_y[x]
                    # convert nan to a extreamly small value
                    dem_2compare[np.isnan(dem_2compare)] = no_data_value

                    # the maximum allowed height along the num_search line for a layover-free imaging
                    hgt_los = np.linspace(reso, reso*num_search_x, num_search_x) * los_inc_angle_tan
                    if np.any(dem_2compare >= hgt_los):
                        layover_rot[y, x] = 1

    prog_bar.close()

    ## rotate back shadow for output
    layover = simple_rdr2geo(layover_rot, dem.shape, los_az_angle, look_direction='right', interp_order=0)

    ## plot
    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[10, 8])
        titles = ['DEM (geo-coord)', 'DEM (radar-coord)', 'layover (geo-coord)', 'layover (radar-coord)']
        for ax, data, title in zip(axs.flatten(), [dem, dem_rot, layover, layover_rot], titles):
            im = ax.imshow(data, interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(title)
        fig.tight_layout()
        plt.show()

    return layover


def calc_local_slope(dem, reso, plot=False):
    """Calculate the local slope and max slope direction/aspect

    Parameters: dem       - 2D np.ndarray, DEM in meters
                reso      - int/float, DEM resolution in meters
    Returns:    slope_deg - 2D np.ndarray, local slope magnitude in degrees
                slope_dir - 2D np.ndarray, local max slope azimuth direction,
                            measured from the north with anti-clockwise as positive
    """
    # calculate the spatial gradient along the X/Y direction in ratios
    reso = 90
    dy, dx = np.gradient(dem)
    dy /= -reso
    dx /= reso

    # compute the local slope magnitude in ratio and in degrees
    slope_ratio = np.sqrt(dx**2 + dy**2)
    slope_deg = np.rad2deg(np.arctan(slope_ratio))

    # compute the local max slope direction (azimuth angle)
    slope_dir = np.rad2deg(np.arctan2(dy, dx)) + 90
    slope_dir -= np.round(slope_dir / 360.) * 360.

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[10, 3])
        titles = ['DEM [m]', 'Slope [deg]', 'Slope Direction']
        for ax, data, title in zip(axs.flatten(), [dem, slope, slope_dir], titles):
            im = ax.imshow(data, interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(title)
        fig.tight_layout()
        plt.show()

    return slope_deg, slope_dir


################################################################################
def simple_geo2rdr(geo_data, los_az_angle, look_direction='right', interp_order=1):
    """Use simple rotation and flip operations to convert data from geo to radar coordiantes.

    Note: this is a very simple operation, the resampling accuracy is NOT guaranteed!

    Parameters: geo_data     - 2D np.ndarray, data in geo-coordinates
                los_az_angle - float, float, azimuth   angle of the LOS vector from target to sensor in degrees
                               measured from the north with anti-clockwise as positve
                look_dir     - str, radar side looking direction, left or right
                interp_order - int, order of the spline interpolation, recommend 1 for int/float and 0 for bool data
    Returns:    rdr_data     - 2D np.ndarray, data in radar-coordinates
    """
    no_data_value = -9999

    # grab orbit direction
    orb_az_angle = los_az_angle - 90 if look_direction == 'right' else los_az_angle + 90
    orb_az_angle -= np.round(orb_az_angle / 360.) * 360.       # round to (-180, 180]
    orb_dir = 'ASCENDING' if np.abs(orb_az_angle) < 90 else 'DESCENDING'

    # 1. rotate the LOS vector (from sensor to target) to the E / W direction for asc / desc orbits
    # calculate the rotate angle
    los_az_angle2 = los_az_angle + 180                       # LOS vector from sensor to target
    los_az_angle2 -= np.round(los_az_angle2 / 360.) * 360.   # round to [0, 360]
    if orb_dir.lower().startswith('a'):
        rot_destination = -90
    else:
        rot_destination = 90
    rot_angle = rot_destination - los_az_angle2

    # rotate
    rdr_data = ndimage.rotate(
        geo_data,
        angle=rot_angle,
        reshape=True,
        order=interp_order,
        mode='constant',
        cval=no_data_value
    ).astype(np.float32)
    rdr_data[rdr_data == no_data_value] = np.nan

    # 2. flip up-down / left-right for ascending / descending orbit
    if orb_dir.lower().startswith('a'):
        rdr_data = np.flipud(rdr_data)
    else:
        rdr_data = np.fliplr(rdr_data)

    return rdr_data


def simple_rdr2geo(rdr_data, out_shape, los_az_angle, look_direction='right', interp_order=0):
    """Use simple rotation and flip operations to convert data from radar- to geo-coordiantes.

    Note: this is a very simple operation, the resampling accuracy is NOT guaranteed!

    Parameters: rdr_data     - 2D np.ndarray, data in rdr-coordinates
                out_shape    - list/tuple of 2 int, output data shape
                los_az_angle - float, float, azimuth   angle of the LOS vector from target to sensor in degrees
                               measured from the north with anti-clockwise as positve
                look_dir     - str, radar side looking direction, left or right
                interp_order - int, order of the spline interpolation, recommend 1 for int/float and 0 for bool data
    Returns:    geo_data     - 2D np.ndarray, data in geo-coordinates
    """
    no_data_value = -9999

    # grab orbit direction
    orb_az_angle = los_az_angle - 90 if look_direction == 'right' else los_az_angle + 90
    orb_az_angle -= np.round(orb_az_angle / 360.) * 360.       # round to (-180, 180]
    orb_dir = 'ASCENDING' if np.abs(orb_az_angle) < 90 else 'DESCENDING'

    # 1. flip up-down / left-right for ascending / descending orbit
    #shadow = np.zeros(out_shape, dtype=np.bool_)
    if orb_dir.lower().startswith('a'):
        geo_data = np.flipud(rdr_data)
    else:
        geo_data = np.fliplr(rdr_data)

    # 2. rotate the LOS vector (from sensor to target) to the E / W direction for asc / desc orbits
    # calculate the rotate angle
    los_az_angle2 = los_az_angle + 180                       # LOS vector from sensor to target
    los_az_angle2 -= np.round(los_az_angle2 / 360.) * 360.   # round to [0, 360]
    if orb_dir.lower().startswith('a'):
        rot_destination = -90
    else:
        rot_destination = 90
    rot_angle = (rot_destination - los_az_angle2) * -1

    # rotate
    geo_data = ndimage.rotate(
        geo_data,
        angle=rot_angle,
        reshape=True,
        order=interp_order,
        mode='constant',
    )

    # 3. crop to the size of the input DEM, assuming consistent center
    r0 = (geo_data.shape[0] - out_shape[0]) // 2
    c0 = (geo_data.shape[1] - out_shape[1]) // 2
    geo_data = geo_data[r0:r0+out_shape[0], c0:c0+out_shape[1]]

    return geo_data
