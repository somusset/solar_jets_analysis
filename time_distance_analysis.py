# -*- coding: utf-8 -*-

import os
import glob

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from scipy import ndimage, datasets

import astropy.units as u
#from astropy.coordinates import Angle ### NOT SURE IT IS USEFUL, TRYING WITHOUT IT
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ImageNormalize, SqrtStretch

import sunpy.coordinates  # NOQA
from sunpy.coordinates import frames
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

from utils.Jet_class import json_export_list, json_import_list, csv_import_list, csv_export_list

### Some global variables

# the sjh catalogue - this could be actually true for anyone working in my repo solar_jet_analysis
json_file = 'exports/Jet_clusters_3.0_2.0.paperID.json'

data_folder = 'C:/Users/sophie.musset/data/sjh_cutouts/'
jsoc_email = "sophie.musset@esa.int"

### function to download the cutouts, with the jet ID being the input.

def query_to_download_jet_cutouts(jetid, wavelength, cadence=12*u.s):
    """
    This function will take the jetid which is a unique jet id from the sjh catalogue,
    and download the cutouts for this jet in the specified wavelength.

        INPUTS
    jetid: str e.g. 'sjh_2011-02-13T05_1'
    wavelength: one of the AIA channel (with units) e.g. 304*u.angstrom
    cadence: the time cadence for the cutout request, defaul is 12 seconds (highest cadence)

        OUTPUT
    query

        DESCRIPTION
    The function check if the data folder associated with this jet already exists, 
        and if not, it will create it.
    It will look into the folder to check if data in this wavelength already exists.
    will look up the sjh catalogue to extract the time information and the HEK id.
    It will then access the HEK catalogue online to get the FOV
    It will then write a cutout request

        WARNING!!
    - If wavelength is not provided in angstrom it will break when looking for the files.
        This should be fix by converting wavelength in angstrom at the beginning of the function

        EXAMPLE 
    query, this_data_dir = query_to_download_jet_cutouts('sjh_2011-02-13T05_1', 304*u.angstrom)
    files = Fido.fetch(query, overwrite=False, path=this_data_dir+'{file}') ### this will actually download the files
    
    """

    ### check if the folder exists, and if not, create it
    this_data_dir = data_folder+jetid+'/'
    if not os.path.isdir(this_data_dir):
        os.mkdir(this_data_dir)
    
    files = glob.glob(this_data_dir+"aia.lev1_euv_12s.*."+str(int(wavelength.value))+".image.fits")
    if len(files) != 0:
        if input("some files already exist in the folder for this wavelength. Continue? [y/n]") != 'y':
            return

    #This function takes the jet id, look it up in the jet catalogue and pull out the needed info
    Jet_clusters=json_import_list(json_file)
    sjh_identifier = np.array([Jet_clusters[i].ID for i in range(len(Jet_clusters))], dtype=str)
    jetid_id = np.where(sjh_identifier == jetid)
    jet_start_time = Jet_clusters[jetid_id][0].obs_time
    jet_end_time = jet_start_time + np.timedelta64(int(Jet_clusters[jetid_id][0].Duration),'m')
    jet_HEK_event = Jet_clusters[jetid_id][0].SOL
    
    ### check if the folder exists, and if not, create it
    this_data_dir = data_folder+jetid+'/'
    if not os.path.isdir(this_data_dir):
        os.mkdir(this_data_dir)
    
    # Find the FOV by looking up the HEK database
    time_range_search = [jet_start_time - np.timedelta64(12,'h'), jet_start_time + np.timedelta64(12,'h')]
    timerange = a.Time(time_range_search[0].astype(str), time_range_search[1].astype(str))
    res = Fido.search(timerange, a.hek.CJ) 
    selection_hek = res['hek'][res['hek']["SOL_standard"]==jet_HEK_event]
    fov = selection_hek['hpc_bbox']
    x1 = float(fov.value[0].strip('POLYGON()').split(',')[0].split()[0])
    x2 = float(fov.value[0].strip('POLYGON()').split(',')[1].split()[0])
    y1 = float(fov.value[0].strip('POLYGON()').split(',')[0].split()[1])
    y2 = float(fov.value[0].strip('POLYGON()').split(',')[2].split()[1])
    
    ### create the cutout request
    start_time_cutout = Time(jet_start_time.astype(str), format='isot', scale='utc')
    end_time_cutout = Time(jet_end_time.astype(str), format='isot', scale='utc')
    bottom_left = SkyCoord(x1*u.arcsec, y1*u.arcsec, obstime=start_time_cutout, observer="earth", frame="helioprojective")
    top_right = SkyCoord(x2*u.arcsec, y2*u.arcsec, obstime=start_time_cutout, observer="earth", frame="helioprojective")
    cutout = a.jsoc.Cutout(bottom_left, top_right=top_right, tracking=False)
    
    query = Fido.search(
        a.Time(start_time_cutout, end_time_cutout),
        a.Wavelength(wavelength),
        a.Sample(cadence),
        a.jsoc.Series.aia_lev1_euv_12s,
        a.jsoc.Notify(jsoc_email),
        a.jsoc.Segment.image,
        cutout,
    )

    return query, this_data_dir

### Function to find the data for a specific jet and load it in a map sequence
    
def load_jet_cutouts_in_maps(jetid, wavelength):
    """
    This function will load the cutouts already downloaded for a jet event in a sunpy map sequence.

        INPUTS
    jetid: str e.g. 'sjh_2011-02-13T05_1'
    wavelength: one of the AIA channel (with units) e.g. 304*u.angstrom

        Warning!!
    - If wavelength is not provided in angstrom it will break when looking for the files.
        This should be fix by converting wavelength in angstrom at the beginning of the function

        EXAMPLE 
    map_seq = load_jet_cutouts_in_maps('sjh_2011-02-13T05_1', 304*u.angstrom)
    """

    ### check if the folder exists, and if not, create it
    this_data_dir = data_folder+jetid+'/'
    if not os.path.isdir(this_data_dir):
        print("No data folder found for this jet")
        return

    files = glob.glob(this_data_dir+"aia.lev1_euv_12s.*."+str(int(wavelength.value))+".image.fits")
    if len(files) == 0:
        print("No file found for this jet in this wavelength")
        return
    
    map_seq = sunpy.map.Map(files, sequence=True)
    return map_seq

### function that takes the jet id and produce the time-distance plot

def coordinates_in_rotated_frame(px,py,rotation_angle,array):
    """
    This function takes pixel coordinates correspondin to one array 
    and returns the corresponding pixel coordinates once the array
    has been rotated

        INPUTS
    px, py: pixel coordinates
    rotation: rotation angle with units
    array: original array (needed only to extract array size?...), before rotation

        OUTPUTS
    np.array countaining new_px, new_py, the corresponding pixel coordinates in rotated frame

        EXAMPLE 
    new_coords = coordinates_in_rotated_frame(px,py,rotation_angle,data_array)
    """
    
    # extract the width and length of the data array
    Lx = np.shape(array)[1]
    Ly = np.shape(array)[0]

    # calculate polar coordinates before rotation
    center_r = np.sqrt(px*px+py*py)
    center_angle = np.arctan(py/px) * u.rad
    
    # polar coordinate in (intermediary) rotated frame 
    new_center_r = center_r
    new_angle = center_angle - rotation_angle 

    # shift in x and y for the final frame
    if (rotation_angle>=0)&(rotation_angle<=90*u.deg):
        x_shift = 0
        y_shift = Lx*np.sin(rotation_angle)
    if (rotation_angle>90*u.deg)&(rotation_angle<=180*u.deg):
        x_shift = Lx*np.abs(np.cos(rotation_angle))
        y_shift = Lx*np.abs(np.sin(rotation_angle)) + Ly*np.abs(np.cos(rotation_angle))
    if (rotation_angle>=-90*u.deg)&(rotation_angle<0*u.deg):
        x_shift = Ly*np.abs(np.sin(rotation_angle))
        y_shift = 0
    if (rotation_angle>-180*u.deg)&(rotation_angle<-90*u.deg):
        x_shift = Lx*np.abs(np.cos(rotation_angle)) + Ly*np.abs(np.sin(rotation_angle))
        y_shift = Ly*np.abs(np.cos(rotation_angle))
    
    #new coordinates
    new_px = new_center_r*np.cos(new_angle) + x_shift
    new_py = new_center_r*np.sin(new_angle) + y_shift

    return np.array([new_px,new_py])

def extract_distance_line_from_box(cutout_map, box_center, box_width, box_length, box_angle):
    """
    This function takes a map and box info and produce 
    the 1D array of data that will be in the time-distance plot for this time

    To do so, it will extract the data along the length of the box, and sum every lines along the width of the box,
    so that it will create a 1D array of information.

        INPUTS
    cutout_map: one map
    box_center: coordinates in arcsec of the box center, with units
    box_width: box width in arcsec, with units
    box_length: box length in arcsec, with units
    box_angle: box angle in degrees, with units

        OUTPUTS
    line: the 1d array of data
    distance_axis: the distance in arcsec with units

        EXAMPLE 
    line, distance = extract_distance_line_from_box(aia_map, 
                                                    rectangle_center, 
                                                    rectangle_width, 
                                                    rectangle_length, 
                                                    rectangle_angle)

    """
    ### Extract information from the map
    data_array = cutout_map.data
    pixel_scale = 0.5*(cutout_map.scale[0]+cutout_map.scale[1])
    
    ### Calculate box information in pixels
    rectangle_width_inpix = box_width/pixel_scale
    rectangle_length_inpix = box_length/pixel_scale
    rectangle_center_coord = SkyCoord(box_center[0], box_center[1], 
                                  obstime=cutout_map.date, ### the box will "follow" the data in time
                                  observer="earth", 
                                  frame="helioprojective")
    with frames.Helioprojective.assume_spherical_screen(cutout_map.observer_coordinate, only_off_disk=True):
        px, py = cutout_map.wcs.world_to_pixel(rectangle_center_coord)
    
    ### rotate data
    rotation_angle = box_angle
    rotated_data = ndimage.rotate(data_array, rotation_angle, reshape=True)
    
    ### calculate coordinates of the box in the rotated data
    new_coords = coordinates_in_rotated_frame(px,py,rotation_angle,data_array)
    rectangle_xx = new_coords[0] + 0.5*rectangle_length_inpix.value*np.array([-1,1])
    rectangle_yy = new_coords[1] + 0.5*rectangle_width_inpix.value*np.array([-1,1])
    
    ### extract data
    in_box = rotated_data[ int(rectangle_yy[0]):int(rectangle_yy[1]), int(rectangle_xx[0]):int(rectangle_xx[1])]
    line = np.sum(in_box,0)
    
    ## create a array for the axis
    distance_axis = np.squeeze(np.arange(len(line))*u.pixel*pixel_scale)
    
    return line, distance_axis

def get_jet_box_angle(jet_entry):
    """
    This function takes a jet entry selected in the json file from the jet database
    and calculate the angle of the box based on the information of individual boxes
    It takes the angle of the biggest box for this cluster jet.

        INPUT
    jet entry in the form Jet_clusters[jet_id]

        OUTPUT
    the angle value (not unit)
    """
    heights = []
    angles = []
    for jet in jet_entry.jets:
        heights.append(jet.height)
        angles.append(jet.angle)
    angle = angles[heights.index(max(heights))]
    return angle

def compute_time_distance(jetid, wavelength):
    """
    This function will create the time-distance array for a given jet in the chosen AIA channel.
    It will take data from my computer if it has been downloaded already.

        INPUTS
    jetid: str e.g. 'sjh_2011-02-13T05_1'
    wavelength: one of the AIA channel (with units) e.g. 304*u.angstrom

        OUTPUTS
    time_distance_array: 2D array containing the time-distance data with time as the first dimension and distance the second (check this)
    time_array: nd array with datetime objects
    distance_array: array of distance with units, starts at 0.

        CALLS
    load_jet_cutouts_in_maps()
    extract_distance_line_from_box()
    get_jet_box_angle()

        Warning!!
    - If wavelength is not provided in angstrom it will break when looking for the files.
        This should be fix by converting wavelength in angstrom at the beginning of the function

        EXAMPLE 
    time_distance_array, time_array, distance_array = compute_time_distance('sjh_2011-02-13T05_1', 304*u.angstrom)
    """

    # load data into map sequence
    map_seq = load_jet_cutouts_in_maps(jetid, wavelength)
    print("number of files for this time-distance plot:")
    print(len(map_seq))
    if len(map_seq) < 2:
        return 'not enough data to make a time distance array','',''

    # find jet information in sjh catalogue
    Jet_clusters=json_import_list(json_file)
    sjh_identifier = np.array([Jet_clusters[i].ID for i in range(len(Jet_clusters))], dtype=str)
    jetid_id = np.where(sjh_identifier == jetid)
    jet_base_center = [Jet_clusters[jetid_id][0].Bx*u.arcsec, Jet_clusters[jetid_id][0].By*u.arcsec]
    jet_box_width = Jet_clusters[jetid_id][0].Width*u.arcsec
    jet_box_length = Jet_clusters[jetid_id][0].Max_Height*u.arcsec
    #jet_box_length = Jet_clusters[jetid_id][0].Height*u.arcsec
    
    # ------------- UPDATE THIS ONCE THE JET ANGLE IS IN THE JSON-------------------
 #   jet_box_angle = Angle(get_jet_box_angle(Jet_clusters[jetid_id][0]), u.degree)
    jet_box_angle = get_jet_box_angle(Jet_clusters[jetid_id][0])*u.degree
    #-------------------------------------------------------------------------------

    # In the database I have the base coordinates but in my routine I use the center of the box
    jet_box_center = [ jet_base_center[0] + 0.5*jet_box_length*np.cos(jet_box_angle), 
                               jet_base_center[1] + 0.5*jet_box_length*np.sin(jet_box_angle) ]
 

    ##### NEED TO ADD SOMETHING THAT CHECKS THAT THE MAP TIME RANGE CORRESPOND TO THE JET TIME RANGE
    
    # loop over maps in map sequence to produce the time-distance plot
    time_distance = []
    times = []
    distances = []
    for aia_map in map_seq:
        line, distance = extract_distance_line_from_box(aia_map, 
                                                    jet_box_center, 
                                                    jet_box_width, jet_box_length, 
                                                    jet_box_angle)
        time_distance.append(line)
        distances.append(distance)
        times.append(aia_map.date.to_datetime())
    
    time_distance_array=np.array(time_distance).transpose()
    time_array = np.array(times)
    distance_array = np.array(distances[0]) ### will need to change when the code breaks because distance is not of the same length for each map...

    return time_distance_array, time_array, distance_array

def plot_time_distance(time_distance_array, time_array, distance_array, vfactor=1.):
    """
    This function takes the time_distance data generated by compute_time_distance() 
    and plot it

        INPUTS
    time_distance_array: 2D array
    time_array: 1D array
    distance_array: 1D array

        KEYWORD
    vfactor: factor to play with display. Set this < 1 to enhance faint features

        OUTPUTS
    None

        EXAMPLE 
    plot_time_distance(time_distance_array, time_array, distance_array, vfactor=0.5)
    """

    fig = plt.figure(dpi=100)
    ax = fig.subplots(1, 1)

    x_lims = mdates.date2num(time_array)
    y_lims = [distance_array.min(), distance_array.max()]

    ax.imshow(np.log(time_distance_array), origin="lower", cmap='sdoaia304', 
          extent = [x_lims[0], x_lims[-1],  y_lims[0], y_lims[-1]], 
          aspect='auto', 
          vmax=np.log(time_distance_array.max()*vfactor))

    ax.set_title('Time-distance plot', fontsize = 15)
    ax.set_xlabel('Time (hour:minute)', fontsize = 12)
    ax.set_ylabel('Distance (arcsec)', fontsize = 12)

    ax.xaxis_date()

    locator = mdates.AutoDateLocator(minticks=3, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.show()
    return
        