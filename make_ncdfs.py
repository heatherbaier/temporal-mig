from mpi4py import MPI

import geopandas as gpd
import rasterio as rio
import netCDF4 as nc
import pandas as pd
import numpy as np
import argparse
import glob
import ast
import os


def get_max_image_size(files):
    max_x, max_y = 0, 0
    opened_files = []
    for i in files:
        f = rio.open(i)
        opened_files.append(f)
        dims = f.shape
        if dims[0] > max_x:
            max_x = dims[0]
        if dims[1] > max_y:
            max_y = dims[1]
    return (max_x, max_y), opened_files


def pad_image(im, image_dims):
    bottom_pad = image_dims[0] - im.shape[0]
    right_pad = image_dims[1] - im.shape[1]
    im = np.pad(im, ((0, bottom_pad), (0, right_pad), (0, 0)))
    return im


def create_nc(files, muni_id, output_dir, y_list):

    image_dims, opened_files = get_max_image_size(files)
    
    file_name = os.path.join(output_dir, str(muni_id) + ".nc")
            
    for c in range(len(files)):
    
        b1, b2, b3 = opened_files[c].read(1), opened_files[c].read(2), opened_files[c].read(3)
        im = np.dstack([b1, b2, b3])
        
        im = pad_image(im, image_dims)
                
        # If it's the first file, make the .nc file
        if c == 0:
            
            # create dimensions
            ds = nc.Dataset(file_name, 'w', format='NETCDF4')
            time = ds.createDimension('time', None)
            
            ds.createDimension('x', image_dims[0])
            ds.createDimension('y', image_dims[1])
                        
            ds.createDimension('z', 3)
            migrants = ds.createDimension('migrants', None)            
            
            # create variables
            times = ds.createVariable('time', 'i1', ('time',))
            migrants = ds.createVariable('migrants', 'i8', ('migrants',))
            ims = ds.createVariable('ims', float, ('time', 'x', 'y', 'z',))
            
            migrants[0], ims[0] = y_list[c], im
                        
        else:
            
            migrants[c], ims[c] = y_list[c], im

        with open(log_name, "a") as f:
            f.write("Image: " + str(c) + "  Migrants: " + str(y_list[c]) + "\n")
                        
    ds.close()


if __name__  == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", help="Country ISO")
    parser.add_argument("world_size", help="ADM level")
    args = parser.parse_args()

    df = pd.read_csv("/sciclone/geograd/heather_data/ncdf_table_0225.csv")
    output_dir = "/sciclone/geograd/heather_data/netCDFs/"
    log_name = "/sciclone/home20/hmbaier/tm/nclog_rank_" + str(args.rank) + ".txt"

    print("WORLD SIZE: ", args.world_size)
    print("RANK: ", args.rank)

    all_munis = np.array_split(df.GEOLEVEL2.to_list(), int(args.world_size))
    rank_munis = all_munis[int(args.rank) - 1]

    df = df[df["GEOLEVEL2"].isin(rank_munis)]

    for col, row in df.iterrows():

        try:

            y_list = row.y_list.strip('][').split(', ')
            y_list = [float(i) for i in y_list]

            imagery_list = row.imagery_list.strip('][').split(', ')
            imagery_list = [i.strip("'") for i in imagery_list]

            with open(log_name, "a") as f:
                f.write("\nMUNI: " + str(row.GEOLEVEL2) + "\n")
                
            create_nc(files = imagery_list, 
                    muni_id = str(row.GEOLEVEL2),
                    output_dir = output_dir,
                    y_list = y_list)

        except Exception as e:

            with open(log_name, "a") as f:
                f.write(str(row.GEOLEVEL2) + " didn't work because " + str(e) + "\n")
