"""
Functions for classification of remote sensing data contained
in an Open Data Cube instance.
"""

from copy import deepcopy

import dask.distributed as dd
import numpy as np
import xarray as xr
from datacube.utils import geometry
from tools.spatial import xr_rasterize


def sklearn_flatten(input_xr):
    """
    Reshape a DataArray or Dataset with spatial (and optionally
    temporal) structure into an np.array with the spatial and temporal
    dimensions flattened into one dimension.

    This flattening procedure enables DataArrays and Datasets to be used
    to train and predict with sklearn models.

    Parameters
    ----------
    input_xr : xarray.DataArray or xarray.Dataset
        Must have dimensions 'x' and 'y', may have dimension 'time'.
        Dimensions other than 'x', 'y' and 'time' are unaffected by the
        flattening.

    Returns
    ----------
    input_np : numpy.array
        A numpy array corresponding to input_xr.data (or
        input_xr.to_array().data), with dimensions 'x','y' and 'time'
        flattened into a single dimension, which is the first axis of
        the returned array. input_np contains no NaNs.

    """
    # cast input Datasets to DataArray
    if isinstance(input_xr, xr.Dataset):
        input_xr = input_xr.to_array()

    # stack across pixel dimensions, handling timeseries if necessary
    if "time" in input_xr.dims:
        stacked = input_xr.stack(z=["x", "y", "time"])
    else:
        stacked = input_xr.stack(z=["x", "y"])

    # finding 'bands' dimensions in each pixel - these will not be
    # flattened as their context is important for sklearn
    pxdims = []
    for dim in stacked.dims:
        if dim != "z":
            pxdims.append(dim)

    # mask NaNs - we mask pixels with NaNs in *any* band, because
    # sklearn cannot accept NaNs as input
    mask = np.isnan(stacked)
    if len(pxdims) != 0:
        mask = mask.any(dim=pxdims)

    # turn the mask into a numpy array (boolean indexing with xarrays
    # acts weird)
    mask = mask.data

    # the dimension we are masking along ('z') needs to be the first
    # dimension in the underlying np array for the boolean indexing to work
    stacked = stacked.transpose("z", *pxdims)
    input_np = stacked.data[~mask]

    return input_np


def _get_training_data_for_shp(
        gdf,
        index,
        row,
        out_arrs,
        out_vars,
        dc,
        dc_query,
        return_coords,
        feature_func=None,
        field=None,
        zonal_stats=None,
):
    """
    This is the core function that is triggered by `collect_training_data`.
    The `collect_training_data` function loops through geometries in a geopandas
    geodataframe and runs the code within `_get_training_data_for_shp`.
    Parameters are inherited from `collect_training_data`.
    See that function for information on the other params not listed below.

    Parameters
    ----------
    index, row : iterables inherited from geopandas object
    out_arrs : list
        An empty list into which the training data arrays are stored.
    out_vars : list
        An empty list into which the data varaible names are stored.


    Returns
    --------
    Two lists, a list of numpy.arrays containing classes and extracted data for
    each pixel or polygon, and another containing the data variable names.

    """

    # prevent function altering dictionary kwargs
    dc_query = deepcopy(dc_query)

    # remove dask chunks if supplied as using
    # mulitprocessing for parallization
    if "dask_chunks" in dc_query.keys():
        dc_query.pop("dask_chunks", None)

    # set up query based on polygon
    geom = geometry.Geometry(geom=gdf.iloc[index].geometry, crs=gdf.crs)
    q = {"geopolygon": geom}

    # merge polygon query with user supplied query params
    dc_query.update(q)

    # Use input feature function
    data = feature_func(dc, dc_query)

    # create polygon mask
    mask = xr_rasterize(gdf.iloc[[index]], data)
    data = data.where(mask)

    # Check that feature_func has removed time
    if "time" in data.dims:
        t = data.dims["time"]
        if t > 1:
            raise ValueError(
                "After running the feature_func, the dataset still has "
                + str(t)
                + " time-steps, dataset must only have"
                + " x and y dimensions."
            )

    if return_coords == True:
        # turn coords into a variable in the ds
        data["x_coord"] = data.x + 0 * data.y
        data["y_coord"] = data.y + 0 * data.x

    # append ID measurement to dataset for tracking failures
    band = [m for m in data.data_vars][0]
    _id = xr.zeros_like(data[band])
    data["id"] = _id
    data["id"] = data["id"] + gdf.iloc[index]["id"]

    # If no zonal stats were requested then extract all pixel values
    if zonal_stats is None:
        flat_train = sklearn_flatten(data)
        flat_val = np.repeat(row[field], flat_train.shape[0])
        stacked = np.hstack((np.expand_dims(flat_val, axis=1), flat_train))

    elif zonal_stats in ["mean", "median", "max", "min"]:
        method_to_call = getattr(data, zonal_stats)
        flat_train = method_to_call()
        flat_train = flat_train.to_array()
        stacked = np.hstack((row[field], flat_train))

    else:
        raise Exception(
            zonal_stats
            + " is not one of the supported"
            + " reduce functions ('mean','median','max','min')"
        )

    out_arrs.append(stacked)
    out_vars.append([field] + list(data.data_vars))


def collect_training_data(
        gdf,
        dc,
        dc_query,
        return_coords=False,
        feature_func=None,
        field=None,
        zonal_stats=None,
        clean=True,
        fail_threshold=0.02,
        fail_ratio=0.5,
        max_retries=3,
):
    """
    This function provides methods for gathering training data from the ODC over 
    geometries stored within a geopandas geodataframe. The function will return a
    'model_input' array containing stacked training data arrays with all NaNs & Infs removed.
    In the instance where ncpus > 1, a parallel version of the function will be run
    (functions are passed to a mp.Pool()). This function can conduct zonal statistics if
    the supplied shapefile contains polygons. The 'feature_func' parameter defines what
    features to produce.

    Parameters
    ----------
    gdf : geopandas geodataframe
        geometry data in the form of a geopandas geodataframe
    dc : datacube instance
    dc_query : dictionary
        Datacube query object, should not contain lat and long (x or y)
        variables as these are supplied by the 'gdf' variable
    return_coords : bool
        If True, then the training data will contain two extra columns 'x_coord' and
        'y_coord' corresponding to the x,y coordinate of each sample. This variable can
        be useful for handling spatial autocorrelation between samples later in the ML workflow.
    feature_func : function
        A function for generating feature layers that is applied to the data within
        the bounds of the input geometry. The 'feature_func' must accept a 'dc_query'
        object, and return a single xarray.Dataset or xarray.DataArray containing
        2D coordinates (i.e x and y, without a third dimension).
        e.g.::

            def feature_function(query):
                dc = datacube.Datacube(app='feature_layers')
                ds = dc.load(**query)
                ds = ds.mean('time')
                return ds

    field : str
        Name of the column in the gdf that contains the class labels
    zonal_stats : string, optional
        An optional string giving the names of zonal statistics to calculate
        for each polygon. Default is None (all pixel values are returned). Supported
        values are 'mean', 'median', 'max', 'min'.
    clean : bool
        Whether or not to remove missing values in the training dataset. If True,
        training labels with any NaNs or Infs in the feature layers will be dropped
        from the dataset.
    fail_threshold : float, default 0.02
        Silent read fails on S3 during mulitprocessing can result in some rows
        of the returned data containing NaN values.
        The'fail_threshold' fraction specifies a % of acceptable fails.
        e.g. Setting 'fail_threshold' to 0.05 means if >5% of the samples in the training dataset
        fail then those samples will be returned to the multiprocessing queue. Below this fraction
        the function will accept the failures and return the results.
    fail_ratio: float
        A float between 0 and 1 that defines if a given training sample has failed.
        Default is 0.5, which means if 50 % of the measurements in a given sample return null
        values, and the number of total fails is more than the 'fail_threshold', the sample
        will be passed to the retry queue.
    max_retries: int, default 3
        Maximum number of times to retry collecting samples. This number is invoked
        if the 'fail_threshold' is not reached.

    Returns
    --------
    Two objects are returned:
    `columns_names`: a list of variable (feature) names
    `model_input`: a numpy.array containing the data values for each feature extracted
    
    """

    # check the dtype of the class field
    if gdf[field].dtype != int:
        raise ValueError(
            'The "field" column of the input vector must contain integer dtypes'
        )

    # set up some print statements
    if feature_func is None:
        raise ValueError(
            "Please supply a feature layer function through the "
            + "parameter 'feature_func'"
        )

    if zonal_stats is not None:
        print("Taking zonal statistic: " + zonal_stats)

    # add unique id to gdf to help with indexing failed rows
    # during multiprocessing
    # if zonal_stats is not None:
    gdf["id"] = range(0, len(gdf))

    # progress indicator
    print("Collecting training data in serial mode")
    i = 0

    # list to store results
    results = []
    column_names = []

    # loop through polys and extract training data
    for index, row in gdf.iterrows():
        print(" Feature {:04}/{:04}\r".format(i + 1, len(gdf)), end="")

        _get_training_data_for_shp(
            gdf,
            index,
            row,
            results,
            column_names,
            dc,
            dc_query,
            return_coords,
            feature_func,
            field,
            zonal_stats,
        )
        i += 1

    # column names are appended during each iteration
    # but they are identical, grab only the first instance
    column_names = column_names[0]

    # Stack the extracted training data for each feature into a single array
    model_input = np.vstack(results)

    # -----------------------------------------------

    # remove id column
    idx_var = column_names[0:-1]
    model_col_indices = [column_names.index(var_name) for var_name in idx_var]
    model_input = model_input[:, model_col_indices]

    if clean == True:
        num = np.count_nonzero(np.isnan(model_input).any(axis=1))
        model_input = model_input[~np.isnan(model_input).any(axis=1)]
        model_input = model_input[~np.isinf(model_input).any(axis=1)]
        print("Removed " + str(num) + " rows wth NaNs &/or Infs")
        print("Output shape: ", model_input.shape)

    else:
        print("Returning data without cleaning")
        print("Output shape: ", model_input.shape)

    return column_names[0:-1], model_input
