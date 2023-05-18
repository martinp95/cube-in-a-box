'''
Spatial analyses functions for Open data cube
'''

# Import required packages
import collections
import xarray as xr
import rasterio.features
import scipy.interpolate
import multiprocessing as mp
from skimage.measure import label
from geopy.geocoders import Nominatim
from datacube.utils.cog import write_cog
from datacube.utils.geometry import assign_crs


def xr_rasterize(gdf,
                 da,
                 attribute_col=False,
                 crs=None,
                 transform=None,
                 name=None,
                 x_dim='x',
                 y_dim='y',
                 export_tiff=None,
                 verbose=False,
                 **rasterio_kwargs):
    """
    Rasterizes a geopandas.GeoDataFrame into an xarray.DataArray.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A geopandas.GeoDataFrame object containing the vector/shapefile
        data you want to rasterise.
    da : xarray.DataArray or xarray.Dataset
        The shape, coordinates, dimensions, and transform of this object
        are used to build the rasterized shapefile. It effectively
        provides a template. The attributes of this object are also
        appended to the output xarray.DataArray.
    attribute_col : string, optional
        Name of the attribute column in the geodataframe that the pixels
        in the raster will contain.  If set to False, output will be a
        boolean array of 1's and 0's.
    crs : str, optional
        CRS metadata to add to the output xarray. e.g. 'epsg:3577'.
        The function will attempt get this info from the input
        GeoDataFrame first.
    transform : affine.Affine object, optional
        An affine.Affine object (e.g. `from affine import Affine;
        Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "6886890.0) giving the
        affine transformation used to convert raster coordinates
        (e.g. [0, 0]) to geographic coordinates. If none is provided,
        the function will attempt to obtain an affine transformation
        from the xarray object (e.g. either at `da.transform` or
        `da.geobox.transform`).
    x_dim : str, optional
        An optional string allowing you to override the xarray dimension
        used for x coordinates. Defaults to 'x'. Useful, for example,
        if x and y dims instead called 'lat' and 'lon'.
    y_dim : str, optional
        An optional string allowing you to override the xarray dimension
        used for y coordinates. Defaults to 'y'. Useful, for example,
        if x and y dims instead called 'lat' and 'lon'.
    export_tiff: str, optional
        If a filepath is provided (e.g 'output/output.tif'), will export a
        geotiff file. A named array is required for this operation, if one
        is not supplied by the user a default name, 'data', is used
    verbose : bool, optional
        Print debugging messages. Default False.
    **rasterio_kwargs :
        A set of keyword arguments to rasterio.features.rasterize
        Can include: 'all_touched', 'merge_alg', 'dtype'.

    Returns
    -------
    xarr : xarray.DataArray

    """

    # Check for a crs object
    try:
        crs = da.geobox.crs
    except:
        try:
            crs = da.crs
        except:
            if crs is None:
                raise ValueError("Please add a `crs` attribute to the "
                                 "xarray.DataArray, or provide a CRS using the "
                                 "function's `crs` parameter (e.g. crs='EPSG:3577')")

    # Check if transform is provided as a xarray.DataArray method.
    # If not, require supplied Affine
    if transform is None:
        try:
            # First, try to take transform info from geobox
            transform = da.geobox.transform
        # If no geobox
        except:
            try:
                # Try getting transform from 'transform' attribute
                transform = da.transform
            except:
                # If neither of those options work, raise an exception telling the
                # user to provide a transform
                raise TypeError("Please provide an Affine transform object using the "
                                "`transform` parameter (e.g. `from affine import "
                                "Affine; Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "
                                "6886890.0)`")

    # Grab the 2D dims (not time)
    try:
        dims = da.geobox.dims
    except:
        dims = y_dim, x_dim

        # Coords
    xy_coords = [da[dims[0]], da[dims[1]]]

    # Shape
    try:
        y, x = da.geobox.shape
    except:
        y, x = len(xy_coords[0]), len(xy_coords[1])

    # Reproject shapefile to match CRS of raster
    if verbose:
        print(f'Rasterizing to match xarray.DataArray dimensions ({y}, {x})')

    try:
        gdf_reproj = gdf.to_crs(crs=crs)
    except:
        # Sometimes the crs can be a datacube utils CRS object
        # so convert to string before reprojecting
        gdf_reproj = gdf.to_crs(crs={'init': str(crs)})

    # If an attribute column is specified, rasterise using vector
    # attribute values. Otherwise, rasterise into a boolean array
    if attribute_col:
        # Use the geometry and attributes from `gdf` to create an iterable
        shapes = zip(gdf_reproj.geometry, gdf_reproj[attribute_col])
    else:
        # Use geometry directly (will produce a boolean numpy array)
        shapes = gdf_reproj.geometry

    # Rasterise shapes into an array
    arr = rasterio.features.rasterize(shapes=shapes,
                                      out_shape=(y, x),
                                      transform=transform,
                                      **rasterio_kwargs)

    # Convert result to a xarray.DataArray
    xarr = xr.DataArray(arr,
                        coords=xy_coords,
                        dims=dims,
                        attrs=da.attrs,
                        name=name if name else None)

    # Add back crs if xarr.attrs doesn't have it
    if xarr.geobox is None:
        xarr = assign_crs(xarr, str(crs))

    if export_tiff:
        if verbose:
            print(f"Exporting GeoTIFF to {export_tiff}")
        write_cog(xarr,
                  export_tiff,
                  overwrite=True)

    return xarr
