"""
Functions for plotting Open Data Cube data.
"""

# Import required packages
import math
import folium
import ipywidgets
import branca
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import matplotlib.cm as cm
from matplotlib import colors as mcolours
from pyproj import Proj, transform
from IPython.display import display
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipyleaflet import Map, Marker, Popup, GeoJSON, basemaps, Choropleth
from skimage import exposure
from branca.colormap import linear
from odc.ui import image_aspect

from matplotlib.animation import FuncAnimation
import pandas as pd
from pathlib import Path
from shapely.geometry import box
from skimage.exposure import rescale_intensity
from tqdm.auto import tqdm
import warnings


def rgb(
    ds,
    bands=["red", "green", "blue"],
    index=None,
    index_dim="time",
    robust=True,
    percentile_stretch=None,
    col_wrap=4,
    size=6,
    aspect=None,
    savefig_path=None,
    savefig_kwargs={},
    **kwargs,
):

    """
    Takes an xarray dataset and plots RGB images using three imagery
    bands (e.g ['red', 'green', 'blue']). The `index`
    parameter allows easily selecting individual or multiple images for
    RGB plotting. Images can be saved to file by specifying an output
    path using `savefig_path`.
    This function was designed to work as an easier-to-use wrapper
    around xarray's `.plot.imshow()` functionality.

    Parameters
    ----------
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array to plot as an RGB
        image. If the array has more than two dimensions (e.g. multiple
        observations along a 'time' dimension), either use `index` to
        select one (`index=0`) or multiple observations
        (`index=[0, 1]`), or create a custom faceted plot using e.g.
        `col="time"`.
    bands : list of strings, optional
        A list of three strings giving the band names to plot. Defaults
        to '['red', 'green', 'blue']'. If the dataset does not contain
        bands named `'red', 'green', 'blue'`, then `bands` must be
        specified.
    index : integer or list of integers, optional
        `index` can be used to select one (`index=0`) or multiple
        observations (`index=[0, 1]`) from the input dataset for
        plotting. If multiple images are requested these will be plotted
        as a faceted plot.
    index_dim : string, optional
        The dimension along which observations should be plotted if
        multiple observations are requested using `index`. Defaults to
        `time`.
    robust : bool, optional
        Produces an enhanced image where the colormap range is computed
        with 2nd and 98th percentiles instead of the extreme values.
        Defaults to True.
    percentile_stretch : tuple of floats
        An tuple of two floats (between 0.00 and 1.00) that can be used
        to clip the colormap range to manually specified percentiles to
        get more control over the brightness and contrast of the image.
        The default is None; '(0.02, 0.98)' is equivelent to
        `robust=True`. If this parameter is used, `robust` will have no
        effect.
    col_wrap : integer, optional
        The number of columns allowed in faceted plots. Defaults to 4.
    size : integer, optional
        The height (in inches) of each plot. Defaults to 6.
    aspect : integer, optional
        Aspect ratio of each facet in the plot, so that aspect * size
        gives width of each facet in inches. Defaults to None, which
        will calculate the aspect based on the x and y dimensions of
        the input data.
    savefig_path : string, optional
        Path to export image file for the RGB plot. Defaults to None,
        which does not export an image file.
    savefig_kwargs : dict, optional
        A dict of keyword arguments to pass to
        `matplotlib.pyplot.savefig` when exporting an image file. For
        all available options, see:
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
    **kwargs : optional
        Additional keyword arguments to pass to `xarray.plot.imshow()`.
        For example, the function can be used to plot into an existing
        matplotlib axes object by passing an `ax` keyword argument.
        For more options, see:
        http://xarray.pydata.org/en/stable/generated/xarray.plot.imshow.html
    Returns
    -------
    An RGB plot of one or multiple observations, and optionally an image
    file written to file.
    """

    # If bands are not in the dataset
    ds_vars = list(ds.data_vars)
    if set(bands).issubset(ds_vars) == False:
        raise ValueError(
            "rgb() bands do not match band names in dataset. "
            "Note the default rgb() bands are ['red', 'green', 'blue']."
        )

    # If ax is supplied via kwargs, ignore aspect and size
    if "ax" in kwargs:

        # Create empty aspect size kwarg that will be passed to imshow
        aspect_size_kwarg = {}
    else:
        # Compute image aspect
        if not aspect:
            aspect = image_aspect(ds)

        # Populate aspect size kwarg with aspect and size data
        aspect_size_kwarg = {"aspect": aspect, "size": size}

    # If no value is supplied for `index` (the default), plot using default
    # values and arguments passed via `**kwargs`
    if index is None:

        # Select bands and convert to DataArray
        da = ds[bands].to_array()

        # If percentile_stretch == True, clip plotting to percentile vmin, vmax
        if percentile_stretch:
            vmin, vmax = da.compute().quantile(percentile_stretch).values
            kwargs.update({"vmin": vmin, "vmax": vmax})

        # If there are more than three dimensions and the index dimension == 1,
        # squeeze this dimension out to remove it
        if (len(ds.dims) > 2) and ("col" not in kwargs) and (len(da[index_dim]) == 1):

            da = da.squeeze(dim=index_dim)

        # If there are more than three dimensions and the index dimension
        # is longer than 1, raise exception to tell user to use 'col'/`index`
        elif (len(ds.dims) > 2) and ("col" not in kwargs) and (len(da[index_dim]) > 1):

            raise Exception(
                f"The input dataset `ds` has more than two dimensions: "
                "{list(ds.dims.keys())}. Please select a single observation "
                "using e.g. `index=0`, or enable faceted plotting by adding "
                'the arguments e.g. `col="time", col_wrap=4` to the function '
                "call"
            )
        da = da.compute()
        img = da.plot.imshow(
            robust=robust, col_wrap=col_wrap, **aspect_size_kwarg, **kwargs
        )

    # If values provided for `index`, extract corresponding observations and
    # plot as either single image or facet plot
    else:

        # If a float is supplied instead of an integer index, raise exception
        if isinstance(index, float):
            raise Exception(
                f"Please supply `index` as either an integer or a list of " "integers"
            )

        # If col argument is supplied as well as `index`, raise exception
        if "col" in kwargs:
            raise Exception(
                f"Cannot supply both `index` and `col`; please remove one and "
                "try again"
            )

        # Convert index to generic type list so that number of indices supplied
        # can be computed
        index = index if isinstance(index, list) else [index]

        # Select bands and observations and convert to DataArray
        da = ds[bands].isel(**{index_dim: index}).to_array().compute()

        # If percentile_stretch == True, clip plotting to percentile vmin, vmax
        if percentile_stretch:
            vmin, vmax = da.compute().quantile(percentile_stretch).values
            kwargs.update({"vmin": vmin, "vmax": vmax})

        # If multiple index values are supplied, plot as a faceted plot
        if len(index) > 1:

            img = da.plot.imshow(
                robust=robust,
                col=index_dim,
                col_wrap=col_wrap,
                **aspect_size_kwarg,
                **kwargs,
            )

        # If only one index is supplied, squeeze out index_dim and plot as a
        # single panel
        else:

            img = da.squeeze(dim=index_dim).plot.imshow(
                robust=robust, **aspect_size_kwarg, **kwargs
            )

    # If an export path is provided, save image to file. Individual and
    # faceted plots have a different API (figure vs fig) so we get around this
    # using a try statement:
    if savefig_path:

        print(f"Exporting image to {savefig_path}")

        try:
            img.fig.savefig(savefig_path, **savefig_kwargs)
        except:
            img.figure.savefig(savefig_path, **savefig_kwargs)


def display_map(x, y, crs="EPSG:4326", margin=-0.5, zoom_bias=0):
    """
    Given a set of x and y coordinates, this function generates an
    interactive map with a bounded rectangle overlayed on Google Maps
    imagery.

    Modified from function written by Otto Wagner available here:
    https://github.com/ceos-seo/data_cube_utilities/tree/master/data_cube_utilities

    Parameters
    ----------
    x : (float, float)
        A tuple of x coordinates in (min, max) format.
    y : (float, float)
        A tuple of y coordinates in (min, max) format.
    crs : string, optional
        A string giving the EPSG CRS code of the supplied coordinates.
        The default is 'EPSG:4326'.
    margin : float
        A numeric value giving the number of degrees lat-long to pad
        the edges of the rectangular overlay polygon. A larger value
        results more space between the edge of the plot and the sides
        of the polygon. Defaults to -0.5.
    zoom_bias : float or int
        A numeric value allowing you to increase or decrease the zoom
        level by one step. Defaults to 0; set to greater than 0 to zoom
        in, and less than 0 to zoom out.
    Returns
    -------
    folium.Map : A map centered on the supplied coordinate bounds. A
    rectangle is drawn on this map detailing the perimeter of the x, y
    bounds.  A zoom level is calculated such that the resulting
    viewport is the closest it can possibly get to the centered
    bounding rectangle without clipping it.
    """

    # Convert each corner coordinates to lat-lon
    all_x = (x[0], x[1], x[0], x[1])
    all_y = (y[0], y[0], y[1], y[1])
    all_longitude, all_latitude = transform(Proj(crs), Proj("EPSG:4326"), all_x, all_y)

    # Calculate zoom level based on coordinates
    lat_zoom_level = (
        _degree_to_zoom_level(min(all_latitude), max(all_latitude), margin=margin)
        + zoom_bias
    )
    lon_zoom_level = (
        _degree_to_zoom_level(min(all_longitude), max(all_longitude), margin=margin)
        + zoom_bias
    )
    zoom_level = min(lat_zoom_level, lon_zoom_level)

    # Identify centre point for plotting
    center = [np.mean(all_latitude), np.mean(all_longitude)]

    # Create map
    interactive_map = folium.Map(
        location=center,
        zoom_start=zoom_level,
        tiles="http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
        attr="Google",
    )

    # Create bounding box coordinates to overlay on map
    line_segments = [
        (all_latitude[0], all_longitude[0]),
        (all_latitude[1], all_longitude[1]),
        (all_latitude[3], all_longitude[3]),
        (all_latitude[2], all_longitude[2]),
        (all_latitude[0], all_longitude[0]),
    ]

    # Add bounding box as an overlay
    interactive_map.add_child(
        folium.features.PolyLine(locations=line_segments, color="red", opacity=0.8)
    )

    # Add clickable lat-lon popup box
    interactive_map.add_child(folium.features.LatLngPopup())

    return interactive_map

def map_shapefile(
    gdf,
    attribute,
    continuous=False,
    cmap="viridis",
    basemap=basemaps.Esri.WorldImagery,
    default_zoom=None,
    hover_col=True,
    **style_kwargs,
):
    """
    Plots a geopandas GeoDataFrame over an interactive ipyleaflet
    basemap, with features coloured based on attribute column values.
    Optionally, can be set up to print selected data from features in
    the GeoDataFrame.
    Last modified: February 2020
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the spatial features to be plotted
        over the basemap.
    attribute: string, required
        An required string giving the name of any column in the
        GeoDataFrame you wish to have coloured on the choropleth.
    continuous: boolean, optional
        Whether to plot data as a categorical or continuous variable.
        Defaults to remapping the attribute which is suitable for
        categorical data. For continuous data set `continuous` to True.
    cmap : string, optional
        A string giving the name of a `matplotlib.cm` colormap that will
        be used to style the features in the GeoDataFrame. Features will
        be coloured based on the selected attribute. Defaults to the
        'viridis' colormap.
    basemap : ipyleaflet.basemaps object, optional
        An optional `ipyleaflet.basemaps` object used as the basemap for
        the interactive plot. Defaults to `basemaps.Esri.WorldImagery`.
    default_zoom : int, optional
        An optional integer giving a default zoom level for the
        interactive ipyleaflet plot. Defaults to None, which infers
        the zoom level from the extent of the data.
    hover_col : boolean or str, optional
        If True (the default), the function will print  values from the
        GeoDataFrame's `attribute` column above the interactive map when
        a user hovers over the features in the map. Alternatively, a
        custom shapefile field can be specified by supplying a string
        giving the name of the field to print. Set to False to prevent
        any attributes from being printed.
    **style_kwargs :
        Optional keyword arguments to pass to the `style` paramemter of
        the `ipyleaflet.Choropleth` function. This can be used to
        control the appearance of the shapefile, for example 'stroke'
        and 'weight' (controlling line width), 'fillOpacity' (polygon
        transparency) and 'dashArray' (whether to plot lines/outlines
        with dashes). For more information:
        https://ipyleaflet.readthedocs.io/en/latest/api_reference/choropleth.html
    """

    def on_hover(event, id, properties):
        with dbg:
            text = properties.get(hover_col, "???")
            lbl.value = f"{hover_col}: {text}"

    # Verify that attribute exists in shapefile
    if attribute not in gdf.columns:
        raise ValueError(
            f"The `attribute` {attribute} does not exist "
            f"in the geopandas.GeoDataFrame. "
            f"Valid attributes include {gdf.columns.values}."
        )

    # If hover_col is True, use 'attribute' as the default hover attribute.
    # Otherwise, hover_col will use the supplied attribute field name
    if hover_col and (hover_col is True):
        hover_col = attribute

    # If a custom string if supplied to hover_col, check this exists
    elif hover_col and (type(hover_col) == str):
        if hover_col not in gdf.columns:
            raise ValueError(
                f"The `hover_col` field {hover_col} does "
                f"not exist in the geopandas.GeoDataFrame. "
                f"Valid attributes include "
                f"{gdf.columns.values}."
            )

    # Convert to WGS 84 and GeoJSON format
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    data_geojson = gdf_wgs84.__geo_interface__

    # If continuous is False, remap categorical classes for visualisation
    if not continuous:

        # Zip classes data together to make a dictionary
        classes_uni = list(gdf[attribute].unique())
        classes_clean = list(range(0, len(classes_uni)))
        classes_dict = dict(zip(classes_uni, classes_clean))

        # Get values to colour by as a list
        classes = gdf[attribute].map(classes_dict).tolist()

    # If continuous is True then do not remap
    else:

        # Get values to colour by as a list
        classes = gdf[attribute].tolist()

    # Create the dictionary to colour map by
    keys = gdf.index
    id_class_dict = dict(zip(keys.astype(str), classes))

    # Get centroid to focus map on
    lon1, lat1, lon2, lat2 = gdf_wgs84.total_bounds
    lon = (lon1 + lon2) / 2
    lat = (lat1 + lat2) / 2

    if default_zoom is None:

        # Calculate default zoom from latitude of features
        default_zoom = _degree_to_zoom_level(lat1, lat2, margin=-0.5)

    # Plot map
    m = Map(
        center=(lat, lon),
        zoom=default_zoom,
        basemap=basemap,
        layout=dict(width="800px", height="600px"),
    )

    # Define default plotting parameters for the choropleth map.
    # The nested dict structure sets default values which can be
    # overwritten/customised by `choropleth_kwargs` values
    style_kwargs = dict({"fillOpacity": 0.8}, **style_kwargs)

    # Get `branca.colormap` object from matplotlib string
    cm_cmap = cm.get_cmap(cmap, 30)
    colormap = branca.colormap.LinearColormap(
        [cm_cmap(i) for i in np.linspace(0, 1, 30)]
    )

    # Create the choropleth
    choropleth = Choropleth(
        geo_data=data_geojson,
        choro_data=id_class_dict,
        colormap=colormap,
        style=style_kwargs,
    )

    # If the vector data contains line features, they will not be
    # be coloured by default. To resolve this, we need to manually copy
    # across the 'fillColor' attribute to the 'color' attribute for each
    # feature, then plot the data as a GeoJSON layer rather than the
    # choropleth layer that we use for polygon data.
    linefeatures = any(
        x in ["LineString", "MultiLineString"] for x in gdf.geometry.type.values
    )
    if linefeatures:

        # Copy colour from fill to line edge colour
        for i in keys:
            choropleth.data["features"][i]["properties"]["style"][
                "color"
            ] = choropleth.data["features"][i]["properties"]["style"]["fillColor"]

        # Add GeoJSON layer to map
        feature_layer = GeoJSON(data=choropleth.data, style=style_kwargs)
        m.add_layer(feature_layer)

    else:

        # Add Choropleth layer to map
        m.add_layer(choropleth)

    # If a column is specified by `hover_col`, print data from the
    # hovered feature above the map
    if hover_col and not linefeatures:

        # Use cholopleth object if data is polygon
        lbl = ipywidgets.Label()
        dbg = ipywidgets.Output()
        choropleth.on_hover(on_hover)
        display(lbl)

    else:

        lbl = ipywidgets.Label()
        dbg = ipywidgets.Output()
        feature_layer.on_hover(on_hover)
        display(lbl)

    # Display the map
    display(m)

def _degree_to_zoom_level(l1, l2, margin=0.0):
    """
    Helper function to set zoom level for `display_map`
    """

    degree = abs(l1 - l2) * (1 + margin)
    zoom_level_int = 0
    if degree != 0:
        zoom_level_float = math.log(360 / degree) / math.log(2)
        zoom_level_int = int(zoom_level_float)
    else:
        zoom_level_int = 18
    return zoom_level_int
