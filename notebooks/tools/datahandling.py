"""
Functions for loading and handling data.
"""

# Import required packages
import warnings
import numpy as np
import xarray as xr
import datetime
import pytz

from collections import Counter
from datacube.utils import masking
from odc.algo import mask_cleanup
from copy import deepcopy
import odc.algo


def _dc_query_only(**kw):
    """
    Remove load-only parameters, the rest
    can be passed to Query
    Returns
    =======
    dict of query parameters
    """

    def _impl(
        measurements=None,
        output_crs=None,
        resolution=None,
        resampling=None,
        skip_broken_datasets=None,
        dask_chunks=None,
        fuse_func=None,
        align=None,
        datasets=None,
        progress_cbk=None,
        group_by=None,
        **query,
    ):
        return query

    return _impl(**kw)


def _common_bands(dc, products):
    """
    Takes a list of products and returns a list of measurements/bands
    that are present in all products
    Returns
    -------
    List of band names
    """
    common = None
    bands = None

    for p in products:
        p = dc.index.products.get_by_name(p)
        if common is None:
            common = set(p.measurements)
            bands = list(p.measurements)
        else:
            common = common.intersection(set(p.measurements))
    return [band for band in bands if band in common]


def load_ard(
    dc,
    products=None,
    min_gooddata=0.0,
    categories_to_mask_ls=dict(
        cloud="high_confidence", cloud_shadow="high_confidence"
    ),
    categories_to_mask_s2=[
        "cloud high probability",
        "cloud medium probability",
        "thin cirrus",
        "cloud shadows",
        "saturated or defective",
    ],
    categories_to_mask_s1=["invalid data"],
    mask_filters=None,
    mask_pixel_quality=True,
    ls7_slc_off=True,
    predicate=None,
    dtype="auto",
    verbose=True,
    **kwargs,
):
    """
    Loads analysis ready data.
    Loads and combines Landsat USGS Collections 2, Sentinel-2, and Sentinel-1 for
    multiple sensors (i.e. ls5t, ls7e, ls8c and ls9 for Landsat; s2a and s2b for Sentinel-2),
    optionally applies pixel quality masks, and drops time steps that
    contain greater than a minimum proportion of good quality (e.g. non-
    cloudy or shadowed) pixels.
    The function supports loading the following DE Africa products:
    Landsat:
        * ls5_sr ('sr' denotes surface reflectance)
        * ls7_sr
        * ls8_sr
        * ls9_sr
        * ls5_st ('st' denotes surface temperature)
        * ls7_st
        * ls8_st
        * ls9_st
    Sentinel-2:
        * s2_l2a
    Sentinel-1:
        * s1_rtc
    Last modified: Feb 2021
    Parameters
    ----------
    dc : datacube Datacube object
        The Datacube to connect to, i.e. `dc = datacube.Datacube()`.
        This allows you to also use development datacubes if required.
    products : list
        A list of product names to load data from. For example:
        * Landsat C2: ``['ls5_sr', 'ls7_sr', 'ls8_sr', 'ls9_sr']``
        * Sentinel-2: ``['s2_l2a']``
        * Sentinel-1: ``['s1_rtc']``
    min_gooddata : float, optional
        An optional float giving the minimum percentage of good quality
        pixels required for a satellite observation to be loaded.
        Defaults to 0.0 which will return all observations regardless of
        pixel quality (set to e.g. 0.99 to return only observations with
        more than 99% good quality pixels).
    categories_to_mask_ls : dict, optional
        An optional dictionary that is used to identify poor quality pixels
        for masking. This mask is used for both masking out low
        quality pixels (e.g. cloud or shadow), and for dropping
        observations entirely based on the `min_gooddata` calculation.
    categories_to_mask_s2 : list, optional
        An optional list of Sentinel-2 Scene Classification Layer (SCL) names
        that identify poor quality pixels for masking.
    categories_to_mask_s1 : list, optional
        An optional list of Sentinel-1 mask names that identify poor
        quality pixels for masking.
    mask_filters : iterable of tuples, optional
        Iterable tuples of morphological operations - ("<operation>", <radius>)
        to apply on mask, where:
        operation: string, can be one of these morphological operations:
            * ``'closing'``  = remove small holes in cloud - morphological closing
            * ``'opening'``  = shrinks away small areas of the mask
            * ``'dilation'`` = adds padding to the mask
            * ``'erosion'``  = shrinks bright regions and enlarges dark regions
        radius: int
        e.g. ``mask_filters=[('erosion', 5),("opening", 2),("dilation", 2)]``
    mask_pixel_quality : bool, optional
        An optional boolean indicating whether to apply the poor data
        mask to all observations that were not filtered out for having
        less good quality pixels than ``min_gooddata``. E.g. if
        ``min_gooddata=0.99``, the filtered observations may still contain
        up to 1% poor quality pixels. The default of ``False`` simply
        returns the resulting observations without masking out these
        pixels; ``True`` masks them and sets them to NaN using the poor data
        mask. This will convert numeric values to floating point values
        which can cause memory issues, set to False to prevent this.
    ls7_slc_off : bool, optional
        An optional boolean indicating whether to include data from
        after the Landsat 7 SLC failure (i.e. SLC-off). Defaults to
        ``True``, which keeps all Landsat 7 observations > May 31 2003.
    predicate : function, optional
        An optional function that can be passed in to restrict the
        datasets that are loaded by the function. A filter function
        should take a `datacube.model.Dataset` object as an input (i.e.
        as returned from `dc.find_datasets`), and return a boolean.
        For example, a filter function could be used to return True on
        only datasets acquired in January:
        ``dataset.time.begin.month == 1``
    dtype : string, optional
        An optional parameter that controls the data type/dtype that
        layers are coerced to after loading. Valid values: ''`native`'',
        ``'auto'``, ``'float{16|32|64}'``.
        When ``'auto'`` is used, the data will be
        converted to ``'float32'`` if masking is used, otherwise data will
        be returned in the native data type of the data. Be aware that
        if data is loaded in its native dtype, nodata and masked
        pixels will be returned with the data's native nodata value
        (typically ``-999``), not ``NaN``.
        NOTE: If loading Landsat, the data is automatically rescaled so
        'native' dtype will return a value error.
    verbose : bool, optional
        If True, print progress statements during loading
    **kwargs : dict, optional
        A set of keyword arguments to ``dc.load`` that define the
        spatiotemporal query used to extract data. This typically
        includes ``measurements``, ``x`, ``y``, ``time``, ``resolution``,
        ``resampling``, ``group_by`` and ``crs``. Keyword arguments can
        either be listed directly in the ``load_ard`` call like any
        other parameter (e.g. ``measurements=['red']``), or by
        passing in a query kwarg dictionary (e.g. ``**query``). For a
        list of possible options, see the ``dc.load`` documentation:
        https://datacube-core.readthedocs.io/en/latest/dev/api/generate/datacube.Datacube.load.html
    Returns
    -------
    combined_ds : xarray Dataset
        An xarray dataset containing only satellite observations that
        contains greater than `min_gooddata` proportion of good quality
        pixels.
    """

    #########
    # Setup #
    #########
    # prevent function altering original query object
    kwargs = deepcopy(kwargs)

    # We deal with `dask_chunks` separately
    dask_chunks = kwargs.pop("dask_chunks", None)
    requested_measurements = kwargs.pop("measurements", None)

    # Warn user if they combine lazy load with min_gooddata
    if verbose:
        if (min_gooddata > 0.0) and dask_chunks is not None:
            warnings.warn(
                "Setting 'min_gooddata' percentage to > 0.0 "
                "will cause dask arrays to compute when "
                "loading pixel-quality data to calculate "
                "'good pixel' percentage. This can "
                "slow the return of your dataset."
            )

    # Verify that products were provided and determine if Sentinel-2
    # or Landsat data is being loaded
    if not products:
        raise ValueError(
            "Please provide a list of product names to load data from. "
            "Valid options are: Landsat C2 SR: ['ls5_sr', 'ls7_sr', 'ls8_sr', 'ls9_sr'], or "
            "Landsat C2 ST: ['ls5_st', 'ls7_st', 'ls8_st', 'ls9_st'], or "
            "Sentinel-2: ['s2_l2a'], or"
            "Sentinel-1: ['s1_rtc'], or"
        )
    
    # convert products to list if user passed as a string
    if type(products) == str:
        products=[products]
    
    if all(["ls" in product for product in products]):
        product_type = "ls"
    elif all(["s2" in product for product in products]):
        product_type = "s2"
    elif all(["s1" in product for product in products]):
        product_type = "s1"

    # check if the landsat product is surface temperature
    st = False
    if (product_type == "ls") & (all(["st" in product for product in products])):
        st = True

    # Check some parameters before proceeding
    if (product_type == "ls") & (dtype == "native"):
        raise ValueError(
            "Cannot load Landsat bands in native dtype "
            "as values require rescaling which converts dtype to float"
        )

    if product_type == "ls":
        if any(k in categories_to_mask_ls for k in ("cirrus", "cirrus_confidence")):
            raise ValueError(
                "'cirrus' categories for the pixel quality mask"
                " are not supported by load_ard"
            )

    # If `measurements` are specified but do not include pixel quality bands,
    #  add these to `measurements` according to collection
    if product_type == "ls":
        if verbose:
            print("Using pixel quality parameters for USGS Collection 2")
        fmask_band = "pixel_quality"

    elif product_type == "s2":
        if verbose:
            print("Using pixel quality parameters for Sentinel 2")
        fmask_band = "SCL"

    elif product_type == "s1":
        if verbose:
            print("Using pixel quality parameters for Sentinel 1")
        fmask_band = "mask"

    measurements = requested_measurements.copy() if requested_measurements else None

    # define a list of acceptable aliases to load landsat. We can't rely on 'common'
    # measurements as native band names have the same name for different measurements.
    ls_aliases = ["pixel_quality", "radiometric_saturation"]
    if st:
        ls_aliases = [
            "surface_temperature",
            "surface_temperature_quality",
            "atmospheric_transmittance",
            "thermal_radiance",
            "emissivity",
            "emissivity_stddev",
            "cloud_distance",
            "upwell_radiance",
            "downwell_radiance",
        ] + ls_aliases
    else:
        ls_aliases = ["red", "green", "blue", "nir", "swir_1", "swir_2"] + ls_aliases

    if measurements is not None:
        if product_type == "ls":

            # check we aren't loading aerosol bands from LS8
            aerosol_bands = [
                "aerosol_qa",
                "qa_aerosol",
                "atmos_opacity",
                "coastal_aerosol",
                "SR_QA_AEROSOL",
            ]
            if any(b in aerosol_bands for b in measurements):
                raise ValueError(
                    "load_ard doesn't support loading aerosol or "
                    "atmospeheric opacity related bands "
                    "for Landsat, instead use dc.load()"
                )

            # check measurements are in acceptable aliases list for landsat
            if set(measurements).issubset(ls_aliases):
                pass
            else:
                raise ValueError(
                    "load_ard does not support all band aliases for Landsat, "
                    "use only the following band names to load Landsat data: "
                    + str(ls_aliases)
                )

    # Deal with "load all" case: pick a set of bands common across
    # all products
    if measurements is None:
        if product_type == "ls":
            measurements = ls_aliases
        else:
            measurements = _common_bands(dc, products)

    # If `measurements` are specified but do not include pq, add.
    if measurements:
        if fmask_band not in measurements:
            measurements.append(fmask_band)

    # Get list of data and mask bands so that we can later exclude
    # mask bands from being masked themselves (also handle the case of rad_sat)
    data_bands = [
        band
        for band in measurements
        if band not in (fmask_band, "radiometric_saturation")
    ]
    mask_bands = [band for band in measurements if band not in data_bands]

    #################
    # Find datasets #
    #################

    # Pull out query params only to pass to dc.find_datasets
    query = _dc_query_only(**kwargs)

    # Extract datasets for each product using subset of dcload_kwargs
    dataset_list = []

    # Get list of datasets for each product
    if verbose:
        print("Finding datasets")
    for product in products:

        # Obtain list of datasets for product
        if verbose:
            print(f"    {product}")

        if product_type == "ls":
            # handle LS seperately to S2/S1 due to collection_category
            # force the user to load Tier 1
            datasets = dc.find_datasets(
                product=product, collection_category='T1', **query
            )
        else:
            datasets = dc.find_datasets(product=product, **query)

        # Remove Landsat 7 SLC-off observations if ls7_slc_off=False
        if not ls7_slc_off and product in ["ls7_sr"]:
            if verbose:
                print("    Ignoring SLC-off observations for ls7")
            datasets = [
                i
                for i in datasets
                if i.time.begin < datetime.datetime(2003, 5, 31, tzinfo=pytz.UTC)
            ]

        # Add any returned datasets to list
        dataset_list.extend(datasets)

    # Raise exception if no datasets are returned
    if len(dataset_list) == 0:
        raise ValueError(
            "No data available for query: ensure that "
            "the products specified have data for the "
            "time and location requested"
        )

    # If predicate is specified, use this function to filter the list
    # of datasets prior to load (this now redundant as dc.load now supports
    # a predicate filter)
    if predicate:
        if verbose:
            print(f"Filtering datasets using filter function")
        dataset_list = [ds for ds in dataset_list if predicate(ds)]

    # Raise exception if filtering removes all datasets
    if len(dataset_list) == 0:
        raise ValueError("No data available after filtering with " "filter function")

    #############
    # Load data #
    #############

    # Note we always load using dask here so that
    # we can lazy load data before filtering by good data
    ds = dc.load(
        datasets=dataset_list,
        measurements=measurements,
        dask_chunks={} if dask_chunks is None else dask_chunks,
        **kwargs,
    )

    ####################
    # Filter good data #
    ####################

    # need to distinguish between products due to different
    # pq band properties

    # collection 2 USGS
    if product_type == "ls":
        mask, _ = masking.create_mask_value(
            ds[fmask_band].attrs["flags_definition"], **categories_to_mask_ls
        )
        
        pq_mask = (ds[fmask_band] & mask) != 0
        
        # only run if data bands are present 
        if len(data_bands) > 0: 
            
        # identify pixels that will become negative after rescaling (but not 0 values)
            invalid = (
                    ((ds[data_bands] < (-1.0 * -0.2 / 0.0000275)) & (ds[data_bands] > 0))
                    .to_array(dim="band")
                    .any(dim="band")
                    )

        #merge masks
        pq_mask = np.logical_or(pq_mask, pq_mask)

    # sentinel 2
    if product_type == "s2":
        pq_mask = odc.algo.enum_to_bool(mask=ds[fmask_band],
                                        categories=categories_to_mask_s2)
        
    # sentinel 1
    if product_type == "s1":
        pq_mask = odc.algo.enum_to_bool(mask=ds[fmask_band],
                                        categories=categories_to_mask_s1)

    # The good data percentage calculation has to load in all `fmask`
    # data, which can be slow. If the user has chosen no filtering
    # by using the default `min_gooddata = 0`, we can skip this step
    # completely to save processing time
    if min_gooddata > 0.0:

        # Compute good data for each observation as % of total pixels.
        # Inveerting the pq_mask for this because cloud=True in pq_mask
        # and we want to sum good pixels
        if verbose:
            print("Counting good quality pixels for each time step")
        data_perc = (~pq_mask).sum(axis=[1, 2], dtype="int32") / (
            pq_mask.shape[1] * pq_mask.shape[2]
        )

        keep = (data_perc >= min_gooddata).persist()

        # Filter by `min_gooddata` to drop low quality observations
        total_obs = len(ds.time)
        ds = ds.sel(time=keep)
        pq_mask = pq_mask.sel(time=keep)

        if verbose:
            print(
                f"Filtering to {len(ds.time)} out of {total_obs} "
                f"time steps with at least {min_gooddata:.1%} "
                f"good quality pixels"
            )

    # morpholigcal filtering on cloud masks
    if (mask_filters is not None) & (mask_pixel_quality):
        if verbose:
            print(f"Applying morphological filters to pq mask {mask_filters}")
        pq_mask = mask_cleanup(pq_mask, mask_filters=mask_filters)

    ###############
    # Apply masks #
    ###############

    # Generate good quality data mask
    mask = None
    if mask_pixel_quality:
        if verbose:
            print("Applying pixel quality/cloud mask")
        mask = pq_mask

    # Split into data/masks bands, as conversion to float and masking
    # should only be applied to data bands
    ds_data = ds[data_bands]
    ds_masks = ds[mask_bands]
    
    # Remove sentinel-2 pixels valued 1 (scene edges, terrain shadow)
    if product_type == "s2":
        valid_data_mask = (ds_data > 1).to_array(dim="band").all(dim="band")
        ds_data =  odc.algo.keep_good_only(ds_data, where=valid_data_mask)
        
    # Mask data if either of the above masks were generated
    if mask is not None:
        ds_data = odc.algo.erase_bad(ds_data, where=mask)
    
    # Automatically set dtype to either native or float32 depending
    # on whether masking was requested
    if dtype == "auto":
        dtype = "native" if mask is None else "float32"

    # Set nodata values using odc.algo tools to reduce peak memory
    # use when converting data dtype
    if dtype != "native":
        ds_data = odc.algo.to_float(ds_data, dtype=dtype)

    # Put data and mask bands back together
    attrs = ds.attrs
    ds = xr.merge([ds_data, ds_masks])
    ds.attrs.update(attrs)

    ###############
    # Return data #
    ###############

    # Drop bands not originally requested by user
    if requested_measurements:
        ds = ds[requested_measurements]

    # Apply the scale and offset factors to Collection 2 Landsat. We need
    # different factors for different bands. Also handle the case where
    # masking_pixel_quaity = False, in which case the dtype is still
    # in int, so we convert it to float
    if product_type == "ls":
        if verbose:
            print("Re-scaling Landsat C2 data")

        sr_bands = ["red", "green", "blue", "nir", "swir_1", "swir_2"]
        radiance_bands = ["thermal_radiance", "upwell_radiance", "downwell_radiance"]
        trans_emiss = ["atmospheric_transmittance", "emissivity", "emissivity_stddev"]
        qa = ["pixel_quality", "radiometric_saturation"]

        if mask_pixel_quality == False:
            # set nodata to NaNs before rescaling
            # in the case where masking hasn't already done this
            for band in ds.data_vars:
                if band not in qa:
                    ds[band] = odc.algo.to_f32(ds[band])

        for band in ds.data_vars:
            if band == "cloud_distance":
                ds[band] = 0.01 * ds[band]

            if band == "surface_temperature_quality":
                ds[band] = 0.01 * ds[band]

            if band in radiance_bands:
                ds[band] = 0.001 * ds[band]

            if band in trans_emiss:
                ds[band] = 0.0001 * ds[band]

            if band in sr_bands:
                ds[band] = 2.75e-5 * ds[band] - 0.2

            if band == "surface_temperature":
                ds[band] = ds[band] * 0.00341802 + 149.0

        # add back attrs that are lost during scaling calcs
        for band in ds.data_vars:
            ds[band].attrs.update(attrs)

    # If user supplied dask_chunks, return data as a dask array without
    # actually loading it in
    if dask_chunks is not None:
        if verbose:
            print(f"Returning {len(ds.time)} time steps as a dask array")
        return ds
    else:
        if verbose:
            print(f"Loading {len(ds.time)} time steps")
        return ds.compute()


def mostcommon_crs(dc, product, query):
    """
    Takes a given query and returns the most common CRS for observations
    returned for that spatial extent. This can be useful when your study
    area lies on the boundary of two UTM zones, forcing you to decide
    which CRS to use for your `output_crs` in `dc.load`.
    Parameters
    ----------
    dc : datacube Datacube object
        The Datacube to connect to, i.e. `dc = datacube.Datacube()`.
        This allows you to also use development datacubes if required.
    product : str
        A product name to load CRSs from
    query : dict
        A datacube query including x, y and time range to assess for the
        most common CRS
    Returns
    -------
    str
        A EPSG string giving the most common CRS from all datasets returned
        by the query above
    """

    # remove dask_chunks & align to prevent func failing
    # prevent function altering dictionary kwargs
    query = deepcopy(query)
    if "dask_chunks" in query:
        query.pop("dask_chunks", None)

    if "align" in query:
        query.pop("align", None)

    # List of matching products
    matching_datasets = dc.find_datasets(product=product, **query)

    # Extract all CRSs
    crs_list = [str(i.crs) for i in matching_datasets]

    # Identify most common CRS
    crs_counts = Counter(crs_list)
    crs_mostcommon = crs_counts.most_common(1)[0][0]

    # Warn user if multiple CRSs are encountered
    if len(crs_counts.keys()) > 1:

        warnings.warn(
            f"Multiple UTM zones {list(crs_counts.keys())} "
            f"were returned for this query. Defaulting to "
            f"the most common zone: {crs_mostcommon}",
            UserWarning,
        )

    return crs_mostcommon
