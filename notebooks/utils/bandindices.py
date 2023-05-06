"""
Functions for computing remote sensing band indices
"""

# Import required packages
import warnings
import numpy as np


# Define custom functions
def calculate_indices(
        ds,
        index=None,
        satellite_mission=None,
        custom_varname=None,
        normalise=True,
        drop=False,
        deep_copy=True,
):
    """
    Takes an xarray dataset containing spectral bands, calculates one of
    a set of remote sensing indices, and adds the resulting array as a
    new variable in the original dataset.

    Last modified: July 2022

    Parameters
    ----------
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array with containing the
        spectral bands required to calculate the index. These bands are
        used as inputs to calculate the selected water index.
        
    index : str or list of strs
        A string giving the name of the index to calculate or a list of
        strings giving the names of the indices to calculate:

        * ``'NDCI'`` (Normalised Difference Chlorophyll Index, Mishra & Mishra, 2012)
        * ``'NDVI'`` (Normalised Difference Vegetation Index, Rouse 1973)
        * ``'IRECI'``(Inverted RedEdge Chlorophyll Index)
        * ``'MTCI'`` (MERIS Terrestrial Chlorophyll Index)
        * ``'OTCI'`` (OLCI Terrestrial Chlorophyll Index)
        * ``'MCARI'`` (Modified Chlorophyll Absortion Ratio Index)
        * ``'CI-RedEdge'`` (Chlorophyll Index Red Edge)
        * ``'CI-GreenEdge'`` (Chlorophyll Index Green Edge)
        * ``'TCARI'`` (Transformed Chlorophyll Absortion in Reflectance Index)
        * ``'OSAVI'`` (Optimized Soil Adjust Vegetation Index)
        
    satellite_mission : str
        An string that tells the function which satellite mission's data is
        being used to calculate the index. This is necessary because
        different satellite missions use different names for bands covering
        a similar spectra.

        Valid options are:

         * ``'ls'`` (for USGS Landsat)
         * ``'s2'`` (for Copernicus Sentinel-2)
         
    custom_varname : str, optional
        By default, the original dataset will be returned with
        a new index variable named after `index` (e.g. 'NDVI'). To
        specify a custom name instead, you can supply e.g.
        `custom_varname='custom_name'`. Defaults to None, which uses
        `index` to name the variable.
        
    normalise : bool, optional
        Some coefficient-based indices (e.g.)
        produce different results if surface reflectance values are not
        scaled between 0.0 and 1.0 prior to calculating the index.
        Setting `normalise=True` first scales values to a 0.0-1.0 range
        by dividing by 10000.0. Defaults to True.
        
    drop : bool, optional
        Provides the option to drop the original input data, thus saving
        space. If `drop=True`, returns only the index and its values.
        
    deep_copy: bool, optional
        If `deep_copy=False`, calculate_indices will modify the original
        array, adding bands to the input dataset and not removing them.
        If the calculate_indices function is run more than once, variables
        may be dropped incorrectly producing unexpected behaviour. This is
        a bug and may be fixed in future releases. This is only a problem
        when `drop=True`.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a
        new varible containing the remote sensing index as a DataArray.
        If drop = True, the new variable/s as DataArrays in the
        original Dataset.
    """

    # Set ds equal to a copy of itself in order to prevent the function
    # from editing the input dataset. This is to prevent unexpected
    # behaviour though it uses twice as much memory.
    if deep_copy:
        ds = ds.copy(deep=True)

    # Capture input band names in order to drop these if drop=True
    if drop:
        bands_to_drop = list(ds.data_vars)
        print(f"Dropping bands {bands_to_drop}")

    # Dictionary containing remote sensing index band recipes
    index_dict = {
        # Normalised Difference Vegation Index, Rouse 1973
        "NDVI": lambda ds: (ds.nir - ds.red) / (ds.nir + ds.red),
        # Normalised Difference Chlorophyll Index,(Mishra & Mishra, 2012)
        "NDCI": lambda ds: (ds.red_edge_1 - ds.red) / (ds.red_edge_1 + ds.red),
        # Inverted RedEdge Chlorophyll Index
        "IRECI": lambda ds: (ds.red_edge_3 - ds.red) * (ds.red_edge_2 / ds.red_edge_1),
        # MERIS Terrestrial Chlorophyll Index
        "MTCI": lambda ds: (ds.red_edge_2 - ds.red_edge_1) / (ds.red_edge_1 - ds.red),
        # OTCI OLCI Terrestrial Chlorophyll Index
        # (Interpolamos las bandas 4 y 5 para poder tener la longitud de onda deseada
        "OTCI": lambda ds: (ds.red_edge_2 - ds.red_edge_1) / (ds.red_edge_1 - (ds.red + ds.red_edge_1) / 2),
        # MCARI Modified Chlorophyll Absortion Ratio Index
        "MCARI": lambda ds: (ds.red_edge_1 - ds.red) - 0.2 * (ds.red_edge_1 - ds.green) / (ds.red_edge_1 / ds.red),
        # CI-RedEdge Chlorophyll Index Red Edge
        "CI_RedEdge": lambda ds: (ds.red_edge_3 / ds.red_edge_1) - 1,
        # CI-GreenEdge Chlorophyll Index Green Edge
        "CI_GreenEdge": lambda ds: (ds.red_edge_3 / ds.green) - 1,
    }

    # TCARI Transformed Chlorophyll Absortion in Reflectance Index
    def TCARI(ds):
        return 3 * (
                (ds.red_edge_1 - ds.red)
                - (0.2 * (ds.red_edge_1 - ds.green)
                   / (ds.red_edge_1 / ds.red)))

    index_dict["TCARI"] = TCARI

    # OSAVI Optimized Soil Adjust Vegetation Index
    def OSAVI(ds):
        return 1.16 * ((ds.red_edge_3 - ds.red) / (ds.red_edge_3 + ds.red + 0.16))

    index_dict["OSAVI"] = OSAVI

    def TCARI_OSAVI(ds):
        t = TCARI(ds)
        o = OSAVI(ds)
        return t / o

    index_dict["TCARI_OSAVI"] = TCARI_OSAVI

    # If index supplied is not a list, convert to list. This allows us to
    # iterate through either multiple or single indices in the loop below
    indices = index if isinstance(index, list) else [index]

    # calculate for each index in the list of indices supplied (indexes)
    for index in indices:

        # Select an index function from the dictionary
        index_func = index_dict.get(str(index))

        # If no index is provided or if no function is returned due to an
        # invalid option being provided, raise an exception informing user to
        # choose from the list of valid options
        if index is None:

            raise ValueError(
                f"No remote sensing `index` was provided. Please "
                "refer to the function \ndocumentation for a full "
                "list of valid options for `index` (e.g. 'NDVI')"
            )

        elif (
                index
                in [
                    ""
                ]
                and not normalise
        ):

            warnings.warn(
                f"\nA coefficient-based index ('{index}') normally "
                "applied to surface reflectance values in the \n"
                "0.0-1.0 range was applied to values in the 0-10000 "
                "range. This can produce unexpected results; \nif "
                "required, resolve this by setting `normalise=True`"
            )

        elif index_func is None:

            raise ValueError(
                f"The selected index '{index}' is not one of the "
                "valid remote sensing index options. \nPlease "
                "refer to the function documentation for a full "
                "list of valid options for `index`"
            )

        # Rename bands to a consistent format if depending on what satellite mission
        # is specified in `satellite_mission`. This allows the same index calculations
        # to be applied to all satellite missions. If no satellite mission was provided,
        # raise an exception.
        if satellite_mission is None:

            raise ValueError(
                "No `satellite_mission` was provided. Please specify "
                "either 'ls' or 's2' to ensure the \nfunction "
                "calculates indices using the correct spectral "
                "bands."
            )

        elif satellite_mission == "ls":
            sr_max = 1.0
            # Dictionary mapping full data names to simpler alias names
            # This only applies to properly-scaled "ls" data i.e. from
            # the Landsat geomedians. calculate_indices will not show 
            # correct output for raw (unscaled) Landsat data (i.e. default
            # outputs from dc.load)
            bandnames_dict = {
                "SR_B1": "blue",
                "SR_B2": "green",
                "SR_B3": "red",
                "SR_B4": "nir",
                "SR_B5": "swir_1",
                "SR_B7": "swir_2",
            }

            # Rename bands in dataset to use simple names (e.g. 'red')
            bands_to_rename = {
                a: b for a, b in bandnames_dict.items() if a in ds.variables
            }

        elif satellite_mission == "s2":
            sr_max = 10000
            # Dictionary mapping full data names to simpler alias names
            bandnames_dict = {
                "nir_1": "nir",
                "B02": "blue",
                "B03": "green",
                "B04": "red",
                "B05": "red_edge_1",
                "B06": "red_edge_2",
                "B07": "red_edge_3",
                "B08": "nir",
                "B11": "swir_1",
                "B12": "swir_2",
            }

            # Rename bands in dataset to use simple names (e.g. 'red')
            bands_to_rename = {
                a: b for a, b in bandnames_dict.items() if a in ds.variables
            }

        # Raise error if no valid satellite_mission name is provided:
        else:
            raise ValueError(
                f"'{satellite_mission}' is not a valid option for "
                "`satellite_mission`. Please specify either \n"
                "'ls' or 's2'"
            )

        # Apply index function
        try:
            # If normalised=True, divide data by 10,000 before applying func
            mult = sr_max if normalise else 1.0
            index_array = index_func(ds.rename(bands_to_rename) / mult)

        except AttributeError:
            raise ValueError(
                f"Please verify that all bands required to "
                f"compute {index} are present in `ds`."
            )

        # Add as a new variable in dataset
        output_band_name = custom_varname if custom_varname else index
        ds[output_band_name] = index_array

    # Once all indexes are calculated, drop input bands if drop=True
    if drop:
        ds = ds.drop(bands_to_drop)

    # Return input dataset with added index variable
    return ds
