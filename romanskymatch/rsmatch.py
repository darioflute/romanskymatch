# Routine for computing cubes of interpolated images

def reprojectAll(
    input_data,
    output_projection,
    shape_out=None,
    reproject_function=None,
    output_arrays=None,
    output_footprints=None,
    **kwargs,
):
    """
    Given a set of input images, reproject these to a single cube.

    This currently only works with 2-d images with celestial WCS.

    Parameters
    ----------
    input_data : iterable
        One or more input datasets to reproject and co-add. This should be an
        iterable containing one entry for each dataset, where a single dataset
        is one of:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * A tuple where the first element is an `~numpy.ndarray` and the
              second element is either a `~astropy.wcs.WCS` or a
              `~astropy.io.fits.Header` object
            * An `~astropy.nddata.NDData` object from which the ``.data`` and
              ``.wcs`` attributes will be used as the input data.

    output_projection : `~astropy.wcs.WCS` or `~astropy.io.fits.Header`
        The output projection, which can be either a `~astropy.wcs.WCS`
        or a `~astropy.io.fits.Header` instance.
    shape_out : tuple, optional
        If ``output_projection`` is a `~astropy.wcs.WCS` instance, the
        shape of the output data should be specified separately.
    input_weights : iterable
        If specified, this should be an iterable with the same length as
        ``input_data``, where each item is one of:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * An `~numpy.ndarray` array

    hdu_in : int or str, optional
        If one or more items in ``input_data`` is a FITS file or an
        `~astropy.io.fits.HDUList` instance, specifies the HDU to use.
    hdu_weights : int or str, optional
        If one or more items in ``input_weights`` is a FITS file or an
        `~astropy.io.fits.HDUList` instance, specifies the HDU to use.
    reproject_function : callable
        The function to use for the reprojection.
    combine_function : { 'mean', 'sum', 'median', 'first', 'last', 'min', 'max' }
        The type of function to use for combining the values into the final
        image. For 'first' and 'last', respectively, the reprojected images are
        simply overlaid on top of each other. With respect to the order of the
        input images in ``input_data``, either the first or the last image to
        cover a region of overlap determines the output data for that region.
    match_background : bool
        Whether to match the backgrounds of the images.
    background_reference : `None` or `int`
        If `None`, the background matching will make it so that the average of
        the corrections for all images is zero. If an integer, this specifies
        the index of the image to use as a reference.
    output_array : array or None
        The final output array.  Specify this if you already have an
        appropriately-shaped array to store the data in.  Must match shape
        specified with ``shape_out`` or derived from the output projection.
    output_footprint : array or None
        The final output footprint array.  Specify this if you already have an
        appropriately-shaped array to store the data in.  Must match shape
        specified with ``shape_out`` or derived from the output projection.
    **kwargs
        Keyword arguments to be passed to the reprojection function.

    Returns
    -------
    array : `~numpy.ndarray`
        The co-added array.
    footprint : `~numpy.ndarray`
        Footprint of the co-added array. Values of 0 indicate no coverage or
        valid values in the input image, while values of 1 indicate valid
        values.
    """

    import numpy as np
    from astropy.wcs import WCS
    from astropy.wcs.wcsapi import SlicedLowLevelWCS
    from reproject.utils import parse_input_data, parse_input_weights, parse_output_projection


    # Validate inputs

    if reproject_function is None:
        raise ValueError(
            "reprojection function should be specified with the reproject_function argument"
        )

    # Parse the output projection to avoid having to do it for each

    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)

    if output_arrays is not None and output_array.shape != shape_out:
        raise ValueError(
            "If you specify an output array, it must have a shape matching "
            f"the output shape {shape_out}"
        )
    if output_footprints is not None and output_footprint.shape != shape_out:
        raise ValueError(
            "If you specify an output footprint array, it must have a shape matching "
            f"the output shape {shape_out}"
        )

    # Start off by reprojecting individual images to the final projection

    ndata = len(input_data)
    output_arrays = np.ones((ndata,shape_out[0],shape_out[1])) * np.nan
    output_footprints = np.zeros((ndata,shape_out[0],shape_out[1]))

    hdu_in = None
    for idata in range(ndata):
        # We need to pre-parse the data here since we need to figure out how to
        # optimize/minimize the size of each output tile (see below).
        array_in, wcs_in = parse_input_data(input_data[idata], hdu_in=hdu_in)
        ny, nx = array_in.shape
        n_per_edge = 11
        xs = np.linspace(-0.5, nx - 0.5, n_per_edge)
        ys = np.linspace(-0.5, ny - 0.5, n_per_edge)
        xs = np.concatenate((xs, np.full(n_per_edge, xs[-1]), xs, np.full(n_per_edge, xs[0])))
        ys = np.concatenate((np.full(n_per_edge, ys[0]), ys, np.full(n_per_edge, ys[-1]), ys))
        xc_out, yc_out = wcs_out.world_to_pixel(wcs_in.pixel_to_world(xs, ys))

        # Determine the cutout parameters

        # In some cases, images might not have valid coordinates in the corners,
        # such as all-sky images or full solar disk views. In this case we skip
        # this step and just use the full output WCS for reprojection.

        if np.any(np.isnan(xc_out)) or np.any(np.isnan(yc_out)):
            imin = 0
            imax = shape_out[1]
            jmin = 0
            jmax = shape_out[0]
        else:
            imin = max(0, int(np.floor(xc_out.min() + 0.5)))
            imax = min(shape_out[1], int(np.ceil(xc_out.max() + 0.5)))
            jmin = max(0, int(np.floor(yc_out.min() + 0.5)))
            jmax = min(shape_out[0], int(np.ceil(yc_out.max() + 0.5)))

        if imax < imin or jmax < jmin:
            continue

        if isinstance(wcs_out, WCS):
            wcs_out_indiv = wcs_out[jmin:jmax, imin:imax]
        else:
            wcs_out_indiv = SlicedLowLevelWCS(
                wcs_out.low_level_wcs, (slice(jmin, jmax), slice(imin, imax))
            )

        shape_out_indiv = (jmax - jmin, imax - imin)

        # TODO: optimize handling of weights by making reprojection functions
        # able to handle weights, and make the footprint become the combined
        # footprint + weight map

        array, footprint = reproject_function(
            (array_in, wcs_in),
            output_projection=wcs_out_indiv,
            shape_out=shape_out_indiv,
            hdu_in=hdu_in,
            **kwargs,
        )
        print(idata, end=' ')
        output_arrays[idata,jmin:jmax,imin:imax]=array
        output_footprints[idata,jmin:jmax,imin:imax]=footprint
    print('')

    return output_arrays, output_footprints

     
    

def computeOffsets(images, footprints, sigmas):
    """
    Input:

    images, cube of interpolated images
    footprints, cube of footprints
    sigmas, cube of uncertainties

    Output:

    offsets, array of offsets
    
    """
    import numpy as np
    import scipy as sp

    nfiles = len(images)
    A = np.zeros((nfiles, nfiles))
    I = np.zeros((nfiles, nfiles))
    B = np.zeros(nfiles)

    for i in range(nfiles-1):
        sigma2_i = sigmas[i]**2
        # Create subcubes
        idx = np.where(footprints[i]>0)
        imin,imax = np.min(idx[0]), np.max(idx[0])
        jmin,jmax = np.min(idx[1]), np.max(idx[1])
        ifootprints = footprints[:,imin:imax,jmin:jmax]
        iimages = images[:,imin:imax,jmin:jmax]
        sigma2 = sigmas[:,imin:imax, jmin:jmax]**2
        sigma2_i = sigma2[i]
        image_i = iimages[i]
        print(' [', i,']: ', end='')
        for j in range(i+1, nfiles):
            idx = np.where((ifootprints[i]  == 1) & (ifootprints[j]  == 1))
            if np.sum(idx) > 0:
                print(j, end=',')
                sigma2_j = sigma2[j]
                image_j = iimages[j]
                A[i,j] = - np.sum(1/(sigma2_i[idx] + sigma2_j[idx]))
                A[j,i] = A[i,j]
                I[i,j] = - np.sum((image_i[idx]-image_j[idx])/(sigma2_i[idx] + sigma2_j[idx]))
                I[j,i] = - I[i,j]
    #for i in range(nfiles-1):
    for i in range(nfiles):
        A[i,i] = -np.sum(A[i,:])
        B[i] = np.sum(I[i,:])

    # Add noise to avoid singular matrix error
    noise = 1e-15*np.random.rand(nfiles, nfiles)
    A += noise
    
    # Put last epsilon to zero - does this really work ?
    #A[nfiles-1,nfiles-1] = 1
    #B[nfiles-1] = 0
   
    # Solve the linear system
    #epsilon = np.linalg.solve(A, B)
    epsilon = sp.linalg.solve(A, B, assume_a='sym')
    # epsilon = sp.sparse.linalg.spsolve(A, B)  # works better fro sparse matrix

    # Implement the global shift minimizing the offsets
    delta = - np.nanmedian(epsilon)
    print('\n Common shift ', delta)
    epsilon += delta

    return epsilon


def computeUncertainties(images, footprints, shape_out):
    import numpy as np

    nfiles = len(images)
    sigmas = np.zeros((nfiles,shape_out[0],shape_out[1]))
    #noise = []
    #zerolev = []
    for sigma, image, footprint in zip(sigmas, images, footprints):
        print('.',end=' ')
        iidx = np.where(footprint>0)
        imin,imax = np.min(iidx[0]), np.max(iidx[0])
        jmin,jmax = np.min(iidx[1]), np.max(iidx[1])
        ifootprints = footprints[:,imin:imax,jmin:jmax]
        iimage = image[imin:imax,jmin:jmax]
        ifootprint = footprint[imin:imax,jmin:jmax]
        idx = np.where(ifootprint)
        data = iimage[idx]
        idx = np.isfinite(data)
        med = np.nanmedian(data[idx])
        mad = np.nanmedian(np.abs(data[idx]-med))
        for k in range(10):
            residuals = data - med
            idx = np.where(np.abs(residuals) < 3 * mad)
            med = np.nanmedian(data[idx])
            mad = np.nanmedian(np.abs(data[idx]-med))
        #noise.append(mad)
        #zerolev.append(med)
        sigma[iidx] = mad
        #sigma[iidx] = 1
 
    return sigmas
