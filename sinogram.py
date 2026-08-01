import numpy as np


def generate_sinogram_image(image, x_coord, y_coord, angles, circular):
    
    """ Radon transform / sinogram of an image
    
    Each pixel is projected onto the detector for every projection angle.
    Pixel projections are linearly interpolated between adjacent detector bins
    Pixel values are then accumulated in the corresponding detector bins
    
    Parameters
    ----------
    image : ndarray, shape (height, width)
        Single channel image, values in [0, 255]
        
    x_coord : ndarray, shape (height, width)
        x-coordinates of pixels relative to the center of the image
        
    y_coord : ndarray, shape (height, width)
        y-coordinates of pixels relative to the center of the image
        
    angles : ndarray, shape (3, n_angles)
        Row 0 : projection angle in radian 
        Row 1 : np.cos(angle) 
        Row 2 : np.sin(angle)
    
    circular : bool
        If True, crop the sinogram and compute average intensity
    
    Returns
    -------
    sinogram : ndarray, shape (s_diameter, n_angles)
        Sinogram of the channel of an image, values in [0, 255]
    """
 
    height, width = image.shape[:2]
    
    center_y = height // 2
    center_x = width // 2
    
    # The detector must be large enough to fit the image diagonal
    s_radius = np.ceil(np.sqrt(center_x**2 + center_y**2)).astype('int')
    s_diameter = 2 * s_radius + 1
    
    sinogram = np.zeros((s_diameter, angles.shape[1], image.shape[2]))
   
    cos, sin = angles[1:, :]
    
    # In testings vectorizing additional dimensions (channels and angles) didn't improve the performance
        

    for i in range(angles.shape[1]):
    
        # Step 1 : Projection  ================================================
        # Compute on which detector bin each pixel along an angled line is projected
        
        projection = x_coord * cos[i] + y_coord * sin[i]  # continuous projection coordinate
        projection += s_radius                            # convert from [-R,R] to [0,2R]
        
        # Step 2 : Interpolation  =============================================
        # Detector bins (row of the sinogram) have integer indices, so continuous projection coordinates
        # are linearly interpolated and their values distributed between adjacent detector bins
        
        lower_bin = np.floor(projection).astype(int)
        upper_bin = lower_bin + 1
        
        for j in range(image.shape[2]):
        
            weight = projection - lower_bin 
            lower_value = (1 - weight) * image[:, :, j]
            upper_value = weight * image[:, :, j]
    
            # Step 3 : Accumulation  ==============================================
            # Accumulate the interpolated pixel intensities into the corresponding detector bins.
            # np.bincount performs the accumulation much faster than a loop over every pixel
            
        
            sinogram[:, i, j] = (np.bincount(lower_bin.ravel(), weights=lower_value.ravel(), minlength=s_diameter) + 
                                 np.bincount(upper_bin.ravel(), weights=upper_value.ravel(), minlength=s_diameter))


    if circular :
        image_radius = image.shape[0] // 2
        image_diameter = 2 * image_radius + 1
        
        # For circular image, pixels are projected on the middle of the detector array, remove the distant detectors
        sinogram = sinogram[s_radius - image_radius : s_radius + image_radius + 1, :, :]
        
        # Projection lengths are the parallel chords lengths of the  circle, they are used
        # to normalize the analytical line projections
        projection_length = np.linspace(-image_radius, image_radius, image_diameter + 2)[1:-1]     # remove start and endpoint
        projection_length = 2 * np.sqrt(image_radius**2 - projection_length**2) 
        
        # Each projection line has a different length, longer lines will project more pixel on a detector bin
        # Dividing by the length gives the average intensity level on a line, removing the bias towards longer lines
        sinogram = sinogram / projection_length[:, None, None]
    
    return sinogram 

def generate_sinogram_line(offset, theta, angles, i_diameter, i_radius, projection_length):
    
    """ Analytical Radon transform / sinogram of a line inscribed in a circle 
    
    The projection of a line is proportional to 1 / |sin(Δ)|, where Δ is the angle between the line
    and the projection direction. The result is then restricted to the circular image domain
    The projection parallel to the line is handeld seperately (to avoid dividing by 0)
    
    Parameters
    ----------
    offset : int
        Signed perpendicular distance between the line and the origin
        
    theta : int
        Index of the angle of the line in degrees
        
    angles : ndarray, shape (3, n_angles)
        Row 0 : projection angle in radian 
        Row 1 : np.cos(angle) 
        Row 2 : np.sin(angle)
        
    i_diameter : int
        Diameter of the circular image = rows of the sinogram = nbr of detector bin

    i_radius : int
        Radius of the circular image : i_radius = i_diameter // 2
        
    length : ndarray, shape (i_diameter,)
        Length of the projection line for each detector bin
        
    Returns
    -------
    sinogram : ndarray, shape (i_diameter, n_angles)
        Sinogram of the line defined by (offset, theta), inscribed in a circle, values in [0, 255]
    """
    
    # Step 1 : prepare variables  =============================================
    
    # Signed perpendicular distance between each projection line and the origin
    s_offset = np.arange(-i_radius, i_radius + 1)[:, None]

    # Projection lines angle in radian. The projection direction parallel to the line (index thata) is
    # removed and handeld at the end (to avoid dividing by sin(0))
    s_angles = np.delete(angles, theta, axis=1)
    s_angles = s_angles[0,:]
    
    angle = angles[0,theta]
    
    delta = s_angles - angle        # Difference between the projection angle line's angle 
    sin_delta = np.sin(delta)
    
    # The analytical Radon transform is defined on an infinite line, whereas our image is limited to 
    # a circular domain. We use the intersection between the line and the projection line to know if the 
    # value lies inside the image circle.

    # Compute the squared distance drom the image center to the intersection between the line and the projection line
    intersect = (s_offset**2 + offset**2 - 2 * s_offset * offset * np.cos(delta)) / sin_delta**2
    
    # Step 2 : generate and filter sinogram  ==================================
    
    # The sinogram is filled with values 1 / |sin(Δ)| if the intersection between the line
    # and the projection line is inside the circle
    sinogram = (1 / abs(sin_delta)) * (intersect <= i_radius**2)

    # Each projection line has a different length, longer lines will project more pixel on a detector bin
    # Dividing by the length gives the average intensity level on a line, removing the bias towards longer lines
    sinogram = (sinogram / projection_length[:,None])
    
    # Step 3 : insert the missing projection angle  ===========================
    
    # When the line is parallel to the projection angle (Δ = 0), all pixels are projected on the same detector bin
    # The analytical expression diverges, so its value is set manually : 1 for this detector bin, 0 for the others
    missing = np.zeros((i_diameter,1))
    missing[i_radius + offset,0] = 1        
    
    sinogram = np.insert(sinogram, theta, missing[:,0], axis=1) # insert it in the sinogram
    
    # Convert normalized intensities to the same [0,255] range as the image sinogram
    sinogram *= 255
    
    return sinogram