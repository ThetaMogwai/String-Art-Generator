from sinogram import generate_sinogram_line
import numpy as np


def greedy_line_pursuit(original, colors, sinogram, angles, n_lines, contrast):
    
    '''Generate a string art representation from a sinogram

    At each iteration, selects the brightest point across all the channels of the sinogram,
    which corresponds to the lines that contributes the most to the current residual image.
    
    The sinogram of this line is then substracted from the sinogram progressively reducing
    the remaining signal of the sinogram.
    
    The selected projection line is converted into two nail coordinates located on the border 
    of the circular image and stored with its corresponding color channel.

    Parameters
    ----------
    original : ndarray, shape (diameter, diameter, colors)
        Original circular image used to determine the nail radius

    colors : list of str
        Color associated with each image channel

    sinogram : ndarray, shape (diameter, n_angles, colors)
        Multi-channel sinogram of the image

    angles : ndarray, shape (3, n_angles)
        Row 0 : projection angle in radian 
        Row 1 : np.cos(angle) 
        Row 2 : np.sin(angle)

    n_lines : int
        Maximum number of lines to generate
    
    darkness : inte
        Multiplier of the black channel for CMYK images, used to increase contrast and details

    Returns
    -------
    sinogram : ndarray, shape (diameter, diameter, colors)
        Residual sinogram after line extraction

    line_coords : list
        List containing the two endpoints of each generated line

    line_colors : list
        Color associated with each generated line
    '''


    def select_line(sinogram, angles, projection_length, image_radius, image_diameter, colors):

        ''' Extract the strongest line from the residual sinogram.

        The highest sinogram value is associated to the projection angle and detector position 
        where the residual image contains the most intensity. 
        
        This line is selected, its analytical Radon transform is removed from the sinogram, and 
        returns the coordinates of the endpoints of the line and the color channel from which it
        was removed.
        
        This function looks through all channels at the same time and never pick the same line
        twice, no matter the color channel. As a result it produces a list of lines with alternating
        colors and requires a lower number of lines 

        Parameters
        ----------
        sinogram : ndarray, shape (diameter, n_angles, colors)
            Current residual sinogram of the image after removing the previous lines

        angles : ndarray, shape (3, n_angles)
            Row 0 : projection angle in radian 
            Row 1 : np.cos(angle) 
            Row 2 : np.sin(angle)


        projection_length : ndarray, shape (diameter,)
            Length of each detector projection used for normalization.

        image_radius : int
            Radius of the circular image

        image_diameter : int
            Diameter of the circular image

        colors : list
            Color associated with each channel

        Returns
        -------
        sinogram : ndarray, shape (diameter, n_angles, colors)
            Updated residual sinogram.

        line_coordinates : list or None
            Two endpoints of the selected line

        line_color : str or None
            Color associated with the selected line
        '''
        
        # Step 1 : find the brightest line in the sinogram ====================
        
        # Locate the highest value in the sinogram to identify the correspond line
        max_index = np.argmax(sinogram)
        offset, theta, channel = np.unravel_index(max_index, sinogram.shape)
        
        # When the max value is 0, the sinogram is depleted and the image is fully explained
        if sinogram[offset, theta, channel] == 0:
            return sinogram, None, None
        
        # Prevent selecting the same line across all channels
        sinogram[offset, theta, :] = 0

        # Step 2 : remove the selected line contribution to the sinogram ======
        
        # convert sinogram indices to coordinates centered around 0
        offset = offset - image_radius
        
        sinogram_line = generate_sinogram_line(offset, theta, angles, image_diameter, image_radius, projection_length)
        
        # Subtract the line contribution from its color channel.
        # Negative residual values are removed because the sinogram value are always positive (sum of positive values)
        sinogram[:, :, channel] = sinogram[:, :, channel] - sinogram_line
        sinogram[:, :, channel] = sinogram[:, :, channel] * (sinogram[:, :, channel] > 0)    # ensure that all sinogram values stay positive
        
    
        # Step 3 : convert line parameters into nail coordinates ===============
        
        # A line is defined by its angle and minimal distance from the center.
        # The intersection with the circular border gives the two nail positions.
        nail_1 = angles[0,theta] - np.acos(offset / image_radius) 
        nail_2 = angles[0,theta] + np.acos(offset / image_radius)
        
        # Convert polar coordinates on the unit circle into cartesian coordinates used by matplotlib 
        # to plot the lines
        nail_1 = [np.cos(nail_1),np.sin(nail_1)]
        nail_2 = [np.cos(nail_2),np.sin(nail_2)]
        
        line_coordinates = [nail_1,nail_2]
        line_color = colors[channel]
        
        return sinogram, line_coordinates, line_color
    
    # Initialization ==========================================================

    # Turn 2D sinograms in 3D arrays to iterate through channels
    if len(sinogram.shape) == 2: 
        sinogram = np.expand_dims(sinogram, axis=2)
    
    
    # Step 1 : precompute variables ===========================================
    
    image_radius = original.shape[0] // 2
    image_diameter = 2 * image_radius + 1        
    
    # Projection lengths are the parallel chords lengths of the  circle, they are used
    # to normalize the analytical line projections
    projection_length = np.linspace(-image_radius, image_radius, image_diameter + 2)[1:-1]     # remove start and endpoint
    projection_length = 2 * np.sqrt(image_radius**2 - projection_length**2) 
    
    # Step 2 : increase sinogram intensity ====================================
    
    # Each line reduce the intensity of the residual sinogram, need to amplify it to make sure we can extract enough
    # lines from the sinogram. Amplifying it too much reduces quality.
    
    # Get the average intensity of the sinogram of a single line
    sinogram_line = generate_sinogram_line(0, angles.shape[1] // 2, angles, image_diameter, image_radius, projection_length)
    sinogram_line = sinogram_line.mean()
    
    # Compute the total intensity required to draw all lines, divide it by the current average 
    # itensity multiplied by the number of channels to find the multiplier
    multiplier = n_lines * sinogram_line / (sinogram.mean() * sinogram.shape[2])
    sinogram *= multiplier
    
    # For CMYK images, increase the grayscale intensity further to increase contrast
    if sinogram.shape[2] == 4:
        sinogram[:, :, 3] *= contrast
    
    # Step 3 : iterative line selection =======================================
     
    lines_coordinates = []
    lines_colors = []
    
    for i in range(n_lines):
        
        sinogram, line_coord, line_color = select_line(sinogram, angles, projection_length, image_radius, image_diameter, colors)
        
        # Stop if the sinogram is empty
        if line_coord == None:
            print(f' - Progress: 100%  (sinogram depleted : {i} lines)')
            break
        
        lines_coordinates.append(line_coord)
        lines_colors.append(line_color)
        
        # Update progress every 10% or at the end
        progress = (i + 1) / n_lines * 100
        if progress % 10 <= 0.001 or i == n_lines - 1:
            print(f' - Progress: {round(progress)}%')
    
    return sinogram, lines_coordinates, lines_colors


