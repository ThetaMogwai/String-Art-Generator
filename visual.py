from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter


def rgba_image(image, mask, circular):

    """ Convert image to RGBA
    
    Parameters
    ----------
    image : ndarray, shape (H, W, C)
        Signed perpendicular distance between the line and the origin
     
    mask : ndarray, shape (H, W), dtype=bool
        Mask applied to the image as a transparency layer   
     
    circular : bool
        
    Returns
    -------
    image : ndarray, shape (H, W, 4)
        Image in RGBA format ready to be saved in .png to preserve transparency
    """

    # Step 1 : Normalize the image to [0, 1] ==================================
    
    min_val = min(np.min(image), 0)
    max_val = max(np.max(image), 255)
    image = (image - min_val) / (max_val - min_val)
    image = (image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8


    # Step 2 : Detect and convert CMYK arrays to RGB arrays ===================

    if len(image.shape) == 3 and np.shape(image)[2] == 4:   # Detect CMYK arrays
        image = Image.fromarray(image, mode='CMYK')         # Turn array in a PIL image
        image = image.convert('RGB')                        # Use PIL to convert image to RGB
        image = np.array(image)                             # Turn RGB image in a 3 channel array
        
    if image.shape[2] == 1 :
        image = np.dstack((image,)*3)
    
    # Step 3 : stack transparency layer to convert to RGBA ====================
    
    if circular:
        mask = (255 * mask).astype(np.uint8)
    else:
        mask = np.full(image.shape[:2],255).astype(np.uint8) 
    
    image = np.dstack((image,mask))
    
    return image




def plot_sinogram(image, sinogram, circular, mask, log_scale):
    """
    Create a side-by-side comparison between the original image on the left and its string art 
    representation on the right (displayed on a circular white background)

    Parameters
    ----------
    image : ndarray, shape (R, R, C)  
    sinogram : ndarray, shape (R, n_angles, C)  
    mask : ndarray, shape (H, W), dtype=bool
    log_scale = bool
       
    Returns
    -------
    fig : matplotlib.figure
        Figure containing the original image and it's sinogram
    """
    
    image = rgba_image(image, mask, circular)
    sinogram = rgba_image(sinogram, mask, circular=False)
    
    fig = plt.figure(figsize=(15,5), facecolor='black')
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], height_ratios=[1],left=0.03, right=0.95, top=0.9, bottom=0.15, hspace=0.5, wspace=0.15)

    axes = [fig.add_subplot(gs[0,0], facecolor='none'),
            fig.add_subplot(gs[0,1], facecolor='none')]

    [ax.tick_params(colors="white", rotation=0) for ax in axes]

    # Plotting image  =========================================================
    
    width = image.shape[1] // 2
    height = image.shape[0] // 2
    
    image = image[1:-1, 1:-1, :]
    image_extent = [-width, width, -height, height] 
    axes[0].imshow(image, aspect='equal', extent=image_extent)
    
    # Disable axes without removing ticks    
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    # Plotting sinogram  ======================================================
    
    if log_scale : sinogram = np.log(sinogram + 1) # using log scale for better contrast
    
    sinogram_radius = (sinogram.shape[0] - 1) // 2    
    sinogram_extent = [0, 180, sinogram_radius, -sinogram_radius]
    axes[1].imshow(sinogram, aspect='auto', cmap='gray', extent=sinogram_extent)
    
    axes[1].set_xlabel('Projection angle (degrees)', color='white', fontsize=10)
    axes[1].set_ylabel('Projection position (pixels)', color='white', fontsize=10)
    fig.patch.set_visible(False)
    
    return fig




def plot_string_art(lines_coordinates, lines_colors, mask):
    '''
    Plot the string art reconstruction on a large circular white background

    Parameters
    ----------
    lines_coordinates : list
    lines_colors : list
    mask : ndarray, shape (H, W), dtype=bool

    Returns
    -------
    fig : matplotlib.figure
        Figure containing the string art visualization
    '''
    
    # Create the figure with a dark background so the string art stands out.
    fig = plt.figure(figsize=(10, 10), facecolor='black')

    # Use a single subplot occupying the full figure.
    ax = fig.add_subplot(111, facecolor='none')
    ax.axis('off')

    # Keep the geometry square so the circle is not distorted.
    ax.set_aspect('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Build a soft circular background from the binary mask.
    # The padding avoids cutting the circle too abruptly at the edges.
    background = np.pad(mask, 2, mode='constant')
    background = gaussian_filter(background.astype(float), sigma=1)
    background = np.clip(background, 0, 1)
    
    # Expand the 2D background to RGBA so it can be drawn as an image.
    background = np.dstack((background,) * 4)
    ax.imshow(background, extent=[-1, 1, -1, 1])
    
    # The first lines of the list are the ones with the higher intensity, the lines are drawn 
    # on top of each other, draw lines in reverse order so the most important lines remains
    # clearly visibles on top of the figure
    lines = LineCollection(lines_coordinates[::-1], linewidths=0.4, alpha=0.25, colors=lines_colors[::-1], linestyles='solid')
    ax.add_collection(lines)
    

    return fig




def plot_comparison (image, lines_coordinates, lines_colors, mask):
    """
    Create a side-by-side comparison between the original image on the left and its string art 
    representation on the right (displayed on a circular white background)

    Parameters
    ----------
    image : ndarray, shape (H, W, C)    
    lines_coordinates : list
    lines_colors : list
    mask : ndarray, shape (H, W), dtype=bool
       
    Returns
    -------
    fig : matplotlib.figure
        Figure containing the original image and the string art visualization
    """
    
    image = rgba_image(image, mask, circular=True)
    
    fig = plt.figure(figsize=(15,7), facecolor='black')
    gs = fig.add_gridspec(1, 2, width_ratios=[1,1], height_ratios=[1],left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.1, wspace=0.2)

    axes = [fig.add_subplot(gs[0,0], facecolor='none'),
            fig.add_subplot(gs[0,1], facecolor='none')]

    [ax.axis('off') for ax in axes]

    # Plotting  image  ========================================================
    
    image = image[1:-1, 1:-1, :]
    axes[0].imshow(image, aspect='equal')

    # Plotting lines  =========================================================
       
    axes[1].set_aspect('equal')
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)
    
    # Build a soft circular background from the binary mask.
    # The padding avoids cutting the circle too abruptly at the edges.
    background = np.pad(mask, 2, mode="constant")
    background = gaussian_filter(background.astype(float), sigma=1)
    background = np.clip(background, 0, 1)

    # Expand the 2D background to RGBA so it can be drawn as an image.
    background = np.dstack((background,) * 4)
    axes[1].imshow(background, extent=[-1, 1, -1, 1])
    
    # The first lines of the list are the ones with the higher intensity, the lines are drawn 
    # on top of each other, draw lines in reverse order so the most important lines remains
    # clearly visibles on top of the figure
    lines = LineCollection(lines_coordinates[::-1], linewidths=0.4, alpha=0.25, colors=lines_colors[::-1], linestyles='solid')
    axes[1].add_collection(lines)
       
    return fig