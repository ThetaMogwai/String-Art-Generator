from PIL import Image
import numpy as np


def load_image(image_path, color, invert):
    
    """ Load image and convert it to requested color representation
    
    The original image is preserved for visualization, the working image is converted
    to the requested color mode and optionally has its grayscale component inverted.
    
    Parameters
    ----------
    image_path : string
        
    color : {"L", "RGB", "CMYK"}
        Desired color representation of the working image.
        
    invert : bool
        If True, invert the grayscale image or the black channel of a CMYK image
        
    Returns
    -------
    original : ndarray, shape (H, W, C) 'L': C = 1, 'RGB': C = 3
        Original image used for visualization
    
    image : ndarray, shape (H, W, C)    'L': C = 1, 'RGB': C = 3, 'CMYK': C = 4
        Working image in the requested color representation.
    """
    
    if color not in {"L", "RGB", "CMYK"}:
        raise ValueError(f'Unsupported color mode : {color}')
    
    image = Image.open(image_path)
    
    grayscale = np.array(image.convert('L'))
            
    if image.mode == 'RGB' and color in ['RGB', 'CMYK']:
        original = np.array(image.convert('RGB'))
        
        if color == 'RGB':
            image = original.copy()
            
        else:
            image = np.array(image.convert('CMYK'))
            
            # Replace CMYK black channel with the grayscale image to improve shading
            image[:, :, -1] = grayscale
    
    else:
        color = 'L'
        original = grayscale.copy()
        original = np.expand_dims(original, axis=2)
        
        image = original.copy()
    
    # invert grayscale image or the black channel of CMYK image
    if invert and color in {"L", "CMYK"}:
        image[:, :, -1] = 255 - image[:, :, -1]
    
    return original, image
        



def resize_image(image, original, size):
    
    """ Downscale an image by block averaging.
    
    Downscale image if it'smallest dimension exceeds the size targe, by averaging 
    square blocks to reduce image resolution
    
    Parameters
    ----------
    image : ndarray, shape (H, W, C)
    
    original : ndarray, shape (H, W, C)
    
    size : int
        Maximum size of the smallest image dimension
    
    Returns
    -------
    original : ndarray
        Resized original image
    
    image : ndarray
        Resized working image
    """
    
    def compress_channel(image, factor):
    
        h = image.shape[0] // factor
        w = image.shape[1] // factor
        
        # Crop the image so that the dimensions are divisible by compression factor
        cropped = image[:h*factor, :w*factor]
        
        # Average each factor × factor block into a single pixel.
        return cropped.reshape(h, factor, w, factor).mean(axis=(1, 3))
    
    min_dimension = min(image.shape[:2])
    original_pixels = np.prod(image.shape[:2])
    
    if min_dimension <= size:
        return original, image
    
    else:
        factor = max(1, min_dimension // size)
        
        image    = np.stack([compress_channel(image[:, :, i], factor)    for i in range(image.shape[2])],    axis=2)
        original = np.stack([compress_channel(original[:, :, i], factor) for i in range(original.shape[2])], axis=2)
    
        reduction  = (1 - np.prod(image.shape[:2]) / original_pixels)
        print(f'Image resized by a factor {factor} to {image.shape[0:2]}, reducing pixel count by {reduction:.1%} \n')
    
        return original, image



def image_geometry(image, original, circular):
    
    """ Prepare image for Radon transform calculations

    Generates centered coordinate grids used for vactorized projection calculations
    Optionally center and crop the image to a squere and applies a circular mask

    Parameters
    ----------
    image : ndarray, shape (H, W, C)

    original : ndarray, shape (H, W, C)

    circular : bool
        If True, center, crop and apply a circular mask

    Returns
    -------
    image : ndarray, shape (H, W, C)

    original : ndarray, shape (H, W, C)
        
    x_coord : ndarray, shape (H, W)
        Pixels x-coordinates centered on the image origin

    y_coord : ndarray, shape (H, W)
        Pixels y-coordinates centered on the image origin
        
    mask : ndarray, shape (H, W), dtype=bool
        Circular mask applied to the image. If ``circular=False``, the mask is
        an array of ones
    """
    
    height, width = image.shape[0:2]
    
    radius = min(height, width) // 2
    center_y = height // 2
    center_x = width // 2
    
    if circular:
        # Crop both images to the largest centered square
        image = image[center_y - radius : center_y + radius + 1,    # Crop and center image along the y axis
                      center_x - radius : center_x + radius + 1,:]  # Crop and center image along the x axis

        original = original[center_y - radius : center_y + radius + 1,
                            center_x - radius : center_x + radius + 1,:]
                
        # New image center is the center of the square
        center_x = radius
        center_y = radius
        
        height, width = image.shape[0:2]

    # Generate coordinate grids centered on the image center because the Radon transform
    # uses distances relative to the center (not pixel indexes)
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    x_coord = x_coords - center_x
    y_coord = center_y - y_coords   
    
    
    if circular:
        # Compute distance from pixel to center and remove pixels outside the circle
        distance = np.sqrt(x_coord**2 + y_coord**2)
        mask = distance <= radius
        
        image = image * mask[:,:,None]
        original = original * mask[:,:,None] 
        
    else:
        mask = np.ones((height, width), dtype=bool)
    
    return original, image, x_coord, y_coord, mask




def angle_preparation(n_projection):
    
    """ Generates projection angles
    
    Generates projection angles in radian, precompute cos and sin for vectorized calculations

    Parameters
    ----------
    n_projection : int
        Number of projections used for the Radon transform

    Returns
    -------
    angles : ndarray, shape (3, n_projection)
        Row 0 : projection angles in radian in [0, pi[
        Row 1 : np.cos(angle) 
        Row 2 : np.sin(angle)
    """
    
    angles = np.linspace(0, 180, n_projection, endpoint=False)
    angles = np.radians(angles)
    
    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)
    
    angles = np.stack((angles, cos_theta, sin_theta), axis=0)
    
    return angles