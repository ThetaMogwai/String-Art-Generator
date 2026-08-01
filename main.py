import preprocessing as PREP
from sinogram import generate_sinogram_image
from string_art import greedy_line_pursuit
import visual as VISU


IMAGE_CONFIG = {
    'name': 'Parrot.jpg',         # Name of the image in input folder
    'color': 'L',            # Desired color mode : 'RGB', 'CMYK', 'L' (grayscale)
    'invert': True,             # Invert grayscale/CMYK channels
    'circular': True,           # Crop and apply circular mask
}

QUALITY_CONFIG = {
    'image_size': 200,          # Increase quality but takes longer, 200 by default
    'n_projections': 180,       # Increase quality but takes longer, 180 by default
    'n_lines': 6000,
    'contrast': 2,
}

OUTPUT_CONFIG = {
    'sinogram': False,
    'string_art': True,
    'comparison': False,
}

COLOR_CHANNELS = {
    1: ['black'],
    3: ['red', 'green', 'blue'],
    4: ['cyan', 'magenta', 'yellow', 'black'],
}


def main():
    
    if (OUTPUT_CONFIG['string_art'] or OUTPUT_CONFIG['comparison']) and not IMAGE_CONFIG['circular']:
        raise ValueError("'circular' = False \n String art generation requires 'circular' = True")
    
    # =========================================================================
    # Step 1 : preprocessing settings and generating variables ================

    image_path = 'input/' + IMAGE_CONFIG['name']
    
    original, image = PREP.load_image(image_path, IMAGE_CONFIG['color'], IMAGE_CONFIG['invert'])
    original, image = PREP.resize_image(image, original, QUALITY_CONFIG['image_size'])
    
    original, image, x_coord, y_coord, mask = PREP.image_geometry(image, original, IMAGE_CONFIG['circular'])
    
    angles = PREP.angle_preparation(QUALITY_CONFIG['n_projections'])
    
    colors = COLOR_CHANNELS[image.shape[2]]
    
    
    # =========================================================================
    # Step 2 : generate sinogram and string art ===============================
    
    # Compute the Radon transform for each color channel,
    # then combine them into a single multi-channel sinogram.
    
    print("Generating sinogram...")
    sinogram = generate_sinogram_image(image, x_coord, y_coord, angles, IMAGE_CONFIG['circular'])
    
    # Save a copy to display later
    original_sinogram = sinogram.copy()
    
    
    if OUTPUT_CONFIG['string_art'] or OUTPUT_CONFIG['comparison']:
        print(f'\nStarting generation of {QUALITY_CONFIG["n_lines"]} lines')
        sinogram, lines_coordinates, lines_colors = greedy_line_pursuit(original, colors, sinogram, angles, QUALITY_CONFIG['n_lines'], QUALITY_CONFIG['contrast'])

    
    # =======================================================================
    # Step 3 : save selected output =========================================
    
    if OUTPUT_CONFIG['sinogram']:
        
        if IMAGE_CONFIG['color'] == 'CMYK':
            original_sinogram = generate_sinogram_image(original, x_coord, y_coord, angles, IMAGE_CONFIG['circular'])

        fig_sinogram = VISU.plot_sinogram(original,original_sinogram, IMAGE_CONFIG['circular'], mask, log_scale=False)
        
        output_path = 'output/' + IMAGE_CONFIG['name'][:-4] + '_sinogram_' + IMAGE_CONFIG['color'] + '.png'
        fig_sinogram.savefig(output_path, transparent=True, facecolor='none', dpi=300, bbox_inches='tight')
        

    if OUTPUT_CONFIG['string_art']:
        fig_string_art = VISU.plot_string_art (lines_coordinates, lines_colors, mask)
        
        output_path = 'output/' + IMAGE_CONFIG['name'][:-4] + '_stringart_' + IMAGE_CONFIG['color'] + '_' + str(QUALITY_CONFIG['n_lines']) + '.png'
        fig_string_art.savefig(output_path, transparent=True, facecolor='none', dpi=400, bbox_inches='tight')
        
        
    if OUTPUT_CONFIG['comparison']:
        fig_comparison = VISU.plot_comparison(original, lines_coordinates, lines_colors, mask)
        
        output_path = 'output/' + IMAGE_CONFIG['name'][:-4] + '_comparison_' + IMAGE_CONFIG['color'] + '_' + str(QUALITY_CONFIG['n_lines']) + '.png'
        fig_comparison.savefig(output_path, transparent=True, facecolor='none', dpi=300, bbox_inches='tight')

    return



if __name__ == "__main__":
    main()