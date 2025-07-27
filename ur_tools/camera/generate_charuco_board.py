# import cv2
# import cv2.aruco as aruco

# # --- Parameters ---
# squares_x = 7                     # Squares along X
# squares_y = 5                     # Squares along Y
# square_length_m = 0.035          # Square length in meters (3.5 cm)
# marker_length_m = 0.02625        # Marker length in meters (2.625 cm)
# dpi = 300                         # Print DPI (dots per inch)
# output_file = "charuco_board_scaled.png"

# # --- Convert physical dimensions to pixels ---
# inch_per_meter = 39.3701
# square_length_inch = square_length_m * inch_per_meter
# width_inch = squares_x * square_length_inch
# height_inch = squares_y * square_length_inch

# # Total pixel size at given DPI
# width_px = int(width_inch * dpi)
# height_px = int(height_inch * dpi)

# # --- Create dictionary and board ---
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
# board = aruco.CharucoBoard((squares_x, squares_y), square_length_m, marker_length_m, aruco_dict)

# # --- Generate image ---
# board_img = board.generateImage((width_px, height_px))

# # --- Save to file ---
# cv2.imwrite(output_file, board_img)
# print(f"Charuco board saved to '{output_file}' at {dpi} DPI")

import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_single_marker():
    """Generate a single ArUco marker (original functionality)"""
    # Parameters
    aruco_dict_type = aruco.DICT_4X4_100
    marker_id = 23
    marker_size_cm = 1.0
    dpi = 300
    side_pixels = int(marker_size_cm / 2.54 * dpi)

    # Generate marker
    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    marker_img = aruco.generateImageMarker(aruco_dict, marker_id, side_pixels)

    # Convert to PIL and save with correct DPI
    img_pil = Image.fromarray(marker_img)
    img_pil.save(f"aruco_{marker_id}_1cm.png", dpi=(dpi, dpi))

    print(f"Generated ArUco marker ID {marker_id} at {marker_size_cm}cm ({side_pixels}x{side_pixels}px @ {dpi}DPI)")

def generate_aruco_board():
    """Generate a board with multiple ArUco markers"""
    # Board parameters
    markers_x = 6                    # Number of markers horizontally
    markers_y = 6                    # Number of markers vertically
    marker_size_cm = 1.0            # Physical size of each marker in cm
    spacing_cm = 0.5                # Spacing between markers in cm
    dpi = 300                       # Print resolution
    aruco_dict_type = aruco.DICT_4X4_100
    
    # Calculate dimensions
    marker_size_pixels = int(marker_size_cm / 2.54 * dpi)
    spacing_pixels = int(spacing_cm / 2.54 * dpi)
    
    # Total board dimensions
    board_width_pixels = markers_x * marker_size_pixels + (markers_x - 1) * spacing_pixels
    board_height_pixels = markers_y * marker_size_pixels + (markers_y - 1) * spacing_pixels
    
    # Add margins (1cm on each side)
    margin_pixels = int(1.0 / 2.54 * dpi)
    total_width = board_width_pixels + 2 * margin_pixels
    total_height = board_height_pixels + 2 * margin_pixels
    
    # Create white background
    board_img = Image.new('RGB', (total_width, total_height), 'white')
    
    # Get ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    
    # Generate markers and place them on the board
    marker_id = 0
    for row in range(markers_y):
        for col in range(markers_x):
            # Generate marker
            marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
            marker_pil = Image.fromarray(marker_img)
            
            # Calculate position
            x = margin_pixels + col * (marker_size_pixels + spacing_pixels)
            y = margin_pixels + row * (marker_size_pixels + spacing_pixels)
            
            # Paste marker onto board
            board_img.paste(marker_pil, (x, y))
            
            # Add marker ID text below each marker
            try:
                # Try to use a default font, fallback to default if not available
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw = ImageDraw.Draw(board_img)
            text = f"ID: {marker_id}"
            
            # Get text size and center it below the marker
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = x + (marker_size_pixels - text_width) // 2
            text_y = y + marker_size_pixels + 5
            
            draw.text((text_x, text_y), text, fill='black', font=font)
            
            marker_id += 1
    
    # Save the board
    output_filename = f"aruco_board_{markers_x}x{markers_y}_{marker_size_cm}cm.png"
    board_img.save(output_filename, dpi=(dpi, dpi))
    
    # Print summary
    board_width_cm = total_width / dpi * 2.54
    board_height_cm = total_height / dpi * 2.54
    
    print(f"Generated ArUco board: {markers_x}x{markers_y} markers")
    print(f"Marker size: {marker_size_cm}cm ({marker_size_pixels}x{marker_size_pixels}px)")
    print(f"Board size: {board_width_cm:.1f}x{board_height_cm:.1f}cm ({total_width}x{total_height}px)")
    print(f"Saved as: {output_filename}")
    print(f"Marker IDs: 0 to {marker_id-1}")

if __name__ == "__main__":
    # Generate both single marker and board
    print("=== Generating single ArUco marker ===")
    generate_single_marker()
    
    print("\n=== Generating ArUco marker board ===")
    generate_aruco_board()

