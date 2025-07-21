import cv2
import cv2.aruco as aruco

# --- Parameters ---
squares_x = 7                     # Squares along X
squares_y = 5                     # Squares along Y
square_length_m = 0.035          # Square length in meters (3.5 cm)
marker_length_m = 0.02625        # Marker length in meters (2.625 cm)
dpi = 300                         # Print DPI (dots per inch)
output_file = "charuco_board_scaled.png"

# --- Convert physical dimensions to pixels ---
inch_per_meter = 39.3701
square_length_inch = square_length_m * inch_per_meter
width_inch = squares_x * square_length_inch
height_inch = squares_y * square_length_inch

# Total pixel size at given DPI
width_px = int(width_inch * dpi)
height_px = int(height_inch * dpi)

# --- Create dictionary and board ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
board = aruco.CharucoBoard((squares_x, squares_y), square_length_m, marker_length_m, aruco_dict)

# --- Generate image ---
board_img = board.generateImage((width_px, height_px))

# --- Save to file ---
cv2.imwrite(output_file, board_img)
print(f"Charuco board saved to '{output_file}' at {dpi} DPI")
