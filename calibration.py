import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def enhance_frame(image, circle_size):
    '''Enhance calibration frame so the circles are more visible
    image: calibration image jpg loaded
    circle_size: window size used, should be around the diameter of the circles
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = torch.from_numpy(gray_image).float()
    kernel = torch.ones((circle_size, circle_size), dtype=torch.float32)
    kernel /= kernel.numel()
    averages_map = F.conv2d(
        gray_image.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=circle_size // 2,
    )
    averages_map = averages_map.squeeze()
    left = averages_map[2*circle_size:, circle_size:-circle_size]
    right = averages_map[:-2*circle_size, circle_size:-circle_size]
    up = averages_map[circle_size:-circle_size, 2*circle_size:]
    down = averages_map[circle_size:-circle_size, :-2*circle_size]
    foreground = averages_map[circle_size:-circle_size, circle_size:-circle_size]
    background = (left + right + up + down) / 4
    correlation = (background - foreground) / (background + 1e-6)
    correlation = correlation.clip(min=0)
    # we need zero pad the correlation map to the original image size
    correlation_final = torch.zeros_like(gray_image)
    correlation_final[circle_size:-circle_size, circle_size:-circle_size] = correlation
    correlation_final = correlation_final.numpy()
    return correlation_final


def filter_components(correlation_bool, threshold):
    '''Filter components based on correlation threshold
    correlation: correlation matrix
    threshold: threshold value
    '''
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        correlation_bool.astype(np.uint8),
        connectivity=8 # Or 4, depending on your needs
    )
    components = []
    for i in range(1, num_labels):
        h, w = stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_WIDTH]
        components.append((*centroids[i], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA] / (h * w)))

    components = np.array(components)
    mean_width = np.mean(components[:, 2])
    mean_height = np.mean(components[:, 3])
    mean_density = np.mean(components[:, 4])
    std_width = np.std(components[:, 2]) * threshold
    std_height = np.std(components[:, 3]) * threshold
    std_density = np.std(components[:, 4]) * threshold
    f = (components[:, 2] > mean_width - std_width) & (components[:, 2] < mean_width + std_width)
    f &= (components[:, 3] > mean_height - std_height) & (components[:, 3] < mean_height + std_height)
    f &= (components[:, 4] > mean_density - std_density) & (components[:, 4] < mean_density + std_density)
    filtered = components[f]
    centroids = torch.from_numpy(filtered[:, :2]).float()
    
    return centroids    


def filter_centroids_2(centroids, threshold):
    disp = torch.cdist(centroids, centroids)
    disp[disp == 0] = torch.inf
    values, indices = disp.min(0)
    f2 = ((values - values.mean())/values.std()).abs()<threshold
    centroids = centroids[f2]
    return centroids

def get_world_points(centroids, world_scale):
    disp = torch.cdist(centroids, centroids)
    disp[disp == 0] = torch.inf
    values, indices = disp.min(0)
    
    # --- PCA approach for rotation angle ---
    # 1. Center the data
    mean_centroid = torch.mean(centroids, dim=0)
    centered_centroids = centroids - mean_centroid
    
    # 2. Compute covariance matrix
    # Using numpy for covariance calculation as torch.cov requires specific input shapes
    cov_matrix = np.cov(centered_centroids.numpy().T) 
    
    # 3. Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Get the principal eigenvector (corresponding to the largest eigenvalue)
    principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    
    # 5. Calculate the angle
    # Angle of the principal component with respect to the positive x-axis
    rot_angle = np.arctan2(principal_eigenvector[1], principal_eigenvector[0])
    # --- End PCA approach ---

    # rotation matrix 
    t = torch.matmul(
        torch.tensor([[np.cos(rot_angle), -np.sin(rot_angle)], 
                      [np.sin(rot_angle), np.cos(rot_angle)]], dtype=centroids.dtype), # Ensure dtype matches
        centroids.T
    ).T

    i = -torch.ones_like(t)
    disp2 = disp.clone()
    adj = (disp2 > 0.5 * values.mean()) & (disp2 < 1.5 * values.mean())

    q = [0]
    visited = torch.zeros_like(disp2[0], dtype=torch.bool)
    visited[0] = True
    while len(q) > 0:
        current = q.pop(0)
        # get the neighbors
        neighbours = adj[current] & ~visited
        if neighbours.sum() == 0:
            continue
        # get the indices of the neighbours
        indices = torch.nonzero(neighbours).flatten()
        # if there are no neighbours, continue
        for index in indices:
            visited[index] = True
            # now check the displacement between the current point and the neighbour
            _disp = t[index] - t[current]
            _disp /= values.mean()
            _disp = _disp.round()
            # assign the displacement to the neighbour
            i[index] = i[current] + _disp
            # add the neighbour to the queue
            q.append(index)
    pixel_pts = centroids.numpy()
    world_pts = ((i - i.min(0).values) * world_scale).numpy()
    return pixel_pts, world_pts, rot_angle, t


def calibrate_pinhole_with_distortion(pixel_pts: np.ndarray,
                                      world_pts: np.ndarray,
                                      image_size: tuple
                                     ):
    """
    Calibrate a single-view pinhole camera + distortion from
    N correspondences on a planar target (Z=0).
    
    pixel_pts: (N,2) float32 image points
    world_pts: (N,2) float32 plane points (X,Y); we'll set Z=0
    image_size: (width, height) of your images
    """
    # Build objectPoints as (N,1,3) with Z=0
    objp = np.zeros((len(world_pts),1,3), dtype=np.float32)
    objp[:,:,0:2] = world_pts.reshape(-1,1,2)
    
    objpoints = [objp]                   # list of length 1
    imgpoints = [pixel_pts.reshape(-1,1,2).astype(np.float32)]
    
    # Flags: fix higher‑order terms if you want, tweak as needed
    flags = (cv2.CALIB_ZERO_TANGENT_DIST   # or allow tangential by omitting
           | cv2.CALIB_FIX_K3)             # fix k3 if you want only k1,k2
    
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )
    if not ret:
        raise RuntimeError("Calibration failed")
    
    # We only have one view, so rvecs[0], tvecs[0] is your plane pose.
    R, _ = cv2.Rodrigues(rvecs[0])
    T    = tvecs[0].reshape(3,1)
    return K, dist.ravel(), R, T


def pixel_to_world_via_plane(pixel_pts: np.ndarray,
                             K: np.ndarray,
                             dist: np.ndarray,
                             R: np.ndarray,
                             T: np.ndarray
                            ) -> np.ndarray:
    """
    Map distorted pixel_pts → world (X,Y) on your Z=0 plane,
    *exactly* recovering your training world_pts if you feed in those same pixel_pts.
    """
    # 1) undistort *into pixel coordinates* by passing P=K
    #    (so you still get u,v in pixels, not normalized coords)
    pix = pixel_pts.reshape(-1,1,2).astype(np.float32)
    undist_pix = cv2.undistortPoints(pix, K, dist, P=K)  # → shape (M,1,2)
    uv = undist_pix.reshape(-1,2)

    # 2) form homogeneous pixel coords [u,v,1]
    ones = np.ones((uv.shape[0],1), dtype=uv.dtype)
    uvh  = np.hstack([uv, ones])  # shape (M,3)

    # 3) build & invert your plane homography H_plane = K [r1 r2 t]
    H_plane = K @ np.hstack((R[:,0:2], T))  # (3×3)
    H_inv   = np.linalg.inv(H_plane)

    # 4) back‑project
    world_h = (H_inv @ uvh.T).T           # (M,3)
    world_h /= world_h[:, 2:3]           # normalize
    return world_h[:, :2]                # return X,Y


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os # For checking file existence

# --- Include necessary functions ---
# (Assuming these are defined as previously discussed or available from calibration.py)
# Note: Added error handling and checks based on previous steps.

def pixel_to_world_via_plane(pixel_pts: np.ndarray,
                             K: np.ndarray,
                             dist: np.ndarray,
                             R: np.ndarray,
                             T: np.ndarray
                            ) -> np.ndarray:
    """
    Map distorted pixel_pts -> world (X,Y) on your Z=0 plane.
    refers to functions adapted from calibration.py
    """
    # 1) Undistort *into pixel coordinates* by passing P=K
    pix = pixel_pts.reshape(-1, 1, 2).astype(np.float32)
    if dist is not None and dist.ndim == 1:
        dist = dist.reshape(1, -1) # Ensure dist has shape (1, k) or (k,) for OpenCV

    undist_pix = cv2.undistortPoints(pix, K, dist, P=K) # # -> shape (M,1,2)
    if undist_pix is None:
         raise ValueError("cv2.undistortPoints returned None. Check K and dist shapes/values.")
    uv = undist_pix.reshape(-1, 2) #

    # 2) Form homogeneous pixel coords [u,v,1]
    ones = np.ones((uv.shape[0], 1), dtype=uv.dtype)
    uvh = np.hstack([uv, ones])  # shape (M,3)

    # 3) Build & invert your plane homography H_plane = K [r1 r2 t]
    H_plane = K @ np.hstack((R[:, 0:2], T))  # (3x3)
    try:
        H_inv = np.linalg.inv(H_plane)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Homography matrix H_plane is singular and cannot be inverted.")

    # 4) Back-project
    world_h = (H_inv @ uvh.T).T           # (M,3)

    # Avoid division by zero or very small numbers during normalization
    world_h_z = world_h[:, 2:3]
    # Use a reasonable epsilon, perhaps related to expected coordinate scale
    epsilon = 1e-8 * np.max(np.abs(world_h[:, :2])) if np.max(np.abs(world_h[:, :2])) > 0 else 1e-8
    world_h_z[np.abs(world_h_z) < epsilon] = np.sign(world_h_z[np.abs(world_h_z) < epsilon] + 1e-15) * epsilon # Replace near-zero Z with small signed number

    world_h = world_h / world_h_z           # normalize
    return world_h[:, :2]                # return X,Y



# --- Updated world_to_pixel_via_plane with DEBUG prints ---
def world_to_pixel_via_plane(world_pts: np.ndarray,
                             K: np.ndarray,
                             dist: np.ndarray,
                             R: np.ndarray,
                             T: np.ndarray
                            ) -> np.ndarray:
    """
    Map world points (X,Y) on the Z=0 plane to distorted pixel coordinates (u,v).
    """
    if world_pts.dtype != np.float32:
        world_pts = world_pts.astype(np.float32)

    num_points = world_pts.shape[0]
    if num_points == 0:
        return np.empty((0, 2), dtype=np.float32)

    world_pts_reshaped = world_pts.reshape(-1, 1, 2)
    object_points = np.zeros((num_points, 1, 3), dtype=np.float32)
    object_points[:, :, 0:2] = world_pts_reshaped

    rvec, _ = cv2.Rodrigues(R)

    if dist is not None:
        dist_coeffs = np.array(dist).flatten().astype(np.float32)
        if dist_coeffs.size == 0:
            dist_coeffs = None
    else:
        dist_coeffs = None

    tvec = T.reshape(3, 1).astype(np.float32)

    pixel_pts_distorted, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)

    # Attempt squeeze first
    squeezed_pts = np.squeeze(pixel_pts_distorted, axis=1)

    # Check if squeeze resulted in the correct shape (N, 2)
    if squeezed_pts.ndim == 2 and squeezed_pts.shape[0] == num_points and squeezed_pts.shape[1] == 2:
        final_pixel_pts = squeezed_pts
    else:
        final_pixel_pts = pixel_pts_distorted.reshape(num_points, 2)

    return final_pixel_pts


# --- create_world_view_image_fixed_canvas (no changes needed here) ---
def create_world_view_image_fixed_canvas(image: np.ndarray,
                                         K: np.ndarray,
                                         dist: np.ndarray,
                                         R: np.ndarray,
                                         T: np.ndarray,
                                         output_size_pixels: tuple = (2000, 2000)):
    """
    Transforms an image to a world-coordinate view (birds-eye view) on a
    fixed-size canvas, loading calibration from a .npz file and automatically
    determining the resolution.
    [Rest of docstring omitted for brevity]
    """
    h, w = image.shape[:2]

    # 2. Define corners of the input image
    corners_pix = np.array([
        [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
        [w/2, 0], [w/2, h-1], [0, h/2], [w-1, h/2]
    ], dtype=np.float32)

    # 3. Map image corners/edges to world coordinates to find the actual bounds
    try:
        corners_world = pixel_to_world_via_plane(corners_pix, K, dist, R, T) # uses this logic
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error mapping image points to world coordinates: {e}")
        return None

    corners_world = corners_world[np.all(np.isfinite(corners_world), axis=1)]
    if corners_world.shape[0] < 4:
        print("Error: Could not determine valid world coordinate bounds from image corners.")
        return None

    world_x_min, world_y_min = np.min(corners_world, axis=0)
    world_x_max, world_y_max = np.max(corners_world, axis=0)
    world_width_mm = world_x_max - world_x_min
    world_height_mm = world_y_max - world_y_min

    if world_width_mm <= 0 or world_height_mm <= 0:
        print(f"Error: Calculated world dimensions are non-positive ({world_width_mm:.2f}x{world_height_mm:.2f} mm). Check calibration or image content.")
        return None

    # 4. Calculate mm_per_pixel
    out_width_req_px, out_height_req_px = output_size_pixels
    if out_width_req_px <= 0 or out_height_req_px <= 0:
        print(f"Error: Invalid output_size_pixels requested: {output_size_pixels}")
        return None
    res_x = world_width_mm / out_width_req_px
    res_y = world_height_mm / out_height_req_px
    mm_per_pixel = max(res_x, res_y) * 1.001

    print(f"World coordinate range of content: X=[{world_x_min:.2f}, {world_x_max:.2f}], Y=[{world_y_min:.2f}, {world_y_max:.2f}] mm")
    print(f"Calculated resolution: {mm_per_pixel:.4f} mm/pixel")

    # 5. Calculate actual content size in pixels
    actual_out_width_px = int(np.ceil(world_width_mm / mm_per_pixel))
    actual_out_height_px = int(np.ceil(world_height_mm / mm_per_pixel))
    actual_out_width_px = min(actual_out_width_px, out_width_req_px)
    actual_out_height_px = min(actual_out_height_px, out_height_req_px)

    # 6. Calculate offsets
    offset_x = (out_width_req_px - actual_out_width_px) // 2
    offset_y = (out_height_req_px - actual_out_height_px) // 2

    # 7. Create mapping grids for the actual content area
    if actual_out_width_px <= 0 or actual_out_height_px <=0:
         print(f"Error: Calculated actual output pixel dimensions are invalid ({actual_out_width_px}x{actual_out_height_px}).")
         return None

    u_actual_out, v_actual_out = np.meshgrid(np.arange(actual_out_width_px), np.arange(actual_out_height_px))
    world_x = world_x_min + u_actual_out * mm_per_pixel
    world_y = world_y_min + v_actual_out * mm_per_pixel
    world_pts_actual_out = np.vstack([world_x.flatten(), world_y.flatten()]).T

    # 8. Map world points back to input pixels
    pixel_pts_in = world_to_pixel_via_plane(world_pts_actual_out, K, dist, R, T) # CALL TO UPDATED FUNCTION

    # 9. Reshape maps (Check consistency)
    map_x = pixel_pts_in[:, 0].reshape(actual_out_height_px, actual_out_width_px).astype(np.float32)
    map_y = pixel_pts_in[:, 1].reshape(actual_out_height_px, actual_out_width_px).astype(np.float32)

    # 10. Perform remapping
    temp_world_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # 11. Create final canvas and place content
    final_canvas = np.zeros((out_height_req_px, out_width_req_px, 3), dtype=np.uint8)
    end_y = offset_y + temp_world_image.shape[0]
    end_x = offset_x + temp_world_image.shape[1]
    if end_y > out_height_req_px or end_x > out_width_req_px:
         print("Warning: Calculated content size exceeds canvas dimensions after offset. Clipping.")
         end_y = min(end_y, out_height_req_px)
         end_x = min(end_x, out_width_req_px)
         # Adjust slice if clipping occurs
         final_canvas[offset_y : end_y, offset_x : end_x] = temp_world_image[:end_y-offset_y, :end_x-offset_x]
    else:
         final_canvas[offset_y : end_y, offset_x : end_x] = temp_world_image


    # 12. Calculate canvas world origin
    canvas_world_x_min = world_x_min - offset_x * mm_per_pixel
    canvas_world_y_min = world_y_min - offset_y * mm_per_pixel
    canvas_world_origin = (canvas_world_x_min, canvas_world_y_min)

    return final_canvas, mm_per_pixel, canvas_world_origin


if __name__ == "__main__":
    # Specify paths and desired canvas size
    calibration_data_file = 'Calibration.npz' # <--- Path to your .npz file
    
    image_path = '/home/hina/WiaT-PIV/DSC09771_adj_cropped.png'         # <--- Path to your input image
    output_canvas_size = (2000, 2000)     # Desired output canvas (width, height)

    # Load the input image
    input_image = cv2.imread(image_path)

    if input_image is None:
        print(f"Error: Could not load image at {image_path}")
    else:
        print(f"Input image loaded: {input_image.shape[1]}x{input_image.shape[0]}")
        # Create the world view on the fixed canvas
        if not os.path.exists(calibration_data_file):
            print(f"Error: Calibration data file {calibration_data_file} does not exist.")
            exit(1)
        calibration_data = np.load(calibration_data_file)
        K = calibration_data['K']
        dist = calibration_data['dist']
        R = calibration_data['R']
        T = calibration_data['T']
        result = create_world_view_image_fixed_canvas(input_image, K, dist, R, T, output_canvas_size)

        if result:
            world_canvas, resolution, canvas_origin = result
            canvas_w, canvas_h = output_canvas_size
            origin_x, origin_y = canvas_origin

            print(f"Successfully created world view canvas: {world_canvas.shape[1]}x{world_canvas.shape[0]}")
            print(f"World coordinates of canvas top-left: ({origin_x:.2f}, {origin_y:.2f}) mm")
            import matplotlib
            matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
            # Display the results
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            plt.title('Original Camera View')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            extent = [origin_x, origin_x + canvas_w * resolution, # X range
                    origin_y + canvas_h * resolution, origin_y] # Y range (inverted for imshow)

            plt.imshow(cv2.cvtColor(world_canvas, cv2.COLOR_BGR2RGB), extent=extent)
            plt.title(f'World View (Fixed Canvas {canvas_w}x{canvas_h})\nResolution: {resolution:.4f} mm/pixel')
            plt.xlabel('World X (mm)')
            plt.ylabel('World Y (mm)')
            plt.gca().invert_yaxis()
            plt.axis('equal')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig('world_view_fixed_canvas.png', dpi=300)

        else:
            print("Failed to create world view image.")
