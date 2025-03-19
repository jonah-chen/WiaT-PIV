import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from openpiv import pyprocess
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def sigma_clip(data, sigma=3, max_iter=5):
    """
    Perform sigma clipping on a 1D array.
    Returns the clipped data.
    """
    data = np.array(data)
    mask = np.ones(data.shape, dtype=bool)
    for _ in range(max_iter):
        if mask.sum() == 0:
            break
        clipped_data = data[mask]
        mean = np.mean(clipped_data)
        std = np.std(clipped_data)
        new_mask = np.abs(data - mean) < sigma * std
        if np.all(new_mask == mask):
            break
        mask = new_mask
    return data[mask]

def main():
    # Load the saved PIV data.
    if not os.path.exists('uvs.npy'):
        logger.error("uvs.npy not found. Run the PIV generation script first.")
        return
    uvs = np.load('uvs.npy')  # shape: (n_steps, ny, nx, 2)
    
    # Load the first frame to reconstruct grid coordinates.
    first_frame_path = 'PIV/first_frame.png'
    if not os.path.exists(first_frame_path):
        logger.error("PIV/first_frame.png not found. Run the PIV generation script first.")
        return
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        logger.error("Could not load the first frame image.")
        return

    # Set interrogation grid parameters.
    searchsize = 76
    overlap = 32
    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).astype(np.int32)
    x, y = pyprocess.get_coordinates(image_size=gray_first.shape, search_area_size=searchsize, overlap=overlap)
    
    # Determine the image dimensions.
    height, width, _ = first_frame.shape
    # Given that the bottom left/right pixels are 7 inches (17.78 cm) from the center,
    # half the image width corresponds to 7 inches.
    # Compute scale in inches per pixel then convert to centimeters per pixel.
    scale_inch = 14.0 / width         # inches per pixel
    scale_cm = scale_inch * 2.54        # cm per pixel

    # Convert grid coordinates to a coordinate system with origin at the bottom center.
    x_cm = (x - (width / 2)) * scale_cm
    y_cm = (height - y) * scale_cm

    # Compute the radial distance for each grid point (in cm).
    r = np.sqrt(x_cm**2 + y_cm**2)

    # Convert the PIV velocities from pixels/frame to cm/s.
    # uvs[..., 0] and uvs[..., 1] are in pixels/frame.
    # Conversion: pixels/frame -> inches/frame -> cm/frame, then multiply by frame rate (30 fps)
    speeds = np.sqrt((uvs[..., 0] * scale_inch)**2 + (uvs[..., 1] * scale_inch)**2)
    speeds_cm_s = speeds * 2.54 * 30  # cm/s

    n_steps, ny, nx = speeds_cm_s.shape
    robust_mean_speed = np.empty((ny, nx))
    robust_std_speed  = np.empty((ny, nx))
    
    # Compute robust (sigma-clipped) mean and standard deviation at each grid point.
    for i in range(ny):
        for j in range(nx):
            data = speeds_cm_s[:, i, j]
            clipped_data = sigma_clip(data, sigma=3, max_iter=5)
            if clipped_data.size > 0:
                robust_mean_speed[i, j] = np.mean(clipped_data)
                robust_std_speed[i, j]  = np.std(clipped_data)
            else:
                robust_mean_speed[i, j] = np.nan
                robust_std_speed[i, j]  = np.nan

    # Flatten arrays for fitting and plotting.
    r_flat = r.flatten()                    # radial distance (cm)
    mean_speed_flat = robust_mean_speed.flatten()  # speed in cm/s
    std_speed_flat  = robust_std_speed.flatten()     # uncertainty in cm/s

    # Fit a line of best fit (linear regression) to the data.
    # Using np.polyfit with cov=True to get the covariance matrix.
    p, cov = np.polyfit(r_flat, mean_speed_flat, 1, cov=True)
    slope, intercept = p
    slope_err = np.sqrt(cov[0, 0])
    
    # Generate fitted values for a smooth line.
    r_fit = np.linspace(r_flat.min(), r_flat.max(), 100)
    speed_fit = slope * r_fit + intercept

    # Convert the slope (which is in units cm/s per cm) to an angular velocity.
    # For a rotating system: v = ω r, hence slope ≡ ω (in rad/s).
    omega = slope        # rad/s
    omega_err = slope_err  # rad/s
    # Convert to rpm: rpm = ω * (60 / (2π))
    rpm = omega * 60 / (2 * np.pi)
    rpm_err = omega_err * 60 / (2 * np.pi)
    
    # Print the slope and its uncertainty as well as the converted rpm values.
    print(f"Slope: {slope:.4f} ± {slope_err:.4f} (cm/s per cm) => Angular velocity: {omega:.4f} ± {omega_err:.4f} rad/s")
    print(f"Converted to rpm: {rpm:.4f} ± {rpm_err:.4f} rpm")
    
    # Plot the data with error bars and the best-fit line.
    plt.figure()
    plt.errorbar(r_flat, mean_speed_flat, yerr=std_speed_flat, fmt='o', markersize=3,
                 ecolor='gray', alpha=0.7, label='Data')
    plt.plot(r_fit, speed_fit, 'r-', linewidth=2,
             label=f'Fit: speed = {slope:.3f}±{slope_err:.3f}*r + {intercept:.2f}')
    plt.xlabel('Radius (cm)')
    plt.ylabel('Speed (cm/s)')
    plt.legend()
    plt.grid(True)
    plot_filename = 'PIV/speed_vs_radius_with_uncertainty_fit.png'
    plt.savefig(plot_filename)
    plt.clf()
    logger.info(f"Post-analysis plot saved as {plot_filename}")

if __name__ == '__main__':
    main()
