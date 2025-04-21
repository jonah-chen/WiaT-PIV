import matplotlib
matplotlib.use("Agg")
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from openpiv import tools, pyprocess, validation, filters, scaling
from tqdm import tqdm
from time import perf_counter
import logging
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

def process_piv(frame_a, frame_b, 
                window_size=64, 
                searchsize=76, 
                overlap=32, 
                dt=1, 
                noise_threshold=1.15,
                mmppx=1  # millimeters per pixel
):
    # Convert frames to grayscale
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.int32)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.int32)

    # Perform PIV analysis
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        gray_a, gray_b,
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=searchsize,
        sig2noise_method='peak2peak'
    )

    # Get coordinates for the interrogation grid
    x, y = pyprocess.get_coordinates(
        image_size=gray_a.shape,
        search_area_size=searchsize,
        overlap=overlap
    )

    # Validate and filter the vectors
    invalid_mask = validation.sig2noise_val(sig2noise, threshold=noise_threshold)
    u, v = filters.replace_outliers(
        u, v,
        invalid_mask,
        method='localmean',
        max_iter=3,
        kernel_size=2
    )

    # Scale the results (adjust scaling_factor as needed)
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=mmppx)
    tools.transform_coordinates(x, y, u, v)

    return x, y, u, v

def display_piv(name, frame, x, y, u, v):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.quiver(x, y, u, v, color='r')
    plt.savefig(name)
    plt.cla()

def process(idx, frame_a, frame_b):
    x, y, u, v = process_piv(frame_a, frame_b)
    # Save individual PIV image (optional)
    display_piv(f'PIV/{idx}.png', frame_a, x, y, u, v)
    return np.stack([u.data, v.data], axis=-1), u.mask

def convert_to_video(fps=30):
    subprocess.run(['ffmpeg', '-r', str(fps), '-i', 'PIV/%d.png', '-vcodec', 'libx264', '-y', 'PIV.mp4'])

def main(video_path, fps=10, displacement_frames=1, start_frame=0, end_frame=2000, **kwargs):
    window_size = kwargs.get("window_size", 64)
    searchsize = kwargs.get("searchsize", 76)
    overlap = kwargs.get("overlap", 32)
    
    os.makedirs('PIV', exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Processing video: {video_path} of {total_frames//vid_fps} seconds")
    
    dt = max(1, int(vid_fps / fps))
    logger.info(f"Video FPS: {vid_fps}, Processing FPS: {fps}, time stride: {dt} frames")
    logger.info(f"Processing frames {start_frame} to {end_frame}")
    logger.info(f"This corresponds to {start_frame / fps:.2f} to {end_frame / fps:.2f} seconds")
    
    if not cap.isOpened():
        logger.error("Error: Could not open video.")
        return

    futures = []
    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Error: Could not read video.")
        return

    # Save the first frame for later use in post processing
    first_frame = prev_frame.copy()
    cv2.imwrite('PIV/first_frame.png', first_frame)

    frame_no = -1
    t = 1
    start_time = perf_counter()

    with ProcessPoolExecutor(max_workers=24) as executor:
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            if t % dt == 0 or (t - displacement_frames) % dt == 0:
                if t % dt == 0:
                    frame_no += 1
                if (t - displacement_frames) % dt == 0:
                    if frame_no > end_frame:
                        logger.info(f"Reached end frame {end_frame}, skipping rest of the video")
                        break
                    frame_no = frame_no - start_frame
                    if frame_no > 0:
                        futures.append(executor.submit(process, frame_no, prev_frame, curr_frame))
                    if frame_no >= 0 and frame_no % 100 == 0:
                        time_elapsed = perf_counter() - start_time
                        logger.info(f"Processed {frame_no} frames in {time_elapsed:.2f} seconds "
                                    f"({frame_no / time_elapsed:.2f} fps), corresponding to {frame_no / fps:.2f} irl seconds")
                    frame_no = frame_no + start_frame
                if t % dt == 0:
                    prev_frame = curr_frame
            t += 1

    cap.release()
    cv2.destroyAllWindows()
    
    logger.info(f"Read {frame_no} frames, processing {len(futures)} frames")

    uvs = []
    masks = []
    for _, future in tqdm(enumerate(futures), total=len(futures)):
        uv, mask = future.result()
        uvs.append(uv)
        masks.append(mask)
    
    logger.info(f"Saving {len(uvs)} PIV results")
    uvs = np.stack(uvs, axis=0)  # shape: (n_steps, ny, nx, 2)
    masks = np.stack(masks, axis=0)

    # Save the PIV data for later analysis.
    np.save('uvs.npy', uvs)
    np.save('masks.npy', masks)
    logger.info("PIV results saved to uvs.npy and masks.npy")
    
    logger.info("Converting PIV images to video")
    convert_to_video()

    # Archive selected results (without deleting any files)
    output_zip = os.path.splitext(video_path)[0] + '.zip'
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in ['PIV.mp4', 'uvs.npy', 'masks.npy', 'PIV/first_frame.png']:
            if os.path.exists(filename):
                zipf.write(filename)
    logger.info(f"Results have been archived to {output_zip}")

    # ----- FINAL AVERAGED PIV IMAGE -----
    avg_uv = np.mean(uvs, axis=0)  # shape: (ny, nx, 2)
    u_avg = avg_uv[..., 0]
    v_avg = avg_uv[..., 1]
    
    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).astype(np.int32)
    x, y = pyprocess.get_coordinates(image_size=gray_first.shape, search_area_size=searchsize, overlap=overlap)
    
    final_image_name = 'PIV/final_average.png'
    display_piv(final_image_name, first_frame, x, y, u_avg, v_avg)
    logger.info(f"Final averaged PIV image saved as {final_image_name}")

if __name__ == '__main__':
    video_path = 'preprocessed.mov'
    main(video_path, fps=3, start_frame=0, end_frame=1000)
