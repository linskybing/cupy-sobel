import time
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import correlate
from PIL import Image

MASK_N = 2
MASK_X = 5
MASK_Y = 5
SCALE = 8

mask = np.array([
    [[ -1, -4, -6, -4, -1],
     [ -2, -8,-12, -8, -2],
     [  0,  0,  0,  0,  0],
     [  2,  8, 12,  8,  2],
     [  1,  4,  6,  4,  1]],

    [[ -1, -2,  0,  2,  1],
     [ -4, -8,  0,  8,  4],
     [ -6,-12,  0, 12,  6],
     [ -4, -8,  0,  8,  4],
     [ -1, -2,  0,  2,  1]]
], dtype=np.float32)

def sobel_cupy(image: np.ndarray) -> np.ndarray:
    image_bgr = image[..., ::-1]
    image_gpu = cp.asarray(image_bgr, dtype=cp.float32)
    mask_gpu = cp.asarray(mask)

    channels = []
    for c in range(3):
        conv_results = []
        for i in range(MASK_N):
            conv = correlate(image_gpu[:, :, c], mask_gpu[i], mode='reflect')
            conv_results.append(conv)
        total = cp.sqrt(conv_results[0]**2 + conv_results[1]**2) / SCALE
        total = cp.clip(total, 0, 255)
        channels.append(total)

    output_bgr = cp.stack(channels, axis=-1)
    output_rgb = output_bgr[..., ::-1]
    return cp.asnumpy(output_rgb.astype(cp.uint8))

def main(input_path, output_path):
    img = Image.open(input_path).convert('RGB')
    img_np = np.array(img)

    start_time = time.time()
    result_np = sobel_cupy(img_np)
    end_time = time.time()

    result_img = Image.fromarray(result_np)
    result_img.save(output_path)

    print(f"[Cupy] Sobel filter took {end_time - start_time:.4f} seconds, output saved to {output_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python sobel.py input.png output.png")
    else:
        main(sys.argv[1], sys.argv[2])
