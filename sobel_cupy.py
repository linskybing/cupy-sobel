import time
import sys
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import convolve
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# 5x5 Sobel masks for X and Y gradient detection
SOBEL_MASKS = np.array([
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

SCALE = 8.0

def sobel_cupy(image_np: np.ndarray) -> np.ndarray:
    image_rgb = image_np.astype(np.float32)

    image_gpu = cp.asarray(image_rgb)
    masks_gpu = cp.asarray(SOBEL_MASKS)

    result_channels = []

    for c in range(3):
        grad_x = convolve(image_gpu[:, :, c], masks_gpu[0], mode='constant', cval=0.0)
        grad_y = convolve(image_gpu[:, :, c], masks_gpu[1], mode='constant', cval=0.0)
        gradient = cp.sqrt(grad_x ** 2 + grad_y ** 2) / SCALE
        gradient = cp.clip(gradient, 0, 255)
        result_channels.append(gradient)

    result_rgb = cp.stack(result_channels, axis=-1).astype(cp.uint8)
    return cp.asnumpy(result_rgb)

def main(input_path: str, output_path: str):
    img = Image.open(input_path).convert('RGB')
    img_np = np.array(img)

    start = time.time()
    result = sobel_cupy(img_np)
    end = time.time()

    Image.fromarray(result).save(output_path)
    print(f"[CuPy] Sobel 5x5 filter took {end - start:.4f} seconds, saved to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python sobel.py input.png output.png")
    else:
        main(sys.argv[1], sys.argv[2])
