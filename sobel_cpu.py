import time
import numpy as np
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
])

def sobel_cpu(image: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    
    output = np.zeros_like(image)

    xBound = MASK_X // 2
    yBound = MASK_Y // 2
    adjustX = 1 if MASK_X % 2 else 0
    adjustY = 1 if MASK_Y % 2 else 0

    val = np.zeros((MASK_N, 3))

    for y in range(height):
        for x in range(width):
            for i in range(MASK_N):
                val[i, 2] = 0.0
                val[i, 1] = 0.0
                val[i, 0] = 0.0
                for v in range(-yBound, yBound + adjustY):
                    for u in range(-xBound, xBound + adjustX):
                        xi = x + u
                        yi = y + v
                        if 0 <= xi < width and 0 <= yi < height:
                            R, G, B = image[yi, xi]

                            val[i, 2] += R * mask[i, u + xBound, v + yBound]
                            val[i, 1] += G * mask[i, u + xBound, v + yBound]
                            val[i, 0] += B * mask[i, u + xBound, v + yBound]
            

            total = np.zeros(3)
            for i in range(MASK_N):
                total += val[i] ** 2
            total = np.sqrt(total) / SCALE
            total = np.clip(total, 0, 255)

            output[y, x, 0] = total[2]
            output[y, x, 1] = total[1]
            output[y, x, 2] = total[0]

    return output.astype(np.uint8)

def main(input_path, output_path):
    img = Image.open(input_path).convert('RGB')
    img_np = np.array(img)

    start_time = time.time()
    result_np = sobel_cpu(img_np)
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