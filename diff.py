import argparse
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

def show_diff_image(img1_path, img2_path, diff_path='diff.png'):
    img1 = np.array(Image.open(img1_path).convert('RGB'))
    img2 = np.array(Image.open(img2_path).convert('RGB'))

    if img1.shape != img2.shape:
        print("Images have different dimensions:", img1.shape, "vs", img2.shape)
        return

    diff = np.abs(img1.astype(int) - img2.astype(int)).astype(np.uint8)

    diff_pixels = np.any(diff > 0, axis=2)
    num_diff_pixels = np.sum(diff_pixels)
    total_pixels = diff_pixels.size
    diff_percent = (num_diff_pixels / total_pixels) * 100

    max_diff = np.max(diff)
    print(f"Max pixel difference (R/G/B): {max_diff}")
    print(f"Different pixels: {num_diff_pixels} / {total_pixels} ({diff_percent:.2f}%)")

    if num_diff_pixels == 0:
        print("Images are exactly identical.")
    else:
        print(f"Images differ. Saving diff image as {diff_path}")
        diff_img = Image.fromarray(diff)
        diff_img.save(diff_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare two images and generate a diff image.")
    parser.add_argument('img1', help="Path to the first image (e.g., ans.png)")
    parser.add_argument('img2', help="Path to the second image (e.g., output.png)")
    parser.add_argument('-o', '--output', default='diff.png', help="Output filename for diff image")
    args = parser.parse_args()

    show_diff_image(args.img1, args.img2, args.output)