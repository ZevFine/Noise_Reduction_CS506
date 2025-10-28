import os
import cv2
import numpy as np
from tqdm import tqdm

# Base path
base_path = "/Users/zevfine/Desktop/CS/cs506/Noise_Reduction_CS506/Photos"

# ---------------------------- Noise functions ----------------------------
def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean, sigma = 0, 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, amount=0.01):
    noisy = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.5)
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

def add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_speckle_noise(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    noisy = image + image * gauss * 0.1
    return np.clip(noisy, 0, 255).astype(np.uint8)

NOISE_FUNCS = {
    "gaussian": add_gaussian_noise,
    "salt_pepper": add_salt_pepper_noise,
    "poisson": add_poisson_noise,
    "speckle": add_speckle_noise
}

# ---------------------------- Folder logic ----------------------------
def create_noise_folders(base_folder):
    parent = os.path.dirname(base_folder)
    folder_name = os.path.basename(base_folder)
    noised_root = os.path.join(parent, f"{folder_name}_noised")
    bw_noised_root = os.path.join(parent, f"{folder_name}_BW_noised")

    os.makedirs(noised_root, exist_ok=True)
    os.makedirs(bw_noised_root, exist_ok=True)

    # Create subfolders for each noise type
    for noise_type in NOISE_FUNCS.keys():
        os.makedirs(os.path.join(noised_root, noise_type), exist_ok=True)
        os.makedirs(os.path.join(bw_noised_root, noise_type), exist_ok=True)

    return noised_root, bw_noised_root

# ---------------------------- Main loop ----------------------------
for root, dirs, _ in os.walk(base_path):
    for folder in dirs:
        full_folder = os.path.join(root, folder)
        noised_root, bw_noised_root = create_noise_folders(full_folder)

        for dirpath, _, filenames in os.walk(full_folder):
            for filename in tqdm(filenames, desc=f"Processing {folder}"):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    continue

                img_path = os.path.join(dirpath, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                base_name, _ = os.path.splitext(filename)

                # Apply each noise type
                for noise_name, func in NOISE_FUNCS.items():
                    noisy_img = func(img)
                    noise_subdir = os.path.join(noised_root, noise_name)
                    cv2.imwrite(os.path.join(noise_subdir, f"{base_name}_{noise_name}.png"), noisy_img)

                    # Convert to BW then add same noise
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    noisy_gray = func(gray_3ch)
                    bw_noise_subdir = os.path.join(bw_noised_root, noise_name)
                    cv2.imwrite(os.path.join(bw_noise_subdir, f"{base_name}_BW_{noise_name}.png"), noisy_gray)
