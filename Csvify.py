import os
import cv2
import csv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy

# ----------------------------- CONFIG -----------------------------
BASE_PATH = "/Users/zevfine/Desktop/CS/cs506/Noise_Reduction_CS506/Photos"
OUTPUT_FOLDER = os.path.join(BASE_PATH, "Photo_Data")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
NOISE_TYPES = ["gaussian", "poisson", "salt_pepper", "speckle"]
ORIGINAL_FOLDERS = ["Pokemon", "Sports", "Pistachio_Image_Dataset", "BSDS300"]

# ----------------------------- CREATE FOLDER STRUCTURE -----------------------------
def create_folder_structure():
    """Create the organized folder structure for CSV files"""
    
    # Remove existing Photo_Data folder to start fresh
    if os.path.exists(OUTPUT_FOLDER):
        import shutil
        shutil.rmtree(OUTPUT_FOLDER)
    
    # Main folders
    original_folder = os.path.join(OUTPUT_FOLDER, "original")
    bw_folder = os.path.join(OUTPUT_FOLDER, "BW")
    color_folder = os.path.join(OUTPUT_FOLDER, "Color")
    
    os.makedirs(original_folder, exist_ok=True)
    os.makedirs(bw_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)
    
    # Create dataset subfolders for BW and Color
    for dataset in ORIGINAL_FOLDERS:
        # Create dataset folders in BW
        dataset_bw_path = os.path.join(bw_folder, dataset)
        os.makedirs(dataset_bw_path, exist_ok=True)
        
        # Create noise type subfolders within each BW dataset
        for noise_type in NOISE_TYPES:
            noise_bw_path = os.path.join(dataset_bw_path, noise_type)
            os.makedirs(noise_bw_path, exist_ok=True)
        
        # Create dataset folders in Color
        dataset_color_path = os.path.join(color_folder, dataset)
        os.makedirs(dataset_color_path, exist_ok=True)
        
        # Create noise type subfolders within each Color dataset
        for noise_type in NOISE_TYPES:
            noise_color_path = os.path.join(dataset_color_path, noise_type)
            os.makedirs(noise_color_path, exist_ok=True)
    
    print("✅ Created folder structure:")
    print(f"   📁 {OUTPUT_FOLDER}/")
    print(f"   ├── original/")
    print(f"   ├── BW/")
    print(f"   │   ├── Pokemon/")
    print(f"   │   │   ├── gaussian/")
    print(f"   │   │   ├── poisson/")
    print(f"   │   │   ├── salt_pepper/")
    print(f"   │   │   └── speckle/")
    print(f"   │   └── ... (other datasets)")
    print(f"   └── Color/")
    print(f"       ├── Pokemon/")
    print(f"       │   ├── gaussian/")
    print(f"       │   ├── poisson/")
    print(f"       │   ├── salt_pepper/")
    print(f"       │   └── speckle/")
    print(f"       └── ... (other datasets)")
    
    return original_folder, bw_folder, color_folder

# ----------------------------- METRICS FUNCTION -----------------------------
def compute_metrics(original, noisy):
    """Compute all required metrics between original and noisy images"""
    try:
        # Convert to grayscale for SSIM
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        ssim_score = ssim(gray_orig, gray_noisy, data_range=gray_noisy.max() - gray_noisy.min())
        
        # Calculate PSNR
        psnr_value = cv2.PSNR(original, noisy)
        
        # Calculate MSE
        mse = np.mean((original.astype("float") - noisy.astype("float")) ** 2)
        
        # Calculate Noise Variance and Std
        noise_diff = (noisy.astype("float") - original.astype("float"))
        noise_var = np.var(noise_diff)
        noise_std = np.std(noise_diff)
        
        # Calculate Entropy
        entropy_val = shannon_entropy(noisy)
        
        # Calculate Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray_noisy, cv2.CV_64F).var()
        
        return {
            "SSIM": ssim_score,
            "PSNR": psnr_value,
            "MSE": mse,
            "Noise_Variance": noise_var,
            "Noise_Std": noise_std,
            "Entropy": entropy_val,
            "Sharpness": sharpness
        }
    except Exception as e:
        print(f"❌ Error computing metrics: {e}")
        return None

# ----------------------------- MAIN PROCESSING -----------------------------
def main():
    print("🚀 Starting image metrics analysis...")
    
    # Create folder structure
    original_folder, bw_folder, color_folder = create_folder_structure()
    
    total_csv_files = 0
    
    for folder_name in ORIGINAL_FOLDERS:
        print(f"\n📁 Processing {folder_name}...")
        
        # Define paths
        original_dataset_folder = os.path.join(BASE_PATH, folder_name)
        color_noised_folder = os.path.join(BASE_PATH, f"{folder_name}_noised")
        bw_noised_folder = os.path.join(BASE_PATH, f"{folder_name}_BW_noised")
        
        # Check if folders exist
        if not os.path.exists(original_dataset_folder):
            print(f"❌ Original folder not found: {original_dataset_folder}")
            continue
        
        print(f"✅ Found original folder: {original_dataset_folder}")
        
        # Load original images
        original_images = {}
        for file in os.listdir(original_dataset_folder):
            if any(file.lower().endswith(ext.lower()) for ext in IMAGE_EXTENSIONS):
                base_name = os.path.splitext(file)[0]
                original_images[base_name] = os.path.join(original_dataset_folder, file)
        
        print(f"📸 Found {len(original_images)} original images")
        
        # Process COLOR noised images
        if os.path.exists(color_noised_folder):
            print(f"🎨 Processing COLOR noised images...")
            
            for noise_type in NOISE_TYPES:
                noise_folder = os.path.join(color_noised_folder, noise_type)
                if not os.path.exists(noise_folder):
                    print(f"   ❌ Noise folder not found: {noise_folder}")
                    continue
                
                print(f"   🔧 Processing {noise_type} noise...")
                metrics_data = []
                
                # Count files in noise folder
                noisy_files = [f for f in os.listdir(noise_folder) if any(f.lower().endswith(ext.lower()) for ext in IMAGE_EXTENSIONS)]
                print(f"   📊 Found {len(noisy_files)} noisy images")
                
                # Process each noisy image
                for noisy_file in noisy_files:
                    # Extract original image name from noisy filename
                    # Remove the noise type suffix and file extension
                    original_base = noisy_file
                    for ext in IMAGE_EXTENSIONS:
                        if original_base.lower().endswith(f"_{noise_type}{ext}".lower()):
                            original_base = original_base[:-len(f"_{noise_type}{ext}")]
                            break
                        elif original_base.lower().endswith(ext.lower()):
                            # Remove just the extension if no noise type in filename
                            original_base = original_base[:-len(ext)]
                            # Try to remove noise type if it exists before extension
                            if original_base.endswith(f"_{noise_type}"):
                                original_base = original_base[:-len(f"_{noise_type}")]
                            break
                    
                    # Find matching original image
                    original_path = original_images.get(original_base)
                    if not original_path:
                        # Try alternative naming patterns
                        alt_base = original_base.replace("_BW", "").replace("_bw", "")
                        original_path = original_images.get(alt_base)
                        if not original_path:
                            print(f"      ❌ Original not found for: {noisy_file} (tried: {original_base})")
                            continue
                    
                    # Load images
                    original_img = cv2.imread(original_path)
                    noisy_img_path = os.path.join(noise_folder, noisy_file)
                    noisy_img = cv2.imread(noisy_img_path)
                    
                    if original_img is None:
                        print(f"      ❌ Failed to load original: {original_path}")
                        continue
                    if noisy_img is None:
                        print(f"      ❌ Failed to load noisy: {noisy_img_path}")
                        continue
                    
                    # Ensure same dimensions
                    if original_img.shape != noisy_img.shape:
                        noisy_img = cv2.resize(noisy_img, (original_img.shape[1], original_img.shape[0]))
                    
                    # Compute metrics
                    metrics = compute_metrics(original_img, noisy_img)
                    if metrics:
                        metrics["Image_Name"] = noisy_file
                        metrics["Original_Image"] = os.path.basename(original_path)
                        metrics_data.append(metrics)
                        print(f"      ✅ Processed: {noisy_file}")
                
                # Save CSV for this noise type
                if metrics_data:
                    csv_name = f"{folder_name}_{noise_type}_metrics.csv"
                    csv_path = os.path.join(color_folder, folder_name, noise_type, csv_name)
                    
                    with open(csv_path, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=metrics_data[0].keys())
                        writer.writeheader()
                        writer.writerows(metrics_data)
                    
                    print(f"      💾 Saved {len(metrics_data)} entries to Color/{folder_name}/{noise_type}/{csv_name}")
                    total_csv_files += 1
                else:
                    print(f"      ❌ No data collected for {folder_name} color {noise_type}")
        
        # Process BW noised images
        if os.path.exists(bw_noised_folder):
            print(f"⚫ Processing BW noised images...")
            
            for noise_type in NOISE_TYPES:
                noise_folder = os.path.join(bw_noised_folder, noise_type)
                if not os.path.exists(noise_folder):
                    print(f"   ❌ Noise folder not found: {noise_folder}")
                    continue
                
                print(f"   🔧 Processing {noise_type} noise...")
                metrics_data = []
                
                # Count files in noise folder
                noisy_files = [f for f in os.listdir(noise_folder) if any(f.lower().endswith(ext.lower()) for ext in IMAGE_EXTENSIONS)]
                print(f"   📊 Found {len(noisy_files)} noisy images")
                
                # Process each noisy image
                for noisy_file in noisy_files:
                    # Extract original image name from noisy filename
                    # Remove the BW noise type suffix and file extension
                    original_base = noisy_file
                    for ext in IMAGE_EXTENSIONS:
                        if original_base.lower().endswith(f"_bw_{noise_type}{ext}".lower()):
                            original_base = original_base[:-len(f"_bw_{noise_type}{ext}")]
                            break
                        elif original_base.lower().endswith(f"_BW_{noise_type}{ext}".lower()):
                            original_base = original_base[:-len(f"_BW_{noise_type}{ext}")]
                            break
                        elif original_base.lower().endswith(ext.lower()):
                            # Remove just the extension if no clear pattern
                            original_base = original_base[:-len(ext)]
                            # Try to remove BW noise type if it exists before extension
                            if original_base.endswith(f"_BW_{noise_type}"):
                                original_base = original_base[:-len(f"_BW_{noise_type}")]
                            elif original_base.endswith(f"_bw_{noise_type}"):
                                original_base = original_base[:-len(f"_bw_{noise_type}")]
                            break
                    
                    # Find matching original image
                    original_path = original_images.get(original_base)
                    if not original_path:
                        print(f"      ❌ Original not found for: {noisy_file} (tried: {original_base})")
                        continue
                    
                    # Load images
                    original_img = cv2.imread(original_path)
                    noisy_img_path = os.path.join(noise_folder, noisy_file)
                    noisy_img = cv2.imread(noisy_img_path)
                    
                    if original_img is None:
                        print(f"      ❌ Failed to load original: {original_path}")
                        continue
                    if noisy_img is None:
                        print(f"      ❌ Failed to load noisy: {noisy_img_path}")
                        continue
                    
                    # Ensure same dimensions
                    if original_img.shape != noisy_img.shape:
                        noisy_img = cv2.resize(noisy_img, (original_img.shape[1], original_img.shape[0]))
                    
                    # Compute metrics
                    metrics = compute_metrics(original_img, noisy_img)
                    if metrics:
                        metrics["Image_Name"] = noisy_file
                        metrics["Original_Image"] = os.path.basename(original_path)
                        metrics_data.append(metrics)
                        print(f"      ✅ Processed: {noisy_file}")
                
                # Save CSV for this noise type
                if metrics_data:
                    csv_name = f"{folder_name}_{noise_type}_metrics.csv"
                    csv_path = os.path.join(bw_folder, folder_name, noise_type, csv_name)
                    
                    with open(csv_path, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=metrics_data[0].keys())
                        writer.writeheader()
                        writer.writerows(metrics_data)
                    
                    print(f"      💾 Saved {len(metrics_data)} entries to BW/{folder_name}/{noise_type}/{csv_name}")
                    total_csv_files += 1
                else:
                    print(f"      ❌ No data collected for {folder_name} BW {noise_type}")
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Total CSV files created: {total_csv_files}")
    print(f"📁 All files organized in: {OUTPUT_FOLDER}")
    
    # Display final structure
    print("\n📁 Final Folder Structure:")
    for root, dirs, files in os.walk(OUTPUT_FOLDER):
        level = root.replace(OUTPUT_FOLDER, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.csv'):
                file_size = os.path.getsize(os.path.join(root, file))
                print(f"{subindent}📄 {file} ({file_size} bytes)")

if __name__ == "__main__":
    main()