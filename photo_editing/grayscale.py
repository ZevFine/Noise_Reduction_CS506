import os
import cv2
import numpy as np
from tqdm import tqdm

def debug_photo_conversion():
    """
    Complete debug script to find and fix the photo conversion issue
    """
    print("🔍 STARTING PHOTO CONVERSION DEBUG")
    print("=" * 50)
    
    # Get current script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"1. Script directory: {script_dir}")
    print(f"   Script folder name: {os.path.basename(script_dir)}")
    
    # Go up one level from photo_editing to parent
    parent_dir = os.path.dirname(script_dir)
    print(f"2. Parent directory: {parent_dir}")
    print(f"   Parent folder name: {os.path.basename(parent_dir)}")
    
    # Go into Photos_Subset folder
    photos_subset_path = os.path.join(parent_dir, "Photos_Subset")
    print(f"3. Photos_Subset path: {photos_subset_path}")
    
    # Check if directory exists
    if not os.path.exists(photos_subset_path):
        print(f" ERROR: Directory '{photos_subset_path}' does not exist!")
        print("\nLet's see what's actually in the parent directory:")
        parent_contents = os.listdir(parent_dir)
        if not parent_contents:
            print("   - (empty directory)")
        else:
            for item in parent_contents:
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path):
                    print(f"   📁 {item}/")
                else:
                    print(f"   📄 {item}")
        
        # Try absolute path as fallback
        print(f"\n Trying absolute path fallback...")
        absolute_path = "./Noise_Reduction_CS506/Photos_Subset"
        if os.path.exists(absolute_path):
            print(f"Found directory at absolute path: {absolute_path}")
            photos_subset_path = absolute_path
        else:
            print(f"Absolute path also not found: {absolute_path}")
            return
    else:
        print(f" Photos_Subset directory exists!")
    
    print(f"\n4. Scanning Photos_Subset directory...")
    
    # List ALL contents of Photos_Subset
    try:
        all_files = os.listdir(photos_subset_path)
        print(f"   Total items in directory: {len(all_files)}")
        
        if not all_files:
            print("   - (empty directory)")
        else:
            # Count file types
            image_files = []
            other_files = []
            
            for item in all_files:
                item_path = os.path.join(photos_subset_path, item)
                if os.path.isfile(item_path):
                    if item.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')):
                        image_files.append(item)
                    else:
                        other_files.append(item)
                else:
                    print(f"   📁 {item}/ (subdirectory)")
            
            print(f"   📸 Image files: {len(image_files)}")
            for img_file in image_files:
                print(f"      - {img_file}")
            
            print(f"   📄 Other files: {len(other_files)}")
            for other_file in other_files:
                print(f"      - {other_file}")
                
    except Exception as e:
        print(f"Error reading directory: {e}")
        return
    
    if not image_files:
        print("No image files found to process!")
        return
    
    print(f"\n5. Starting black and white conversion...")
    print(f"   Will create copies with '_bw' suffix")
    
    # Process each image with detailed feedback
    success_count = 0
    for filename in tqdm(image_files, desc="Converting images"):
        print(f"\n   Processing: {filename}")
        try:
            input_path = os.path.join(photos_subset_path, filename)
            
            # Check if file exists and is readable
            if not os.path.exists(input_path):
                print(f"       File not found: {input_path}")
                continue
                
            if not os.access(input_path, os.R_OK):
                print(f"      File not readable: {filename}")
                continue
            
            # Read the image
            img = cv2.imread(input_path)
            
            if img is not None:
                print(f"       Successfully read image")
                print(f"       Image dimensions: {img.shape}")
                
                # Convert to grayscale
                bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                bw_img_bgr = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR)
                
                # Create output filename
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_bw{ext}"
                output_path = os.path.join(photos_subset_path, output_filename)
                
                # Check if output file already exists
                if os.path.exists(output_path):
                    print(f"        Output file already exists, overwriting: {output_filename}")
                
                # Save the black and white version
                success = cv2.imwrite(output_path, bw_img_bgr)
                if success:
                    print(f"       Successfully saved: {output_filename}")
                    success_count += 1
                    
                    # Verify the file was created
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        print(f"       File created, size: {file_size} bytes")
                    else:
                        print(f"       File not found after saving: {output_filename}")
                else:
                    print(f"       FAILED to save: {output_filename}")
                
            else:
                print(f"       Could not read image data from: {filename}")
                print(f"       Try opening this file manually with an image viewer")
                
        except Exception as e:
            print(f"       Error processing {filename}: {str(e)}")
    
    print(f"\n6. CONVERSION SUMMARY")
    print("=" * 50)
    print(f"   Total images processed: {len(image_files)}")
    print(f"   Successful conversions: {success_count}")
    print(f"   Failed conversions: {len(image_files) - success_count}")
    
    # Show final directory contents
    print(f"\n7. FINAL DIRECTORY CONTENTS")
    print("=" * 50)
    try:
        final_files = os.listdir(photos_subset_path)
        bw_files = [f for f in final_files if '_bw' in f and os.path.isfile(os.path.join(photos_subset_path, f))]
        original_files = [f for f in final_files if '_bw' not in f and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'))]
        
        print(f"   Total files in directory: {len(final_files)}")
        print(f"   Original image files: {len(original_files)}")
        print(f"   Black & white files: {len(bw_files)}")
        
        if bw_files:
            print(f"\n   📷 Black & white files created:")
            for bw_file in sorted(bw_files):
                file_path = os.path.join(photos_subset_path, bw_file)
                file_size = os.path.getsize(file_path)
                print(f"      - {bw_file} ({file_size} bytes)")
        else:
            print(f"\n    No black and white files were created!")
            
    except Exception as e:
        print(f"   Error reading final directory: {e}")
    
    print(f"\n🎯 DEBUG COMPLETE")

# Run the debug script
if __name__ == "__main__":
    debug_photo_conversion()