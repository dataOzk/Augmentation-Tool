import cv2
import numpy as np
import random
import argparse
import shutil
import os




def normalize_coordinates(coord, size):
    return coord / size



def denormalize_coordinates(coord, size):
    return int(coord * size)



def create_mask_from_polygon(img_size, polygon_coords):
    mask = np.zeros(img_size, dtype=np.uint8)

    # Reshape the polygon coordinates to (num_points, 2)
    polygon_coords = np.array(polygon_coords).reshape(-1, 2)

    # Denormalize the coordinates
    img_height, img_width = img_size
    denormalized_polygon = [(denormalize_coordinates(x, img_width), denormalize_coordinates(y, img_height)) for x, y in polygon_coords]

    # Convert denormalized polygon to NumPy array
    pts = np.array(denormalized_polygon, dtype=np.int32)

    # Create a binary mask
    cv2.fillPoly(mask, [pts], color=255)  # Use color=1 to set the mask values to 1

    return mask



def adjust_mask(mask, x_start, x_end, y_start, y_end):
    adjusted_mask = mask[y_start:y_end, x_start:x_end].copy()
    return adjusted_mask



def get_contours(mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours



def normalize_contour_coordinates(contours, img_size):
    normalized_contours = []

    for contour in contours:
        normalized_contour = []
        for point in contour:
            x, y = point[0]  # Access the single element directly
            normalized_x = normalize_coordinates(x, img_size[1])  # Normalize x-coordinate
            normalized_y = normalize_coordinates(y, img_size[0])  # Normalize y-coordinate
            normalized_contour.extend([normalized_x, normalized_y])
        normalized_contours.append(normalized_contour)

    return normalized_contours



def random_zoom(image_path, area_size, save_original):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Get the original height and width
    original_height, original_width = img.shape[:2]

    # Choose a random pixel
    random_pixel = (random.randint(area_size // 2, original_width - 1 - (area_size // 2)),
                    random.randint(area_size // 2, original_height - 1 - (area_size // 2)))

    # Calculate the selected area boundaries
    x_start = random_pixel[0] - (area_size // 2)
    x_end = x_start + area_size
    y_start = random_pixel[1] - (area_size // 2)
    y_end = y_start + area_size

    if x_start >= x_end or y_start >= y_end:
        print("Selected area size is larger than the image. Skipping.")
        return

    # Prepare output paths
    file_name = os.path.basename(image_path)
    base_name, extension = file_name.rsplit('.', 1)
    label_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
    output_image_name = f"{base_name}Zoomed.jpg"
    output_label_name = f"{base_name}Zoomed.txt"
    output_image_path = os.path.join(os.path.dirname(image_path), output_image_name)
    output_label_path = os.path.join(os.path.dirname(image_path), output_label_name)
    

    

    # Read and adjust labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
        adjusted_lines = []
        for line in lines:
            data = line.split()
            class_index = int(data[0])
            polygon_coords = [float(coord) for coord in data[1:]]
            original_img_size = (original_height, original_width)
            mask = create_mask_from_polygon(original_img_size, polygon_coords)
            adjusted_mask = adjust_mask(mask, x_start, x_end, y_start, y_end)
            contours = get_contours(adjusted_mask)

            if contours:
                normalized_contours = normalize_contour_coordinates(contours, (x_end - x_start, y_end - y_start))

                adjusted_line = f"{class_index} {' '.join(map(str, normalized_contours[0]))}\n"
                adjusted_lines.append(adjusted_line)
            

    # Save the cropped image as the output zoomed image
    cropped_img = img[y_start:y_end, x_start:x_end]
    cv2.imwrite(output_image_path, cropped_img)

    # Write the adjusted labels to the output label file
    with open(output_label_path, 'w') as f:
        f.writelines(adjusted_lines)

    # Delete the original files if save_original is False
    if not save_original:
        os.remove(image_path)
        os.remove(label_path)

    return output_image_path, output_label_path



def clip_coordinates(value):
    return max(0, min(1, value))



def pad_and_resize_image(image_path, output_size, save_original):
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Get the original height and width
    original_height, original_width = img.shape[:2]

    # Determine the target size
    target_size = max(original_height, original_width)

    # Calculate padding values
    pad_top = max(0, (target_size - original_height) // 2)
    pad_bottom = max(0, target_size - original_height - pad_top)
    pad_left = max(0, (target_size - original_width) // 2)
    pad_right = max(0, target_size - original_width - pad_left)

    # Add black padding only to the shorter edge
    padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Resize the padded image to the target size
    resized_img = cv2.resize(padded_img, (output_size, output_size))

# Prepare output paths
    file_name = os.path.basename(image_path)
    base_name, extension = file_name.rsplit('.', 1)
    label_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
    output_image_name = f"{base_name}Padded.jpg"
    output_label_name = f"{base_name}Padded.txt"
    output_image_path = os.path.join(os.path.dirname(image_path), output_image_name)
    output_label_path = os.path.join(os.path.dirname(image_path), output_label_name)

    # Save the modified image
    cv2.imwrite(output_image_path, resized_img)

    # Read and adjust labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    adjusted_lines = []

    for line in lines:
        data = line.split()
        class_index = int(data[0])

        # Adjust segmentation mask coordinates
        adjusted_segmentation = []
        for i in range(1, len(data), 2):
            x_coord = clip_coordinates((float(data[i]) * original_width + pad_left) / (original_width + pad_left + pad_right))
            y_coord = clip_coordinates((float(data[i + 1]) * original_height + pad_top) / (original_height + pad_top + pad_bottom))
            adjusted_segmentation.extend([x_coord, y_coord])

        # Adjusted bounding box information
        adjusted_line = f"{class_index}"

        # Adjusted segmentation mask coordinates
        adjusted_line += " " + " ".join(map(str, adjusted_segmentation))

        adjusted_lines.append(adjusted_line + "\n")

    # Write the adjusted labels to the output label file
    with open(output_label_path, 'w') as f:
        f.writelines(adjusted_lines)

    # Delete the original files
    if not save_original:
        os.remove(image_path)
        os.remove(label_path)

    return output_image_path, output_label_path



def blur_image(image_path, blur_margin, save_original):
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(img, (blur_margin, blur_margin), 0)
    

    # Prepare output paths in the same folder as the original image
    file_name = os.path.basename(image_path)
    base_name, extension = file_name.rsplit('.', 1)
    label_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
    output_image_name = f"{base_name}Blurred.jpg"
    output_label_name = f"{base_name}Blurred.txt"
    output_image_path = os.path.join(os.path.dirname(image_path), output_image_name)
    output_label_path = os.path.join(os.path.dirname(image_path), output_label_name)

    # Save the blurred image
    cv2.imwrite(output_image_path, blurred_img)

    # Copy the original label file to the output label file for the blurred image
    
    shutil.copy(label_path, output_label_path)

    # Delete the original files
    if not save_original:
        os.remove(image_path)
        os.remove(label_path)

    return output_image_path, output_label_path



def flip_image_and_labels(image_path, flip_mode, save_original):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    file_name = os.path.basename(image_path)
    base_name, extension = file_name.rsplit('.', 1)
    label_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
    output_image_name = f"{base_name}Flipped.jpg"
    output_label_name = f"{base_name}Flipped.txt"
    output_image_path = os.path.join(os.path.dirname(image_path), output_image_name)
    output_label_path = os.path.join(os.path.dirname(image_path), output_label_name)



    with open(label_path, 'r') as f:
        lines = f.readlines()

    adjusted_lines = []

    for line in lines:
        data = line.split()
        class_index = int(data[0])

        # Adjust segmentation mask coordinates
        adjusted_segmentation = []
        for i in range(1, len(data), 2):
            if flip_mode == "horizontal":
                x_coord = clip_coordinates(1 - float(data[i]))
                y_coord = clip_coordinates(float(data[i + 1]))
            elif flip_mode == "vertical":
                x_coord = clip_coordinates(float(data[i]))
                y_coord = clip_coordinates(1 - float(data[i + 1]))
            elif flip_mode == "both":
                x_coord = clip_coordinates(1 - float(data[i]))
                y_coord = clip_coordinates(1 - float(data[i + 1]))
            else:
                raise ValueError("Invalid flip_mode. Use 'horizontal', 'vertical', or 'both'.")

            adjusted_segmentation.extend([x_coord, y_coord])

        # Adjusted bounding box information
        adjusted_line = f"{class_index} {' '.join(map(str, adjusted_segmentation))}\n"
        adjusted_lines.append(adjusted_line)


    # Write the adjusted labels to the output label file
    with open(output_label_path, 'w') as f:
        f.writelines(adjusted_lines)

    # Flip the image
    flipped_img = cv2.flip(img, -1 if flip_mode == 'both' else 1 if flip_mode == 'horizontal' else 0)

    # Save the flipped image
    cv2.imwrite(output_image_path, flipped_img)

    # Delete the original files
    if not save_original:
        os.remove(image_path)
        os.remove(label_path)

    return output_image_path, output_label_path



def adjust_brightness(image_path, brightness_factor, save_original):
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Set label path based on the image path


    file_name = os.path.basename(image_path)
    base_name, extension = file_name.rsplit('.', 1)
    label_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
    output_image_name_brighter = f"{base_name}Brighter.jpg"
    output_label_name_brighter = f"{base_name}Brighter.txt"
    output_image_path_brighter = os.path.join(os.path.dirname(image_path), output_image_name_brighter)
    output_label_path_brighter = os.path.join(os.path.dirname(image_path), output_label_name_brighter)
    output_image_name_darker = f"{base_name}Darker.jpg"
    output_label_name_darker = f"{base_name}Darker.txt"
    output_image_path_darker = os.path.join(os.path.dirname(image_path), output_image_name_darker)
    output_label_path_darker = os.path.join(os.path.dirname(image_path), output_label_name_darker)


    # Adjust brightness
    brighter_img = cv2.convertScaleAbs(img, alpha=1 + brightness_factor)
    darker_img = cv2.convertScaleAbs(img, alpha=1 - brightness_factor)

    # Save the brighter and darker images
    cv2.imwrite(output_image_path_brighter, brighter_img)
    cv2.imwrite(output_image_path_darker, darker_img)

    # Copy the original label file to the output label file for the brighter image
    shutil.copy(label_path, output_label_path_brighter)

    # Copy the original label file to the output label file for the darker image
    shutil.copy(label_path, output_label_path_darker)

    # Delete the original files if save_original is False
    if not save_original:
        os.remove(image_path)
        os.remove(label_path)

    return output_image_path_brighter, output_label_path_brighter, output_image_path_darker, output_label_path_darker
 


def convert_to_grayscale(image_path, save_original):
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    
    # Convert to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    file_name = os.path.basename(image_path)
    base_name, extension = file_name.rsplit('.', 1)
    label_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
    output_image_name = f"{base_name}Grayscaled.jpg"
    output_label_name = f"{base_name}Grayscaled.txt"
    output_image_path = os.path.join(os.path.dirname(image_path), output_image_name)
    output_label_path = os.path.join(os.path.dirname(image_path), output_label_name)

    # Save the grayscale image
    cv2.imwrite(output_image_path, grayscale_img)

    # Copy the original label file to the output label file for the grayscale image
    shutil.copy(label_path, output_label_path)

    # Delete the original files if save_original is False
    if not save_original:
        os.remove(image_path)
        os.remove(label_path)

    return output_image_path, output_label_path



def adjust_exposure(image_path, exposure_factor, contrast_factor, save_original):
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")


    # Adjust brightness for the brighter image
    brighter_img = cv2.convertScaleAbs(img, alpha=1 + exposure_factor, beta=0)

    # Adjust brightness for the darker image
    darker_img = cv2.convertScaleAbs(img, alpha=1 - exposure_factor, beta=0)

    # Apply contrast adjustment
    brighter_img = (contrast_factor * brighter_img).astype(np.uint8)
    darker_img = (contrast_factor * darker_img).astype(np.uint8)



    # Prepare output paths in the same folder as the original image
    file_name = os.path.basename(image_path)
    base_name, extension = file_name.rsplit('.', 1)
    label_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
    output_brighter_name = f"{base_name}OverExposed.jpg"
    output_darker_name = f"{base_name}UnderExposed.jpg"
    output_brighter_label_name = f"{base_name}OverExposed.txt"
    output_darker_label_name = f"{base_name}UnderExposed.txt"
    output_brighter_path = os.path.join(os.path.dirname(image_path), output_brighter_name)
    output_darker_path = os.path.join(os.path.dirname(image_path), output_darker_name)
    output_brighter_label_path = os.path.join(os.path.dirname(image_path), output_brighter_label_name)
    output_darker_label_path = os.path.join(os.path.dirname(image_path), output_darker_label_name)

    

    # Save the brighter and darker images
    cv2.imwrite(output_brighter_path, brighter_img)
    cv2.imwrite(output_darker_path, darker_img)

    # Copy the original label file to the output label files for the brighter and darker images
    shutil.copy(label_path, output_brighter_label_path)
    shutil.copy(label_path, output_darker_label_path)

    # Delete the original files if save_original is False
    if not save_original:
        os.remove(image_path)
        os.remove(label_path)

    return output_brighter_path, output_darker_path, output_brighter_label_path, output_darker_label_path



#--------------------------------------ARG-PARSER--------------------------------------



def copy_files(input_folder, output_folder):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist. Terminating.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Walk through all files and subdirectories in the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Create the full path for both the source and destination files
            source_path = os.path.join(root, file)
            destination_path = os.path.join(output_folder, file)

            # Check if the source file exists
            if not os.path.exists(source_path):
                print(f"Error: Source file '{source_path}' not found. Terminating.")
                return

            # Copy the file to the output folder
            shutil.copy2(source_path, destination_path)



import os
import shutil
import random

def split_folder(folder, train,test):
    # Create the split folders if they don't exist
    for split_folder in ['train', 'test', 'valid']:
        os.makedirs(os.path.join(folder, split_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(folder, split_folder, 'labels'), exist_ok=True)

    # Get a list of all image files (png/jpg) in the folder
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg'))]

    # Shuffle the list for randomness
    random.shuffle(image_files)

    # Calculate the number of images for each split
    total_images = len(image_files)
    train_count = int(total_images * train)
    test_count = int(total_images * test)

    # Assign images to train, test, and valid folders
    train_images = image_files[:train_count]
    test_images = image_files[train_count:train_count + test_count]
    valid_images = image_files[train_count + test_count:]

    # Move images and labels to their respective folders
    move_images_and_labels(folder, 'train', train_images)
    move_images_and_labels(folder, 'test', test_images)
    move_images_and_labels(folder, 'valid', valid_images)

def move_images_and_labels(folder, split_folder, images):
    for image in images:
        # Move image to the corresponding split folder's 'images' subfolder
        source_image_path = os.path.join(folder, image)
        destination_image_path = os.path.join(folder, split_folder, 'images', image)
        shutil.move(source_image_path, destination_image_path)

        # Move label file to the corresponding split folder's 'labels' subfolder
        label_filename = os.path.splitext(image)[0] + '.txt'
        source_label_path = os.path.join(folder, label_filename)
        destination_label_path = os.path.join(folder, split_folder, 'labels', label_filename)
        shutil.move(source_label_path, destination_label_path)


import argparse

def main():
    parser = argparse.ArgumentParser(description="Image Augmentation Tool")

    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder.")

    parser.add_argument("--random_zoom", nargs='*', type=str, help="Apply random zoom to images. Default: percentage=0.1, area_size=640, save_original=True")
    parser.add_argument("--pad_and_resize", nargs='*', type=str, help="Apply padding to images to make them square and resize them to your desired size. Default: percentage=1, output_size=640, save_original=False")
    parser.add_argument("--blur", nargs='*', type=str, help="Apply blur to images. Default: percentage=0.1, blur_margin=5(Must be odd number), save_original=True")
    parser.add_argument("--flip", nargs='*', type=str, help="Apply flip to images. Default: percentage=1, flip_mode=horizontal, save_original=True")
    parser.add_argument("--brightness", nargs='*', type=str,help="Adjust brightness of the image, creating one brighter one darker image. Default: percentage=0.1, brightness_factor=0.1, save_original=True")
    parser.add_argument("--grayscale", nargs='*', type=str, help="Grayscale images. Default: percentage=0.1, save_original=True")
    parser.add_argument("--exposure", nargs='*', type=str, help="Adjust exposure of the image, creating one underexposed one overexposed image. Default: percentage=0.1, exposure_factor=0.1, contrast_factor=1, save_original=True")


    parser.add_argument("--split", nargs='*', type=str, help="Split the dataset into test, train and validation datasets. Only enter two values first one for training second one for testing, the rest will be validation. Default train=0.7, test=0.1")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    copy_files(input_folder, output_folder)


    # PAD AND RESIZE   
    pad_and_resize_args = args.pad_and_resize
    if pad_and_resize_args is not None:
        if not pad_and_resize_args:
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * 1)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                pad_and_resize_image(os.path.join(output_folder, file), 640, False)
        else:
            percentage = float(pad_and_resize_args[0])
            output_size = int(pad_and_resize_args[1])
            save_original = True if pad_and_resize_args[2].lower() == "true" else False
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * percentage)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                pad_and_resize_image(os.path.join(output_folder, file), output_size, save_original)




    # RANDOM ZOOM
    random_zoom_args = args.random_zoom
    if random_zoom_args is not None:
        if not random_zoom_args:
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * 0.1)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                random_zoom(os.path.join(output_folder, file), 640, True)
        else:
            percentage = float(random_zoom_args[0])
            output_size = int(random_zoom_args[1])
            save_original = True if random_zoom_args[2].lower() == "true" else False
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * percentage)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                random_zoom(os.path.join(output_folder, file), output_size, save_original)



    # BLUR
    blur_args = args.blur
    if blur_args is not None:
        if not blur_args:
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * 0.1)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                blur_image(os.path.join(output_folder, file), 5 , True)
        else:
            percentage = float(blur_args[0])
            blur_margin = int(blur_args[1])
            save_original = True if blur_args[2].lower() == "true" else False
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * percentage)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                blur_image(os.path.join(output_folder, file), blur_margin, save_original)



    # FLIP
    flip_args = args.flip
    if flip_args is not None:
        if not flip_args:
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * 1)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                flip_image_and_labels(os.path.join(output_folder, file), "horizontal" , True)
        else:
            percentage = float(flip_args[0])
            flip_type = str(flip_args[1])
            save_original = True if flip_args[2].lower() == "true" else False
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * percentage)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                flip_image_and_labels(os.path.join(output_folder, file), flip_type, save_original)



    # BRIGHTNESS
    brightness_args = args.brightness
    if brightness_args is not None:
        if not brightness_args:
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * 0.1)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                adjust_brightness(os.path.join(output_folder, file), 0.6 , True)
        else:
            percentage = float(brightness_args[0])
            brightness_factor = float(brightness_args[1])
            save_original = True if brightness_args[2].lower() == "true" else False
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * percentage)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                adjust_brightness(os.path.join(output_folder, file), brightness_factor, save_original)





    # GRAYSCALE 
    grayscale_args = args.grayscale
    if grayscale_args is not None:
        if not grayscale_args:
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * 0.1)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                convert_to_grayscale(os.path.join(output_folder, file), True)
        else:
            percentage = float(grayscale_args[0])
            save_original = True if grayscale_args[1].lower() == "true" else False
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * percentage)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                convert_to_grayscale(os.path.join(output_folder, file), save_original)
     
    

    # Exposure
    exposure_args = args.exposure
    if exposure_args is not None:
        if not exposure_args:
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * 0.1)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                adjust_exposure(os.path.join(output_folder, file),0.6, 2, True)
        else:
            percentage = float(exposure_args[0])
            exposure_factor = float(exposure_args[1])
            contrast_factor = float(exposure_args[2])
            save_original = True if exposure_args[3].lower() == "true" else False
            all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith((".jpg",".png"))]
            num_images = int(len(all_files) * percentage)
            selected_files = random.sample(all_files, min(num_images, len(all_files)))
            for file in selected_files:
                adjust_exposure(os.path.join(output_folder, file), exposure_factor, contrast_factor, save_original)


    # Split
    split_args = args.split
    if split_args is not None:
        if not split_args:
            split_folder(output_folder,0.7,0.1)
        else:
            train = float(split_args[0])
            test  = float(split_args[1])
            split_folder(output_folder,train,test)


if __name__ == "__main__":
    main()


    