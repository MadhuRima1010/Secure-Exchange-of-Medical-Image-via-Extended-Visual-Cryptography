import pydicom
import os
import numpy as np
import cv2
import random
from skimage.metrics import structural_similarity as ssim
import math
from pydicom.dataset import Dataset, FileDataset
import datetime
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import hashlib

##------------------------------------------------------------------------------------------------

def compute_sha256(file_path):
    """
    Computes the SHA-256 hash of a given file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        for byte_block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_hash_to_file(dicom_file_path, hash_file_path):
    """
    Computes SHA-256 hash of the original DICOM file and saves it in a text file.
    """
    dicom_hash = compute_sha256(dicom_file_path)
    with open(hash_file_path, "w") as hash_file:
        hash_file.write(dicom_hash)
    print(f"SHA-256 hash saved to {hash_file_path}")

original_dicom_path = "Dicom File Embedding/dicom_file.dcm"
hash_file_path = "Dicom File Embedding/dicom_hash.txt"

save_hash_to_file(original_dicom_path, hash_file_path)

##------------------------------------------------------------------------------------------------


def extract_metadata_from_dicom(ds):
    """
    Extracts all metadata from a DICOM file.
    """
    metadata = {
        "Patient ID": ds.get("PatientID", "N/A"),
        "Patient Name": ds.get("PatientName", "N/A"),
        "Issuer of Patient ID": ds.get("IssuerOfPatientID", "N/A"),
        "Study Instance UID": ds.get("StudyInstanceUID", "N/A"),
        "Series Instance UID": ds.get("SeriesInstanceUID", "N/A"),
        "SOP Instance UID": ds.get("SOPInstanceUID", "N/A"),
        "Accession Number": ds.get("AccessionNumber", "N/A"),
        "Study Date": ds.get("StudyDate", "N/A"),
        "Study Time": ds.get("StudyTime", "N/A"),
        "Modality": ds.get("Modality", "N/A"),
        "Referring Physician": ds.get("ReferringPhysicianName", "N/A"),
        "Study Description": ds.get("StudyDescription", "N/A"),
        "Series Description": ds.get("SeriesDescription", "N/A"),
        "Station Name": ds.get("StationName", "N/A"),
        "Manufacturer": ds.get("Manufacturer", "N/A"),
        "Manufacturer Model Name": ds.get("ManufacturerModelName", "N/A"),
        "Institution Name": ds.get("InstitutionName", "N/A"),
        "Institution Address": ds.get("InstitutionAddress", "N/A"),
        "Referring Physician Name": ds.get("ReferringPhysicianName", "N/A"),
    }

    return metadata

def save_metadata_as_text(metadata, output_path):
    """
    Saves extracted metadata into a text file.
    """
    with open(output_path, 'w', encoding='utf-8') as text_file:
        for key, value in metadata.items():
            text_file.write(f"{key}: {value}\n")
    print(f"Metadata saved to: {output_path}")


def read_dicom_file(filepath, output_path):
    '''
    Read a DICOM file and save the pixel data (Medical Image) as a 16-bit PNG image using OpenCV.
    '''
    # Check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"DICOM file not found: {filepath}")

    # Read the DICOM file
    ds = pydicom.dcmread(filepath)

    # Extract pixel data
    if 'PixelData' in ds:
        print(f"Data type of the DICOM image: {ds.pixel_array.dtype}")  # Print dtype

        pixel_array = ds.pixel_array  # Keep precision

        # Normalize to 16-bit range (0-65535) if needed
        if pixel_array.max() > 0:
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 65535

        pixel_array = pixel_array.astype(np.uint16)  # Convert to 16-bit
        
        # Save as 16-bit PNG using OpenCV
        cv2.imwrite(output_path, pixel_array)

        medical_image = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)  # To check if the image was saved correctly
        
        cv2.imshow('Medical Image From DICOM File', medical_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"DICOM image saved as PNG at: {output_path}")
        
        output_metadata_text_path = "Dicom File Embedding/dicom_metadata.txt"

        metadata = extract_metadata_from_dicom(ds)

        # Save metadata to text file
        save_metadata_as_text(metadata, output_metadata_text_path)
        
    else:
        print("No pixel data found in the DICOM file.")

dicom_filepath = 'Dicom File Embedding/dicom_file.dcm'
dicom_image_output_path = 'Dicom File Embedding/Extracted_Dicom_Medical_Image.png'

read_dicom_file(dicom_filepath, dicom_image_output_path)

##------------------------------------------------------------------------------------------------

def medical_image_preprocessing(medical_image_path, preprocessed_image_output_path):
    '''
    Preprocess the medical image from the dicom file using OpenCV.
    '''
    
    medical_image = cv2.imread(medical_image_path, cv2.IMREAD_UNCHANGED)

    # Resize first
    medical_image = cv2.resize(medical_image.copy(), (255, 255), interpolation=cv2.INTER_LINEAR)

    # Convert to 12-bit
    medical_image_12bit = medical_image.copy() >> 4  # Right shift 4 bits
    
    medical_image_12bit = np.clip(medical_image_12bit.copy(), 0, 4095)  # Ensure within range
    
    medical_image_12bit = medical_image_12bit.astype(np.uint16)
    print(f"The dtype of the medical image {medical_image_12bit.dtype}")
    
    print(f"Is preprocessed medical image 12 bit?: {medical_image.dtype == np.uint16 and np.max(medical_image) <= 4095:}")
    
    cv2.imwrite(preprocessed_image_output_path, medical_image_12bit.copy())
    cv2.imwrite('Dicom File Embedding/Medical_Image_8Bit.png', (medical_image_12bit.copy() / 16).astype(np.uint8))
    
    cv2.imshow('Preprocessed Medical Image', (medical_image_12bit.copy() / 16).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
dicom_medical_image_path = 'Dicom File Embedding/Extracted_Dicom_Medical_Image.png'
medical_image_12bit_path = 'Dicom File Embedding/Medical_Image_12Bit.png'

medical_image_preprocessing(dicom_medical_image_path, medical_image_12bit_path)

## ------------------------------------------------------------------------------------------------

def cover_image_preprocessing(cover_image_path, preprocessed_cover_image_output_path):
    '''
    Preprocess the cover image using OpenCV.
    '''
    
    cover_image = cv2.imread(cover_image_path, cv2.IMREAD_UNCHANGED)

    cover_image = cv2.resize(cover_image, (255, 255), interpolation=cv2.INTER_LINEAR)

    cover_image_24bit = None

    # Check if the image is already 24-bit (3 channels, 8 bits each)
    if cover_image is None:
        print(f"Error: Unable to load the cover image from {cover_image_path}. Please check the file path and file integrity.")
    else:
        if cover_image.shape[2] == 3 and cover_image.dtype == 'uint8':
            print("The image is already in 24-bit format.")
            cover_image_24bit = cover_image.copy()  # Assign cover_image directly if already 24-bit
        else:
            # Convert image to 24-bit by ensuring 3 channels and 8-bit depth
            cover_image_24bit = cv2.cvtColor(cover_image.copy(), cv2.COLOR_GRAY2BGR) if len(cover_image.shape) == 2 else cover_image

    # Save the 24-bit image
    cv2.imwrite(preprocessed_cover_image_output_path, cover_image_24bit)

    print("Cover Image dtype:", cover_image_24bit.dtype)

    # Display the 24-bit image
    cv2.imshow("24-bit Image", cover_image_24bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cover_image_path = 'Dicom File Embedding/cover_image.jpg'
cover_image_24bit_path = 'Dicom File Embedding/Cover_Image_24Bit.png'

cover_image_preprocessing(cover_image_path, cover_image_24bit_path)

## ------------------------------------------------------------------------------------------------


# Function to perform circular shift encryption
def circular_right_shift(matrix):
    return np.roll(matrix, 1, axis=1)

def circular_down_shift(matrix):
    return np.roll(matrix, 1, axis=0)

def circular_shift_encrypt(matrix, key, encrypted_image_path):
    '''
    Perform circular shift encryption on the medical image matrix using the given key.
    '''
    
    encrypted_matrix = matrix.copy()
    for i, shift_value in enumerate(key):
        if shift_value == 0:
            continue
        elif i % 2 == 0:
            for _ in range(shift_value):
                encrypted_matrix = circular_right_shift(encrypted_matrix)
        else:
            for _ in range(shift_value):
                encrypted_matrix = circular_down_shift(encrypted_matrix)

    print("Encrypted matrix values after shifts:")
    print(encrypted_matrix[:5, :5])  # Print the first few pixels of the encrypted matrix
           
    # Save and Display the encrypted image (scaled for 8-bit representation)
    cv2.imwrite(encrypted_image_path, encrypted_matrix)
    cv2.imwrite('Dicom File Embedding/Encrypted_Medical_Image_8Bit.png', (encrypted_matrix / 16).astype(np.uint8))
    
    cv2.imshow("Encrypted Medical Image", (encrypted_matrix / 16).astype(np.uint8))  # Convert back to uint8 for display
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to generate a random key for encryption
def generate_random_key(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]

# Load the preprocessed medical image
medical_image_12bit_path = 'Dicom File Embedding/Medical_Image_12Bit.png'
medical_image_12bit = cv2.imread(medical_image_12bit_path, cv2.IMREAD_UNCHANGED)

encrypted_medical_image_path = 'Dicom File Embedding/Encrypted_Medical_Image.png'

key_length = 25
max_shift_value = max(medical_image_12bit.copy().shape) - 1
key = generate_random_key(key_length, max_shift_value)

# Perform circular shift encryption
circular_shift_encrypt(medical_image_12bit.copy(), key, encrypted_medical_image_path)

##------------------------------------------------------------------------------------------------

def extract_msb_lsb(encrypted_image_path, msb_4_image_path , lsb_8_image_path):
    '''
    Extract the 4 MSB and 8 LSB from the encrypted medical image.
    '''
    encrypted_image = cv2.imread(encrypted_image_path, cv2.IMREAD_UNCHANGED)

    msb_4 = encrypted_image.copy() >> 8

    lsb_8 = encrypted_image.copy() & 0xFF

    # Scale 4 MSB and 8 LSB images to 8-bit for display
    msb_4_8bit = (msb_4.copy() * 16).astype(np.uint8)  # Multiply by 16 to fit the 8-bit range
    lsb_8_8bit = lsb_8.copy().astype(np.uint8)  # Already in 8-bit range

    # Save the 4 MSB and 8 LSB images
    cv2.imwrite(msb_4_image_path, msb_4)
    cv2.imwrite('Dicom File Embedding/4_MSB_Image_8bit.png', msb_4_8bit)
    cv2.imwrite(lsb_8_image_path, lsb_8)
    cv2.imwrite('Dicom File Embedding/8_LSB_Image_8bit.png', lsb_8_8bit)

    # Display the 4 MSB and 8 LSB images
    cv2.imshow("4 MSB Encrypted Medical Image", msb_4_8bit)
    cv2.imshow("8 LSB Encrypted Medical Image", lsb_8_8bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"4 msb image dtype: {msb_4.dtype}")
    print(f"8 lsb image dtype: {lsb_8.dtype}")

# Load the 12-bit encrypted image
encrypted_medical_image_path = 'Dicom File Embedding/Encrypted_Medical_Image.png'

msb_4_image_path = 'Dicom File Embedding/4_MSB_Image.png'
lsb_8_image_path = 'Dicom File Embedding/8_LSB_Image.png'

extract_msb_lsb(encrypted_medical_image_path, msb_4_image_path, lsb_8_image_path)

##------------------------------------------------------------------------------------------------


# # Visualization function
# def visualize_embedding(cover_image, embedding_positions):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     channels = ['R', 'G', 'B']
    
#     for i, channel in enumerate(channels):
#         img = np.copy(cover_image[:, :, i])  # Copy channel for visualization
        
#         for block, pixels in embedding_positions[channel].items():
#             for x, y in pixels:
#                 img[x, y] = 255  # Highlight embedded pixels in white
        
#         axes[i].imshow(img, cmap='gray')
#         axes[i].set_title(f'{channel} Channel Embedding')
#         axes[i].axis('off')
    
#     plt.show()

# Combined embedding function for MSB and LSB
def embed_msb_lsb_into_cover(cover_image, msb_array, lsb_array):
    msb_positions = {'R': {}, 'G': {}, 'B': {}}
    lsb_positions = {'R': {}, 'G': {}, 'B': {}}

    block_size = 3
    num_blocks = cover_image.shape[0] // block_size
    msb_index = 0
    lsb_index = 0

    flat_msb_array = msb_array.flatten()
    flat_lsb_array = lsb_array.flatten()

    for i in range(num_blocks):
        for j in range(num_blocks):
            if msb_index >= len(flat_msb_array) and lsb_index >= len(flat_lsb_array):
                break

            start_x = i * block_size
            start_y = j * block_size
            pixel_indices = [(x, y) for x in range(start_x, start_x + block_size)
                                      for y in range(start_y, start_y + block_size)]

            # Embed MSB values
            selected_pixels = random.sample(pixel_indices, 3)
            for channel, color in enumerate(['R', 'G', 'B']):
                for pixel in selected_pixels:
                    if msb_index < len(flat_msb_array):
                        x, y = pixel
                        msb_value = flat_msb_array[msb_index]
                        cover_image[x, y, channel] = (cover_image[x, y, channel] & 0xF0) | msb_value
                        if (i, j) not in msb_positions[color]:
                            msb_positions[color][(i, j)] = []

                        msb_positions[color][(i, j)].append((x, y))
                        msb_index += 1

            for channel, color in enumerate(['R', 'G', 'B']):
                msb_used_positions = msb_positions[color].get((i, j), [])
                available_pixels = list(set(pixel_indices) - set(msb_used_positions))

                if len(available_pixels) >= 3:
                    selected_pixels = random.sample(available_pixels, 3)
                    for pixel in selected_pixels:
                        if lsb_index < len(flat_lsb_array):                              
                            # Track LSB position
                            if (i, j) not in lsb_positions[color]:
                                lsb_positions[color][(i, j)] = []
                                
                            x, y = pixel
                            lsb_value = int(flat_lsb_array[lsb_index])

                            # Replace the pixel value with the LSB value
                            cover_image[x, y, channel] = lsb_value

                            # Calculate the average with unmodified neighboring pixels
                            neighbors = [(nx, ny) for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
                                                if start_x <= nx < start_x + block_size and 
                                                    start_y <= ny < start_y + block_size and
                                                    (nx, ny) not in msb_used_positions and
                                                    (nx, ny) not in selected_pixels]
                            
                            if neighbors:             
                                neighbor_values = [int(cover_image[nx, ny, channel]) for nx, ny in neighbors]
                                
                                num_neighbours = len(neighbor_values)
                                averaged_value = int((int(cover_image[x, y, channel]) + sum(neighbor_values)) / (num_neighbours + 1))
                                
                                # print(f"Neighbors: {neighbors + [(x, y)]} and values: {neighbor_values} and lsb value was {lsb_value} and current pixel value: {cover_image[x, y, channel]} and averaged value: {averaged_value}")
                                cover_image[x, y, channel] = min(max(averaged_value, 0), 254)

                            lsb_positions[color][(i, j)].append((x, y))
                            lsb_index += 1
                        
    # visualize_embedding(cover_image, msb_positions)
    # visualize_embedding(cover_image, lsb_positions)
    return cover_image, msb_positions, lsb_positions

cover_image_24bit_path = 'Dicom File Embedding/Cover_Image_24Bit.png'
msb_4_image_path = 'Dicom File Embedding/4_MSB_Image.png'
lsb_8_image_path = 'Dicom File Embedding/8_LSB_Image.png'

cover_image_24bit = cv2.imread(cover_image_24bit_path, cv2.IMREAD_UNCHANGED)
msb_4 = cv2.imread(msb_4_image_path, cv2.IMREAD_UNCHANGED)
lsb_8 = cv2.imread(lsb_8_image_path, cv2.IMREAD_UNCHANGED)

cover_image_with_msb_lsb, tracking_info_msb, tracking_info_lsb = embed_msb_lsb_into_cover(cover_image_24bit.copy(), msb_4, lsb_8)

print("Cover Image With MSB and LSB shape", cover_image_with_msb_lsb.shape)

# Save and display the new cover image
cover_image_with_msb_lsb_path = 'Dicom File Embedding/Cover_Image_With_MSB_LSB.png'

cv2.imwrite(cover_image_with_msb_lsb_path, cover_image_with_msb_lsb)
cv2.imshow("Cover Image With MSB and LSB", cover_image_with_msb_lsb)
cv2.waitKey(0)
cv2.destroyAllWindows()

##------------------------------------------------------------------------------------------------

# Metadata embedding
def embed_metadata_into_cover(cover_image, msb_positions, lsb_positions, dicom_metadata_text):
    """
    Embeds metadata into the cover image without modifying MSB/LSB embedded data.
    Tracks only pixel positions and channels used, without storing pixel values.
    """
    height, width, _ = cover_image.shape
    all_pixel_positions = [(i, j) for i in range(height) for j in range(width)]

    # Collect all MSB/LSB modified pixel positions
    occupied_pixels = set()
    for color in ['R', 'G', 'B']:
        for pos_list in msb_positions[color].values():
            occupied_pixels.update(pos_list)
        for pos_list in lsb_positions[color].values():
            occupied_pixels.update(pos_list)

    # Ensure metadata pixels are selected from unmodified pixels only
    available_pixels = list(set(all_pixel_positions) - occupied_pixels)
    np.random.shuffle(available_pixels)  # Randomize selection

    if len(available_pixels) < len(dicom_metadata_text) * 2:
        raise ValueError("Not enough available pixels to embed the metadata.")

    metadata_ascii = [ord(char) for char in dicom_metadata_text]  # Convert metadata to ASCII
    metadata_positions = []  # Store positions and channels used for metadata
    index = 0  # Tracking embedding position
    for char in metadata_ascii:
        if index >= len(available_pixels) - 1:
            raise ValueError("Not enough pixels to embed the metadata.")

        high_4_bits = (char >> 4) & 0xF  # First 4 bits
        low_4_bits = char & 0xF  # Last 4 bits

        # Select two available pixels
        (x1, y1) = available_pixels[index]
        (x2, y2) = available_pixels[index + 1]
        index += 2  # Move to next letter

        # Select random channels (ensuring we distribute across R, G, B)
        channel1 = random.choice([0, 1, 2])  # 0 = R, 1 = G, 2 = B
        channel2 = random.choice([0, 1, 2])

        # Modify only the last 4 bits of the selected channels
        cover_image[x1, y1, channel1] = (cover_image[x1, y1, channel1] & 0xF0) | high_4_bits
        cover_image[x2, y2, channel2] = (cover_image[x2, y2, channel2] & 0xF0) | low_4_bits

        # Store only the positions and channels (not pixel values)
        metadata_positions.append({"pixel1": (x1, y1, channel1), "pixel2": (x2, y2, channel2)})

    print(f"Metadata successfully embedded into the cover image.")
    return cover_image, metadata_positions

cover_image_with_msb_lsb_path = 'Dicom File Embedding/Cover_Image_With_MSB_LSB.png'
cover_image_with_msb_lsb = cv2.imread(cover_image_with_msb_lsb_path, cv2.IMREAD_UNCHANGED)

dicom_metadata_text_path = 'Dicom File Embedding/dicom_metadata.txt'

with open(dicom_metadata_text_path, 'r', encoding='utf-8') as text_file:
    dicom_metadata_text = text_file.read()
    
cover_image_with_metadata, metadata_positions = embed_metadata_into_cover(cover_image_with_msb_lsb.copy(), tracking_info_msb, tracking_info_lsb, dicom_metadata_text)

# Save and display the new cover image with metadata
cover_image_with_metadata_path = 'Dicom File Embedding/Cover_Image_With_Metadata.png'
cv2.imwrite(cover_image_with_metadata_path, cover_image_with_metadata)
cv2.imshow("Cover Image With Metadata", cover_image_with_metadata)
cv2.waitKey(0)
cv2.destroyAllWindows()

##------------------------------------------------------------------------------------------------

# Metadata extraction
def extract_metadata_from_cover(embedded_image, metadata_positions, output_text_file):
    """
    Extracts metadata from the cover image using only the metadata_positions variable.
    Saves the extracted metadata to a text file.
    """
    if embedded_image is None:
        raise ValueError("Embedded image is not loaded.")

    extracted_ascii_values = []

    for entry in metadata_positions:
        x1, y1, channel1 = entry["pixel1"]
        x2, y2, channel2 = entry["pixel2"]

        # Extract 4-bit values from the stored positions
        high_4_bits = embedded_image[x1, y1, channel1] & 0x0F
        low_4_bits = embedded_image[x2, y2, channel2] & 0x0F

        char_value = (high_4_bits << 4) | low_4_bits
        extracted_ascii_values.append(chr(char_value))

    # Convert extracted ASCII values back into text
    extracted_metadata = "".join(extracted_ascii_values)

    # Save extracted metadata to a text file
    with open(output_text_file, 'w', encoding='utf-8') as text_file:
        text_file.write(extracted_metadata)

    print(f"Extracted metadata saved to: {output_text_file}")
    return extracted_metadata

cover_image_with_metadata_path = 'Dicom File Embedding/Cover_Image_With_Metadata.png'
output_metadata_text_path = 'Dicom File Embedding/extracted_metadata.txt'

cover_image_with_metadata = cv2.imread(cover_image_with_metadata_path, cv2.IMREAD_UNCHANGED)

extracted_metadata = extract_metadata_from_cover(cover_image_with_metadata.copy(), metadata_positions, output_metadata_text_path)

##------------------------------------------------------------------------------------------------

# Extraction function for MSB
def extract_msb_from_cover(cover_image, embedding_positions, msb_shape=(255,255)):
    extracted_msb = np.zeros(msb_shape, dtype=np.uint8)
    flat_extracted = extracted_msb.flatten()
    msb_index = 0

    for (block_i, block_j), _ in embedding_positions['R'].items():
        for channel, color in enumerate(['R', 'G', 'B']):
            if (block_i, block_j) in embedding_positions[color]:
                for x, y in embedding_positions[color][(block_i, block_j)]:
                    if msb_index < len(flat_extracted):
                        msb_value = cover_image[x, y, channel] & 0x0F
                        flat_extracted[msb_index] = msb_value
                        msb_index += 1

    return flat_extracted.reshape(msb_shape)

# Extraction function for LSB
def extract_lsb_from_cover(cover_image, msb_positions, lsb_positions, lsb_shape=(255,255)):
    extracted_lsb = np.zeros(lsb_shape, dtype=np.uint8)
    flat_extracted = extracted_lsb.flatten()
    lsb_index = 0
    block_size = 3

    for (block_i, block_j), _ in lsb_positions['R'].items():  # Process each block
        for channel, color in enumerate(['R', 'G', 'B']):
            if (block_i, block_j) in lsb_positions[color]:  # Process the specific color channel
                for x, y in lsb_positions[color][(block_i, block_j)]:
                    start_x = block_i * block_size
                    start_y = block_j * block_size

                    # print(f"Processing pixel ({x}, {y}) in block ({block_i}, {block_j}) and block start position ({start_x}, {start_y})")
                    if lsb_index < len(flat_extracted):
                        lsb_value = int(cover_image[x, y, channel]) # Extract pixel value
                        
                        # Identify unmodified neighboring pixels inside the block
                        neighbors = [(nx, ny) for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
                                    if start_x <= nx < start_x + block_size 
                                    and start_y <= ny < start_y + block_size
                                    and (nx, ny) not in msb_positions[color].get((block_i, block_j), [])
                                    and (nx, ny) not in lsb_positions[color][(block_i, block_j)]]  # Ensure not modified by MSB in the same channel

                        if neighbors:
                            # Extract unmodified neighbor values
                            neighbor_values = [int(cover_image[nx, ny, channel]) for nx, ny in neighbors]
                            
                            # Reverse the averaging operation to get the original LSB
                            num_neighbors = len(neighbor_values)
                            original_lsb = max(0, min(254, ((num_neighbors + 1) * lsb_value) - sum(neighbor_values)))
                            
                            flat_extracted[lsb_index] = original_lsb  # Store in the flat array
                        else:
                            flat_extracted[lsb_index] = lsb_value  # Store without correction if no neighbors available
                    
                        lsb_index += 1

    return flat_extracted.reshape(lsb_shape)

# No need to reset the pixels actually.....
def reset_cover_image(cover_image, msb_positions, lsb_positions):
    for channel, color in enumerate(['R', 'G', 'B']):
        for (_, _), pixels in msb_positions[color].items():
            for x, y in pixels:
                cover_image[x, y, channel] &= 0xF0 # Reset MSB bits to 0

        for (_, _), pixels in lsb_positions[color].items():
            for x, y in pixels:
                cover_image[x, y, channel] &= 0  # Reset LSB-embedded pixels to 0

    return cover_image

# cover_image_with_msb_lsb_path = 'Dicom File Embedding/Cover_Image_With_MSB_LSB.png'
cover_image_with_metadata_path = 'Dicom File Embedding/Cover_Image_With_Metadata.png'
cover_image_with_metadata = cv2.imread(cover_image_with_metadata_path, cv2.IMREAD_UNCHANGED)

extracted_msb_4 = extract_msb_from_cover(cover_image_with_metadata.copy(), tracking_info_msb)
extracted_lsb_8 = extract_lsb_from_cover(cover_image_with_metadata.copy(), tracking_info_msb, tracking_info_lsb)
extracted_cover_image = reset_cover_image(cover_image_with_metadata.copy(), tracking_info_msb, tracking_info_lsb)

print(f"Extracted MSB 4 dtype: {extracted_msb_4.dtype}")
print(f"Extracted LSB 8 dtype: {extracted_lsb_8.dtype}")

# Save and display the extracted images
extracted_encrypted_image = (extracted_msb_4.copy().astype(np.uint16) << 8) | (extracted_lsb_8.copy().astype(np.uint16))

print(f"extracted encrypted medical image dtype: {extracted_encrypted_image.dtype}")

print(f"extracted cover image dtype: {extracted_cover_image.dtype}")

cv2.imwrite('Dicom File Embedding/Extracted_Encrypted_Medical_Image.png', extracted_encrypted_image)

cv2.imwrite('Dicom File Embedding/Extracted_Cover_Image_24bit.png', extracted_cover_image)
cv2.imwrite('Dicom File Embedding/Extracted_4_MSB_Image.png', extracted_msb_4)
cv2.imwrite('Dicom File Embedding/Extracted_4_MSB_Image_8bit.png', (extracted_msb_4 * 16).astype(np.uint8))

cv2.imwrite('Dicom File Embedding/Extracted_8_LSB_Image.png', (extracted_lsb_8.astype(np.uint8)))

cv2.imwrite('Dicom File Embedding/Extracted_Encrypted_Image.png', (extracted_lsb_8.astype(np.uint8)))
cv2.imwrite('Dicom File Embedding/Extracted_Encrypted_Image_8bit.png', (extracted_lsb_8.astype(np.uint8)))

cv2.imshow("Extracted Cover Image", extracted_cover_image)
cv2.imshow("Extracted MSB Image", (extracted_msb_4 * 16).astype(np.uint8))
cv2.imshow("Extracted LSB Image", (extracted_lsb_8.astype(np.uint8)))
cv2.imshow("Extracted Encrypted Medical Image", (extracted_encrypted_image / 16).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

##------------------------------------------------------------------------------------------------

# Function to perform circular left shift
def circular_left_shift(matrix):
    return np.roll(matrix, -1, axis=1)

# Function to perform circular up shift
def circular_up_shift(matrix):
    return np.roll(matrix, -1, axis=0)

# Function to decrypt the matrix
def circular_shift_decrypt(encrypted_matrix, key):
    decrypted_matrix = encrypted_matrix.copy()

    # Reverse the order of the key and shifts
    for i, shift_value in reversed(list(enumerate(key))):
        if shift_value == 0:
            continue
        elif i % 2 == 0:  # For right shifts in encryption, use left shifts
            for _ in range(shift_value):
                decrypted_matrix = circular_left_shift(decrypted_matrix)
        else:  # For down shifts in encryption, use up shifts
            for _ in range(shift_value):
                decrypted_matrix = circular_up_shift(decrypted_matrix)

    print("Decrypted matrix values after shifts:")
    print(decrypted_matrix[:5, :5])

    return decrypted_matrix

extracted_encrypted_image = cv2.imread('Dicom File Embedding/Extracted_Encrypted_Medical_Image.png', cv2.IMREAD_UNCHANGED)

decrypted_medical_image = circular_shift_decrypt(extracted_encrypted_image.copy(), key)

cv2.imwrite('Dicom File Embedding/Extracted_Medical_Image_12bit.png', decrypted_medical_image)
cv2.imwrite('Dicom File Embedding/Extracted_Medical_Image_12bit_8bit.png', (decrypted_medical_image / 16).astype(np.uint8))

cv2.imshow("Decrypted Medical Image", (decrypted_medical_image / 16).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

##------------------------------------------------------------------------------------------------

def rebuild_dicom(extracted_image_path, extracted_metadata_path, output_dicom_path):
    """
    Rebuilds a valid DICOM file using the extracted medical image and metadata.
    """
    # Step 1: Load the extracted medical image
    extracted_image = cv2.imread(extracted_image_path, cv2.IMREAD_UNCHANGED)
    
    if extracted_image is None:
        raise ValueError(f"Error: Failed to load extracted image from {extracted_image_path}")
    
    print(f"Extracted medical image loaded successfully. Shape: {extracted_image.shape}, Dtype: {extracted_image.dtype}")

    # Convert to 16-bit format (if necessary)
    if extracted_image.dtype != np.uint16:
        extracted_image = extracted_image.astype(np.uint16)

    # Step 2: Read the extracted metadata
    metadata = {}
    with open(extracted_metadata_path, 'r', encoding='utf-8') as meta_file:
        for line in meta_file:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                key, value = parts
                metadata[key] = value
    
    print("Extracted metadata loaded successfully.")

    # Step 3: Create DICOM File Meta Information
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    # Step 4: Create a new DICOM dataset
    ds = Dataset()

    # Assign metadata fields
    ds.PatientID = metadata.get("Patient ID", "Unknown")
    ds.PatientName = metadata.get("Patient Name", "Anonymous")
    ds.StudyInstanceUID = metadata.get("Study Instance UID", generate_uid())
    ds.SeriesInstanceUID = metadata.get("Series Instance UID", generate_uid())
    ds.SOPInstanceUID = metadata.get("SOP Instance UID", generate_uid())
    ds.AccessionNumber = metadata.get("Accession Number", "Unknown")
    ds.StudyDate = metadata.get("Study Date", datetime.datetime.now().strftime("%Y%m%d"))
    ds.StudyTime = metadata.get("Study Time", datetime.datetime.now().strftime("%H%M%S"))
    ds.Modality = metadata.get("Modality", "OT")  # "OT" (Other) if unknown
    ds.Manufacturer = metadata.get("Manufacturer", "Unknown")
    ds.StudyDescription = metadata.get("Study Description", "Rebuilt DICOM File")
    
    # Image Data
    ds.Rows, ds.Columns = extracted_image.shape[:2]
    ds.BitsAllocated = 16
    ds.BitsStored = 12  # If the original medical image was 12-bit
    ds.HighBit = 11
    ds.SamplesPerPixel = 1  # Grayscale image
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    
    # Assign pixel data
    ds.PixelData = extracted_image.tobytes()

    # Step 5: Wrap dataset in FileDataset to ensure correct DICOM format
    dicom_dataset = FileDataset(output_dicom_path, ds, file_meta=file_meta, preamble=b"\0" * 128)

    # Step 6: Save the rebuilt DICOM file
    dicom_dataset.save_as(output_dicom_path, write_like_original=False)

    print(f"Rebuilt DICOM file saved successfully at: {output_dicom_path}")

extracted_image_path = "Dicom File Embedding/Extracted_Medical_Image_12bit.png"
extracted_metadata_path = "Dicom File Embedding/extracted_metadata.txt"
output_dicom_path = "Dicom File Embedding/Rebuilt_DICOM_File.dcm"

rebuild_dicom(extracted_image_path, extracted_metadata_path, output_dicom_path)

##------------------------------------------------------------------------------------------------

def verify_dicom_integrity(rebuilt_dicom_path, hash_file_path):
    """
    Verifies the integrity of the rebuilt DICOM file by comparing its SHA-256 hash 
    with the original hash stored in a file.
    """
    # Compute SHA-256 hash of the received/rebuilt DICOM file
    received_hash = compute_sha256(rebuilt_dicom_path)

    # Read the original hash from the key file
    with open(hash_file_path, "r") as hash_file:
        original_hash = hash_file.read().strip()

    # Compare both hashes
    if received_hash == original_hash:
        print(f"DICOM file integrity verified! Hashes match.")
        print(f"SHA-256: {received_hash}")
        return True
    else:
        print(f"WARNING: DICOM file integrity compromised! Hashes do NOT match.")
        print(f"Expected SHA-256: {original_hash}")
        print(f"Received SHA-256: {received_hash}")
        return False

# Example Usage: Verify the Reconstructed DICOM File
rebuilt_dicom_path = "Dicom File Embedding/Rebuilt_DICOM_File.dcm"

verify_dicom_integrity(rebuilt_dicom_path, hash_file_path)


##------------------------------------------------------------------------------------------------

# PSNR and SSIM calculation function
def calculate_psnr(original_image, reconstructed_image):
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((original_image - reconstructed_image) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
        return float('inf')
    PIXEL_MAX = 255.0
    psnr_value = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr_value

def calculate_ssim(original_image, reconstructed_image):
    # For all images except cover image
    ssim_value = ssim(original_image, reconstructed_image)
    return ssim_value

def calculate_ssim_cover(original_image, reconstructed_image):    
    # For Cover Image (channel_axis=2: Specifies that the third axis corresponds to the color channels.)
    ssim_value = ssim(original_image, reconstructed_image, channel_axis=2)
    return ssim_value



######## MSB IMAGE
original_image = cv2.imread('Dicom File Embedding/4_MSB_Image.png', cv2.IMREAD_UNCHANGED)
reconstructed_image = cv2.imread('Dicom File Embedding/Extracted_4_MSB_Image.png', cv2.IMREAD_UNCHANGED)

if original_image.shape != reconstructed_image.shape:
    original_image = cv2.resize(original_image, reconstructed_image.shape, interpolation=cv2.INTER_AREA)

psnr_value = calculate_psnr(original_image, reconstructed_image)
ssim_value = calculate_ssim(original_image, reconstructed_image)

print(f"PSNR of MSB Image: {psnr_value}")
print(f"SSIM of MSB Image: {ssim_value}")



######## LSB IMAGE
original_image = cv2.imread('Dicom File Embedding/8_LSB_Image.png', cv2.IMREAD_UNCHANGED)
reconstructed_image = cv2.imread('Dicom File Embedding/Extracted_8_LSB_Image.png', cv2.IMREAD_UNCHANGED)

if original_image.shape != reconstructed_image.shape:
    original_image = cv2.resize(original_image, reconstructed_image.shape, interpolation=cv2.INTER_AREA)

psnr_value = calculate_psnr(original_image, reconstructed_image)
ssim_value = calculate_ssim(original_image, reconstructed_image)

print(f"PSNR of LSB Image: {psnr_value}")
print(f"SSIM of LSB Image: {ssim_value}")



######## MEDICAL IMAGE
original_image = cv2.imread('Dicom File Embedding/Medical_Image_12Bit.png', cv2.IMREAD_UNCHANGED)
reconstructed_image = cv2.imread('Dicom File Embedding/Extracted_Medical_Image_12bit.png', cv2.IMREAD_UNCHANGED)

if original_image.shape != reconstructed_image.shape:
    original_image = cv2.resize(original_image, reconstructed_image.shape, interpolation=cv2.INTER_AREA)

psnr_value = calculate_psnr(original_image, reconstructed_image)
ssim_value = calculate_ssim(original_image, reconstructed_image)

print(f"PSNR of medical image: {psnr_value}")
print(f"SSIM of medical image: {ssim_value}")


######## COVER IMAGE
original_image = cv2.imread('Dicom File Embedding/Cover_Image_24Bit.png', cv2.IMREAD_UNCHANGED)
reconstructed_image = cv2.imread('Dicom File Embedding/Cover_Image_With_MSB_LSB.png', cv2.IMREAD_UNCHANGED)

if original_image.shape != reconstructed_image.shape:
    original_image = cv2.resize(original_image, reconstructed_image.shape, interpolation=cv2.INTER_AREA)

psnr_value = calculate_psnr(original_image, reconstructed_image)
ssim_value = calculate_ssim_cover(original_image, reconstructed_image)

print(f"PSNR of cover image: {psnr_value}")
print(f"SSIM of cover image: {ssim_value}")