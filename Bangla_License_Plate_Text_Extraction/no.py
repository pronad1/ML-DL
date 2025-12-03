import os
import cv2
import easyocr
import re
import pandas as pd

# Folder path
folder_path = r"C:\LEVEL-3 SEM-1\LEVEL-3 SEM-1\Numerical Method Sessional-312\New folder\train\train"
image_extensions = ('.png', '.jpg')

# Initialize EasyOCR
reader = easyocr.Reader(['bn'], gpu=False)

# Regex patterns
bangla_pattern = re.compile(r'[\u0980-\u09FF]+')   # Bangla letters only
bangla_digits_pattern = re.compile(r'[\u09E6-\u09EF]+')  # Bangla digits only

results = []

for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(image_extensions):
        img_path = os.path.join(folder_path, file_name)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # OCR
        ocr_results = reader.readtext(gray)
        text_all = ' '.join([res[1] for res in ocr_results]).strip()
        
        # Clean Bangla text: only Bangla letters + space
        bangla_matches = bangla_pattern.findall(text_all)
        clean_bangla = ' '.join(bangla_matches)
        
        # Extract only Bangla digits
        digits_matches = bangla_digits_pattern.findall(text_all)
        digits = ''.join(digits_matches)
        
        # Print clean output
        print("Filename:", file_name)
        print("Clean Bangla Text:", clean_bangla)
        print("Digits (Bangla only):", digits)
        print("-"*50)
        
        # Append to results
        results.append({
            "filename": file_name,
            "text_bangla": clean_bangla,
            "digits": digits
        })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("bangla_text_digits_clean_only.csv", index=False)
print("All cleaned data saved to bangla_text_digits_clean_only.csv")
