import os
import cv2
import easyocr
import re
import pandas as pd
from difflib import SequenceMatcher

# =============================
# 1️⃣ Configuration
# =============================
DATASET_PATH = r"C:\LEVEL-3 SEM-1\LEVEL-3 SEM-1\Numerical Method Sessional-312\New folder\train\train"
OUTPUT_CSV = r"C:\LEVEL-3 SEM-1\LEVEL-3 SEM-1\Numerical Method Sessional-312\New folder\bangla_text_results.csv"
LABELS_CSV = r"C:\LEVEL-3 SEM-1\LEVEL-3 SEM-1\Numerical Method Sessional-312\New folder\labels.csv"   # <-- add your ground truth file path here
REPORT_CSV = r"C:\LEVEL-3 SEM-1\LEVEL-3 SEM-1\Numerical Method Sessional-312\New folder\ocr_accuracy_report.csv"
IMAGE_EXT = ('.png', '.jpg', '.jpeg')

# =============================
# 2️⃣ Initialize OCR Reader
# =============================
reader = easyocr.Reader(['bn'], gpu=False)

# =============================
# 3️⃣ Regex for Bangla text and digits
# =============================
bangla_text_pattern = re.compile(r'[\u0980-\u09FF]+')
bangla_digit_pattern = re.compile(r'[\u09E6-\u09EF]+')

# =============================
# 4️⃣ OCR Processing
# =============================
results = []

for file_name in os.listdir(DATASET_PATH):
    if not file_name.lower().endswith(IMAGE_EXT):
        continue

    image_path = os.path.join(DATASET_PATH, file_name)
    img = cv2.imread(image_path)
    if img is None:
        continue

    # --- Preprocessing ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # --- OCR ---
    ocr_result = reader.readtext(gray, detail=0, paragraph=True)
    text_all = ' '.join(ocr_result)

    # --- Extract clean Bangla text ---
    bangla_words = bangla_text_pattern.findall(text_all)
    clean_text = ' '.join(bangla_words)

    # --- Extract Bangla digits ---
    digits = ''.join(bangla_digit_pattern.findall(text_all))

    print(f"\n📄 {file_name}")
    print("📝 Bangla Text:", clean_text)
    print("🔢 Bangla Digits:", digits)
    print("-" * 60)

    results.append({
        "filename": file_name,
        "text_bangla": clean_text,
        "digits": digits
    })

# =============================
# 5️⃣ Save OCR Results
# =============================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n✅ All OCR results saved to: {OUTPUT_CSV}")

# =============================
# 6️⃣ Accuracy Calculation
# =============================
if os.path.exists(LABELS_CSV):
    print("\n📊 Calculating OCR Accuracy...")

    ocr_df = pd.read_csv(OUTPUT_CSV)
    truth_df = pd.read_csv(LABELS_CSV)

    # Merge OCR results with ground truth
    merged = pd.merge(ocr_df, truth_df, on="filename", how="inner")

    # Text similarity function
    def text_accuracy(pred, truth):
        pred = str(pred).strip()
        truth = str(truth).strip()
        return SequenceMatcher(None, pred, truth).ratio()

    # Apply accuracy function
    merged["accuracy"] = merged.apply(lambda x: text_accuracy(x["text_bangla"], x["ground_truth"]), axis=1)

    # Overall accuracy
    overall_acc = merged["accuracy"].mean() * 100

    # Save accuracy report
    merged.to_csv(REPORT_CSV, index=False, encoding='utf-8-sig')

    print(merged[["filename", "text_bangla", "ground_truth", "accuracy"]])
    print(f"\n✅ Overall OCR Accuracy: {overall_acc:.2f}%")
    print(f"📁 Detailed report saved to: {REPORT_CSV}")

else:
    print("\n⚠️ labels.csv file not found! Please create it to calculate accuracy.")

