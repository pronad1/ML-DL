import os
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

# ------------------ CONFIG ------------------
IMAGE_FOLDER = "train/train"  # folder with images
LABELS_FILE = "labels.csv"
OUTPUT_DIR = "trocr_bangla_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ LOAD DATA ------------------
labels_df = pd.read_csv(LABELS_FILE)
print("CSV Columns:", labels_df.columns.tolist())

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(labels_df)

# ------------------ PREPROCESSING ------------------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1").to(DEVICE)

# Set decoder start token if not set
if model.config.decoder_start_token_id is None:
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id

def preprocess(example):
    # Load image
    image_path = os.path.join(IMAGE_FOLDER, example["filename"])
    image = Image.open(image_path).convert("RGB")
    # Encode text
    labels = processor.tokenizer(example["text_bangla"], padding="max_length", truncation=True, max_length=128).input_ids
    return {"pixel_values": processor(images=image, return_tensors="pt").pixel_values.squeeze(), "labels": labels}

dataset = dataset.map(preprocess)

# ------------------ TRAINING ARGUMENTS ------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=10,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
)

# ------------------ TRAINER ------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor,
)

# Train the model (fine-tuning)
trainer.train()

# ------------------ PREDICTION ------------------
predictions = []
for example in dataset:
    image_path = os.path.join(IMAGE_FOLDER, example["filename"])
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    predictions.append({"filename": example["filename"], "ground_truth": example["text_bangla"], "prediction": text})
    print(f"📄 {example['filename']}\n📝 Ground Truth: {example['text_bangla']}\n🔤 Prediction : {text}\n{'-'*50}")

# ------------------ SAVE PREDICTIONS ------------------
pred_df = pd.DataFrame(predictions)
pred_df.to_csv("ocr_predictions.csv", index=False)
print("✅ Predictions saved to ocr_predictions.csv")
