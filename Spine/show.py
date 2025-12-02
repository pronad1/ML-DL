import pydicom
import matplotlib.pyplot as plt

# Load the file
filename = "0437508b2e4d9419247f1ee4075917f5.dicom"
ds = pydicom.dcmread(filename)

# 1. Print Metadata (What kind of scan is this?)
print(f"Modality: {ds.get('Modality', 'Unknown')}")        # e.g., CT, MR, DX (X-Ray)
print(f"Body Part: {ds.get('BodyPartExamined', 'Unknown')}")
print(f"Patient ID: {ds.get('PatientID', 'Unknown')}")

# 2. View the Image
# DICOM pixel data is stored in 'pixel_array'
plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 
plt.title(f"Scan: {ds.get('Modality', 'Unknown')}")
plt.axis('off')
plt.show()