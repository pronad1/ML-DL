from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing
from PIL import Image

# -----------------------------
# 1. CONNECT YOUR IMAGE FOLDER
# -----------------------------
folder_path = r"C:\\LEVEL-3 SEM-1\\LEVEL-3 SEM-1\\Numerical Method Sessional-312\\New folder\\train\\train"
image_extensions = ('.png', '.jpg')

image_data = []
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(image_extensions):
        img_path = os.path.join(folder_path, file_name)
        try:
            img = Image.open(img_path)
            width, height = img.size
            mode = img.mode  # e.g., RGB, L, etc.
            image_data.append({
                "filename": file_name,
                "width": width,
                "height": height,
                "mode": mode
            })
        except:
            print(f"Could not read: {img_path}")

# Create DataFrame
df = pd.DataFrame(image_data)
df.dataframeName = "Image_Metadata"
print(df.head())

# -------------------------------------
# 2. EXISTING FUNCTIONS (FIXED FOR STRINGS & NUMERIC ONLY)
# -------------------------------------

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df_use = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    nRow, nCol = df_use.shape
    columnNames = list(df_use)

    nGraphRow = (nCol + nGraphPerRow - 1) // nGraphPerRow  # integer rows

    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df_use.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()

# Correlation matrix (numeric columns only)
def plotCorrelationMatrix(df, graphWidth):
    filename = getattr(df, "dataframeName", "DataFrame")
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        print(f'No correlation plots shown: not enough numeric columns ({numeric_df.shape[1]})')
        return

    corr = numeric_df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots (numeric columns only)
def plotScatterMatrix(df, plotSize, textSize):
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        print("Not enough numeric data for scatter plot.")
        return
    
    ax = pd.plotting.scatter_matrix(numeric_df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = numeric_df.corr().values
    for i, j in zip(*np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j],
                          (0.8, 0.2), xycoords='axes fraction',
                          ha='center', va='center', size=textSize)
    
    plt.suptitle('Scatter and Density Plot')
    plt.show()

# --------------------------------------------------
# 3. RUN THE FUNCTIONS ON IMAGE METADATA
# --------------------------------------------------
plotPerColumnDistribution(df, nGraphShown=10, nGraphPerRow=3)
plotCorrelationMatrix(df, graphWidth=8)
plotScatterMatrix(df, plotSize=8, textSize=10)
