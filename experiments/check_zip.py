import zipfile

with zipfile.ZipFile('Antigravity_ML_submission_optimized.zip', 'r') as z:
    for info in z.infolist():
        print(f"{info.filename}: {info.file_size:,d} bytes")
