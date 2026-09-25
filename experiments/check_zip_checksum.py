import os, hashlib, zipfile

zip_path = "Antigravity_ML_submission_optimized.zip"
if not os.path.exists(zip_path):
    print(f"Error: {zip_path} not found!")
    exit(1)

# SHA256
sha256 = hashlib.sha256()
with open(zip_path, "rb") as f:
    while chunk := f.read(1024 * 1024):
        sha256.update(chunk)
hex_digest = sha256.hexdigest().upper()
print(f"ZIP Path: {zip_path}")
print(f"ZIP Size: {os.path.getsize(zip_path):,d} bytes ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")
print(f"SHA256:   {hex_digest}")

print("\n--- ZIP Contents ---")
with zipfile.ZipFile(zip_path, "r") as z:
    for info in z.infolist():
        print(f"  {info.filename:<60} {info.file_size:>10,d} bytes")
