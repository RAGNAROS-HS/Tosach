r"""
Memory-efficient extraction of large ZIP files using stream-unzip.
Extracts synthtiger_v1.0.zip to E:\synthtiger_data without loading entire archive into memory.
"""
import os
from stream_unzip import stream_unzip

ZIP_PATH = r"E:\synthtiger_v1.0.zip"
OUTPUT_DIR = r"E:\synthtiger_data"

def extract_large_zip():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    extracted_count = 0
    
    def zip_chunks():
        with open(ZIP_PATH, 'rb') as f:
            while chunk := f.read(65536):  # 64KB chunks
                yield chunk
    
    for file_name, file_size, unzipped_chunks in stream_unzip(zip_chunks()):
        file_name = file_name.decode('utf-8')
        output_path = os.path.join(OUTPUT_DIR, file_name)
        
        # Create parent directories
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Skip if it's a directory entry
        if file_name.endswith('/'):
            continue
            
        # Write file in chunks
        with open(output_path, 'wb') as f:
            for chunk in unzipped_chunks:
                f.write(chunk)
        
        extracted_count += 1
        if extracted_count % 10000 == 0:
            print(f"Extracted {extracted_count} files...")
    
    print(f"\nDone! Extracted {extracted_count} files to {OUTPUT_DIR}")

if __name__ == "__main__":
    print(f"Extracting {ZIP_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("This may take a while for 38GB...\n")
    extract_large_zip()
