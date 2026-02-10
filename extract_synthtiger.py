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
    skipped_count = 0
    
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
            for _ in unzipped_chunks:  # Must consume chunks even if skipping
                pass
            continue
        
        # Skip if already extracted (resume after restart)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            for _ in unzipped_chunks:  # Must consume chunks even if skipping
                pass
            skipped_count += 1
            if skipped_count % 10000 == 0:
                print(f"Skipped {skipped_count} already-extracted files...")
            continue
            
        # Atomic write: write to temp file, then rename
        # Prevents corrupt files if the computer crashes mid-write
        tmp_path = output_path + '.tmp'
        with open(tmp_path, 'wb') as f:
            for chunk in unzipped_chunks:
                f.write(chunk)
        os.replace(tmp_path, output_path)  # Atomic on Windows (NTFS)
        
        extracted_count += 1
        if extracted_count % 10000 == 0:
            print(f"Extracted {extracted_count} files (skipped {skipped_count})...")
    
    # Clean up any orphaned .tmp files from previous crashed runs
    for dirpath, _, filenames in os.walk(OUTPUT_DIR):
        for fname in filenames:
            if fname.endswith('.tmp'):
                os.remove(os.path.join(dirpath, fname))
    
    print(f"\nDone! Extracted {extracted_count} new files, skipped {skipped_count} existing.")
    print(f"Total in {OUTPUT_DIR}")

if __name__ == "__main__":
    print(f"Extracting {ZIP_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("This may take a while for 38GB...\n")
    extract_large_zip()
