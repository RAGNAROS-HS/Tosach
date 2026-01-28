import os
from collections import Counter
import matplotlib.pyplot as plt

data_dir = r'C:\Users\Hugo\Downloads\ocr\kaggle_data\training_data'
class_counts = Counter()
for cls in os.listdir(data_dir):
    cls_path = os.path.join(data_dir, cls)
    if os.path.isdir(cls_path):
        count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_counts[cls] = count

print("Per-class counts:", dict(class_counts))
print(f"Total: {sum(class_counts.values())} images")


plt.figure(figsize=(10, 8))
plt.barh(list(class_counts.keys()), list(class_counts.values()))
plt.xlabel('Images per Class')
plt.title('Class Distribution (Total: 22,364)')
plt.tight_layout()
plt.show()  # Or plt.savefig('distribution.png')
