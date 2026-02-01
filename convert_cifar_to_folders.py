import pickle
import os
from PIL import Image

CIFAR_DIR = "cifar-10-batches-py"
OUTPUT_DIR = "data"
NUM_CLIENTS = 10
IMAGES_PER_CLIENT = 100
TOTAL_IMAGES = NUM_CLIENTS * IMAGES_PER_CLIENT
CENTRAL_IMAGES = 500  # size of server-side test set
TOTAL_REQUIRED = TOTAL_IMAGES + CENTRAL_IMAGES


def unpickle(file):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


# Load label names
meta = unpickle(os.path.join(CIFAR_DIR, "batches.meta"))
label_names = [name.decode() for name in meta[b"label_names"]]

# Create output directories
for cid in range(NUM_CLIENTS):
    for label in label_names:
        os.makedirs(f"{OUTPUT_DIR}/client_{cid}/{label}", exist_ok=True)

for label in label_names:
    os.makedirs(f"{OUTPUT_DIR}/central_test/{label}", exist_ok=True)

image_count = 0  # GLOBAL counter

# Read CIFAR batches
for batch_id in range(1, 6):
    if image_count >= TOTAL_REQUIRED:
        break

    batch = unpickle(os.path.join(CIFAR_DIR, f"data_batch_{batch_id}"))
    images = batch[b"data"]
    labels = batch[b"labels"]

for i in range(len(images)):
    if image_count >= TOTAL_REQUIRED:
        break

        
       

    img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
    label = label_names[labels[i]]
    
    if image_count < TOTAL_IMAGES:
    # Client data
            client_id = image_count // IMAGES_PER_CLIENT
            target_root = f"{OUTPUT_DIR}/client_{client_id}"
            filename = f"{image_count % IMAGES_PER_CLIENT}.png"
    else:
    # Central test data
            target_root = f"{OUTPUT_DIR}/central_test"
            filename = f"{image_count - TOTAL_IMAGES}.png"

    img_path = f"{target_root}/{label}/{filename}"
    img_pil = Image.fromarray(img)
        
    img_pil.save(img_path)

    image_count += 1
