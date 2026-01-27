import pickle
import os
from PIL import Image

CIFAR_DIR = "cifar-10-batches-py"
OUTPUT_DIR = "data"
NUM_CLIENTS = 5
IMAGES_PER_CLIENT = 100
TOTAL_IMAGES = NUM_CLIENTS * IMAGES_PER_CLIENT


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

image_count = 0  # GLOBAL counter

# Read CIFAR batches
for batch_id in range(1, 6):
    if image_count >= TOTAL_IMAGES:
        break

    batch = unpickle(os.path.join(CIFAR_DIR, f"data_batch_{batch_id}"))
    images = batch[b"data"]
    labels = batch[b"labels"]

    for i in range(len(images)):
        if image_count >= TOTAL_IMAGES:
            break

        # Decide client deterministically
        client_id = image_count // IMAGES_PER_CLIENT

        img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        label = label_names[labels[i]]

        img_pil = Image.fromarray(img)
        img_path = (
            f"{OUTPUT_DIR}/client_{client_id}/"
            f"{label}/{image_count % IMAGES_PER_CLIENT}.png"
        )
        img_pil.save(img_path)

        image_count += 1
