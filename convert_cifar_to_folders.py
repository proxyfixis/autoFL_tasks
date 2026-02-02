import pickle
import os
import random
from PIL import Image

CIFAR_DIR = "cifar-10-batches-py"
OUTPUT_DIR = "data"

NUM_CLIENTS = 10
CLASSES_PER_CLIENT = 2
MIN_SAMPLES = 50
MAX_SAMPLES = 300


def unpickle(file):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


# Load label names
meta = unpickle(os.path.join(CIFAR_DIR, "batches.meta"))
label_names = [name.decode() for name in meta[b"label_names"]]


# Create client directories
for cid in range(NUM_CLIENTS):
    for label in label_names:
        os.makedirs(f"{OUTPUT_DIR}/client_{cid}/{label}", exist_ok=True)


# STEP 1: load all images grouped by label
label_to_images = {label: [] for label in label_names}

for batch_id in range(1, 6):
    batch = unpickle(os.path.join(CIFAR_DIR, f"data_batch_{batch_id}"))
    images = batch[b"data"]
    labels = batch[b"labels"]

    for i in range(len(images)):
        img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        label = label_names[labels[i]]
        label_to_images[label].append(img)

# Shuffle each label pool
for label in label_to_images:
    random.shuffle(label_to_images[label])


# STEP 2: assign label subsets to clients
all_classes = label_names.copy()
random.shuffle(all_classes)

client_classes = {}
for cid in range(NUM_CLIENTS):
    client_classes[cid] = all_classes[
        cid * CLASSES_PER_CLIENT : (cid + 1) * CLASSES_PER_CLIENT
    ]


# STEP 3: allocate non-IID data to clients
for cid in range(NUM_CLIENTS):
    classes = client_classes[cid]
    total_images = random.randint(MIN_SAMPLES, MAX_SAMPLES)
    per_class = total_images // len(classes)

    img_idx = 0
    for cls in classes:
        imgs = label_to_images[cls][:per_class]
        label_to_images[cls] = label_to_images[cls][per_class:]

        for img in imgs:
            path = f"{OUTPUT_DIR}/client_{cid}/{cls}/{img_idx}.png"
            Image.fromarray(img).save(path)
            img_idx += 1
