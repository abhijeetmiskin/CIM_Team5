# ---- traffic sign trainer (auto-split + clean) ----
import os
import sys
import random
import shutil
import warnings

# OPTIONAL: keep logs quieter
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # silence oneDNN info logs

# Reproducibility
random.seed(42)

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_NORMAL = os.path.join(SCRIPT_DIR, "normal_signs")
RAW_DAMAGED = os.path.join(SCRIPT_DIR, "damaged_signs")
SPLIT_ROOT = os.path.join(SCRIPT_DIR, "output_split")
TRAIN_DIR = os.path.join(SPLIT_ROOT, "train")
TEST_DIR = os.path.join(SPLIT_ROOT, "test")

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

# ---- Utilities ----
def is_image_file(path: str) -> bool:
    return path.lower().endswith(VALID_EXTS)

def safe_copy(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, dst_dir)

def create_split_from_raw(test_ratio=0.2):
    """Create output_split/train|test from normal_signs/ & damaged_signs/ (next to this script)."""
    if not (os.path.isdir(RAW_NORMAL) and os.path.isdir(RAW_DAMAGED)):
        print("✖ No split found and raw folders missing.\n"
              f"  Expected: '{RAW_NORMAL}' and '{RAW_DAMAGED}'")
        sys.exit(1)

    print("➡ Creating train/test split from raw folders...")
    # Clear old split if present
    if os.path.isdir(SPLIT_ROOT):
        shutil.rmtree(SPLIT_ROOT)
    os.makedirs(SPLIT_ROOT, exist_ok=True)

    for cls_name, src_dir in [("normal_signs", RAW_NORMAL), ("damaged_signs", RAW_DAMAGED)]:
        files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if is_image_file(f)]
        files = [f for f in files if os.path.isfile(f)]
        random.shuffle(files)
        if not files:
            print(f"✖ No images found in {src_dir}. Add real images and rerun.")
            sys.exit(1)

        cut = int(len(files) * (1 - test_ratio))
        train_files, test_files = files[:cut], files[cut:]

        for f in train_files:
            safe_copy(f, os.path.join(TRAIN_DIR, cls_name))
        for f in test_files:
            safe_copy(f, os.path.join(TEST_DIR, cls_name))

    print(f"✔ Split created at: {SPLIT_ROOT}")

# Remove corrupted / non-image files
def clean_folder(folder):
    from PIL import Image
    removed = 0
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            if not is_image_file(path):
                try:
                    os.remove(path); removed += 1
                except Exception:
                    pass
                continue
            try:
                with Image.open(path) as im:
                    im.verify()  # quick integrity check
            except Exception:
                try:
                    os.remove(path); removed += 1
                    print(f"❌ Removed corrupted: {path}")
                except Exception:
                    pass
    return removed

def count_images(folder):
    total = 0
    for root, _, files in os.walk(folder):
        total += sum(1 for f in files if is_image_file(f))
    return total

# ---- Ensure dataset is ready ----
if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(TEST_DIR)):
    create_split_from_raw(test_ratio=0.2)

print("🧹 Cleaning datasets (removing corrupted or non-images)...")
removed_train = clean_folder(TRAIN_DIR)
removed_test = clean_folder(TEST_DIR)
if removed_train or removed_test:
    print(f"Removed {removed_train} bad files from train, {removed_test} from test.")

n_train = count_images(TRAIN_DIR)
n_test = count_images(TEST_DIR)
if n_train == 0 or n_test == 0:
    print(f"✖ After cleaning, images remaining — train: {n_train}, test: {n_test}. "
          "Please place real JPG/PNG images in 'normal_signs' and 'damaged_signs' and rerun.")
    sys.exit(1)

print(f"✔ Images ready — train: {n_train}, test: {n_test}")

# ---- Model / Training ----
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

tf.random.set_seed(42)

NUM_CLASSES = 2
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 6

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Base model
base = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base.input, outputs=out)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print("🚀 Training...")
history = model.fit(train_gen, epochs=EPOCHS, validation_data=test_gen)

# Save model and label map
model_path = os.path.join(SCRIPT_DIR, "traffic_sign_model.h5")
model.save(model_path)
print(f"💾 Model saved to: {model_path}")

# Save class indices (label mapping)
labels_path = os.path.join(SCRIPT_DIR, "class_indices.txt")
with open(labels_path, "w") as f:
    for k, v in train_gen.class_indices.items():
        f.write(f"{v}:{k}\n")
print(f"💾 Label map saved to: {labels_path} (format: index:label)")

# Plot history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.tight_layout()
plt.show()
# ---- end ----
