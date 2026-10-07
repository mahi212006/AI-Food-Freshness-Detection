import os
import pickle
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATASET_DIR = "dataset"
IMG_SIZE = (128, 128)
CATEGORIES = ["fresh", "rotten"]


def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img) / 255.0
    gray = np.mean(arr, axis=2)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    color_features = []

    for channel in range(3):
        mean = np.mean(arr[:, :, channel])
        std = np.std(arr[:, :, channel])
        p25 = np.percentile(arr[:, :, channel], 25)
        p75 = np.percentile(arr[:, :, channel], 75)

        color_features.extend([mean, std, p25, p75])

    features = np.concatenate([
        hog_features,
        np.array(color_features)
    ])

    return features


def load_dataset():

    X = []
    y = []

    print("\nLoading dataset...\n")

    for label, category in enumerate(CATEGORIES):

        folder = os.path.join(DATASET_DIR, category)

        if not os.path.exists(folder):
            print("Folder not found:", folder)
            continue

        print("Loading:", category)

        for filename in os.listdir(folder):

            image_path = os.path.join(folder, filename)

            try:
                features = extract_features(image_path)

                X.append(features)
                y.append(label)

            except Exception:
                print("Skipped:", filename)

    return np.array(X), np.array(y)


print("\n=================================")
print(" AI FOOD FRESHNESS DETECTION")
print(" MACHINE LEARNING MODEL TRAINING")
print("=================================\n")


X, y = load_dataset()


if len(X) < 20:

    print("\nNot enough images!")
    print("Please add more images.")

    input("\nPress Enter to exit...")
    raise SystemExit


if len(set(y)) < 2:

    print("\nBoth classes are required!")

    print("Add images to:")
    print("dataset/fresh")
    print("dataset/rotten")

    input("\nPress Enter to exit...")
    raise SystemExit


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))


model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",        probability=True,
        random_state=42
    )
)


print("\nTraining model...")

model.fit(X_train, y_train)


prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)


print("\n=================================")
print("MODEL PERFORMANCE")
print("=================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        prediction,
        target_names=CATEGORIES
    )
)


os.makedirs("model", exist_ok=True)

model_path = "model/freshness_model.pkl"


with open(model_path, "wb") as file:
    pickle.dump(model, file)


print("\n=================================")
print("MODEL SAVED SUCCESSFULLY!")
print("=================================")

print(model_path)


input("\nPress Enter to exit...")
