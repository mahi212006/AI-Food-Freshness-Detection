import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
from skimage.feature import hog

with open("model/freshness_model.pkl", "rb") as f:
    model = pickle.load(f)

def get_features(path):
    img = Image.open(path).convert("RGB").resize((128,128))
    a = np.array(img) / 255.0
    gray = np.mean(a, axis=2)

    h = hog(gray, orientations=9, pixels_per_cell=(16,16),
            cells_per_block=(2,2), block_norm="L2-Hys")

    c = []
    for i in range(3):
        c += [np.mean(a[:,:,i]), np.std(a[:,:,i]),
              np.percentile(a[:,:,i],25),
              np.percentile(a[:,:,i],75)]

    return np.concatenate([h,c]).reshape(1,-1)

def predict():
    path = filedialog.askopenfilename(
        filetypes=[("All Files","*.*")]
    )
    if path:
        x = get_features(path)
        p = int(model.predict(x)[0])
        prob = model.predict_proba(x)[0][p] * 100
        result.config(text=f"{'FRESH' if p==0 else 'ROTTEN'}\nConfidence: {prob:.2f}%")

root = tk.Tk()
root.title("AI Food Freshness Detection")
root.geometry("500x300")

tk.Label(root,text="AI Food Freshness Detection",
         font=("Arial",20,"bold")).pack(pady=35)

tk.Button(root,text="Select Food Image",
          command=predict,font=("Arial",14)).pack(pady=20)

result = tk.Label(root,text="Result: -",
                  font=("Arial",18,"bold"))
result.pack(pady=25)

root.mainloop()
