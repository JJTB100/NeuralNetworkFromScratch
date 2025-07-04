import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import pandas as pd
# Load trained parameters
params = np.load("emnist_digits_all_weights.npz")
W1, b1 = params["W1"], params["b1"]
W2, b2 = params["W2"], params["b2"]
W3, b3 = params["W3"], params["b3"]



def leaky_ReLU(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)

def softmax(x):
    f = np.exp(x - np.max(x, axis=0, keepdims=True))
    return f / f.sum(axis=0, keepdims=True)

def predict(X):
    Z1 = W1.dot(X) + b1
    A1 = leaky_ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = leaky_ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    top3 = np.argsort(A3.flatten())[::-1][:3]
    top3_conf = A3.flatten()[top3]
    return top3, top3_conf

# Drawing app
class DrawApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a Digit")
        self.canvas_size = 280
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.predict_digit)

        self.button_clear = tk.Button(self, text="Clear", command=self.clear)
        self.button_clear.pack()

        self.label_result = tk.Label(self, text="Draw a digit", font=("Helvetica", 16))
        self.label_result.pack()
        
        self.label_result1 = tk.Label(self, text="", font=("Helvetica", 16))
        self.label_result1.pack()
        self.label_result2 = tk.Label(self, text="", font=("Helvetica", 16))
        self.label_result2.pack()

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=0)
        self.label_result.config(text="Draw a digit")
        self.label_result1.config(text="")
        self.label_result2.config(text="")

    def predict_digit(self, event=None):

        img = self.center_and_scale(self.image)
        img_data = np.asarray(img).astype(np.float32) / 255.0
        img_flat = img_data.reshape(784, 1)
        
        #self.display_image(img_data)  # Display the drawn image in the canvas
        #self.test_on_sample()  # Test on a known sample before prediction

        pred_3, conf_3 = predict(img_flat)
        print(conf_3)
        self.label_result.config(text=f"Prediction: {(pred_3[0])}  (Confidence: {conf_3[0]*100:.2f}%)")
        self.label_result1.config(text=f"Prediction: {(pred_3[1])}  (Confidence: {conf_3[1]*100:.2f}%)")
        self.label_result2.config(text=f"Prediction: {(pred_3[2])}  (Confidence: {conf_3[2]*100:.2f}%)")


    def center_and_scale(self, img):
        # Crop to content
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        
        # Resize while maintaining aspect ratio
        max_side = max(img.size)
        new_img = Image.new("L", (max_side, max_side), 0)  # Black background
        # Add padding: 20% of the max_side as margin
        margin = int(0.2 * max_side)
        padded_side = max_side + 2 * margin
        padded_img = Image.new("L", (padded_side, padded_side), 0)
        # Center the cropped image in the padded image
        padded_img.paste(img, (margin + (max_side - img.width) // 2, margin + (max_side - img.height) // 2))
        new_img = padded_img
        
        # Finally resize to 28x28
        new_img = new_img.resize((28, 28), Image.Resampling.LANCZOS)
        
        return new_img
    
    def display_image(self, img_data):
        # Display the preprocessed image in a new window for verification
        img_show = Image.fromarray((img_data * 255).astype(np.uint8))
        img_show = img_show.resize((140, 140), Image.NEAREST)
        win = tk.Toplevel(self)
        win.title("Preprocessed Image")
        tk_img = ImageTk.PhotoImage(img_show)
        label = tk.Label(win, image=tk_img)
        label.image = tk_img
        label.pack()
        
    def test_on_sample(self):
        data = pd.read_csv('emnist-digits-test.csv')
        data = np.array(data)
        m, n = data.shape
        np.random.shuffle(data) # shuffle before splitting into dev and training sets

        data_train = data[:m].T
        Y_train = data_train[0]
        X_train = data_train[1:n]
        X_train = X_train / 255.
        _,m_train = X_train.shape

        sample_img = X_train[:, 0].reshape(784, 1)  # first training image
        pred, conf = predict(sample_img)
        print(f"Known sample prediction: {pred[0]} with confidence {conf[0]*100:.2f}%. Was: {Y_train[0]}")
        
        # Display the actual test image in a popup window
        img_array = (sample_img.reshape(28, 28) * 255).astype(np.uint8)
        img_show = Image.fromarray(img_array)
        img_show = img_show.resize((140, 140), Image.NEAREST)
        win = tk.Toplevel(self)
        win.title(f"Test Sample (Label: {Y_train[0]})")
        tk_img = ImageTk.PhotoImage(img_show)
        label = tk.Label(win, image=tk_img)
        label.image = tk_img
        label.pack()
        
# Run app
app = DrawApp()
app.mainloop()
