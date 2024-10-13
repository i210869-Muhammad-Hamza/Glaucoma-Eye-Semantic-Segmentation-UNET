import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
CDR=0
class_colors = {
    0: (0, 0, 0),   # Black
    1: (255, 0, 0), # Red (for segmentation 1)
    2: (0, 255, 0)  # Green (for segmentation 2)
}

# Convert mask array to PIL image
def create_image_from_mask(mask_array):
    height, width = mask_array.shape
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = class_colors[mask_array[y, x]]
    return img

# Function to exit the application
def exit_app():
    app.destroy()

# Function to open a file dialog for image selection
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        update_image(file_path)

def apply_morphological_close(image, kernel_size=(5,5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


# Function to update the image on the GUI and glaucoma image
# Load the input image and resize
def update_image(image_path):        
    input_image = Image.open(image_path)
    input_image = input_image.resize((256, 256))
    input_array = np.array(input_image) / 255.0
    input_array = np.expand_dims(input_array, axis=0)

    # Load the model and make predictions
    unet_model = load_model('unet_model.h5')
    #predictions = model.predict(input_array)
    #threshold = 0.15
    #redicted_mask = (predictions > threshold).astype(np.uint8)
    #mask_img = predicted_mask
    #if mask_img.ndim > 2:
     #   mask_img = mask_img[0, :, :, 0]

    #yahan se meri
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    imad=np.array(image)
    predict=unet_model.predict(imad)
    predicted_masks = (np.argmax(predict, axis=-1)).astype(np.uint8)
    predicted_masks = apply_morphological_close(predicted_masks)


    # Create ImageTk.PhotoImage from the predicted mask
    mask_image = create_image_from_mask(predicted_masks[0])
    x=predicted_masks.flatten()
    unique_values, counts = np.unique(x, return_counts=True)
    ones_count=0
    two_count=0
    for value, count in zip(unique_values, counts):
        if(value==1):
            ones_count=count
        elif(value==2):
            two_count=count 
    global CDR
    CDR=(two_count/(ones_count+two_count))*100
    rounded=round(CDR,2)
    heading_label3 = tk.Label(heading_frame3, text="CDR Ratio:"+str(rounded), font=("Helvetica", 20, "bold"), bg="black", fg="white")
    heading_label3.pack()         
    #mask_image = Image.fromarray(predicted_masks[0] * 255, mode='L')
    resized_image = mask_image.resize((256, 256))
    mask_image_tk = ImageTk.PhotoImage(resized_image)

    # Create ImageTk.PhotoImage from the input image
    input_image_tk = ImageTk.PhotoImage(input_image)

    # Clear the canvas
    canvas.delete("all")

    # Display the images side by side on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=bg_image)
    canvas.create_image(app.winfo_screenwidth() // 4 +110 , (app.winfo_screenheight() // 2) - 10, anchor=tk.CENTER, image=input_image_tk)
    canvas.create_image(app.winfo_screenwidth() // 4 + 620, (app.winfo_screenheight() // 2) - 10, anchor=tk.CENTER, image=mask_image_tk)

    canvas.input_image_tk = input_image_tk
    canvas.mask_image_tk = mask_image_tk

# GUI setup
app = tk.Tk()
app.title("Glaucoma Eye Segmentation")

# Load background image
bg_image_path = "brain_bg.jpg"  
bg_image = Image.open(bg_image_path)
bg_image = ImageTk.PhotoImage(bg_image.resize((app.winfo_screenwidth(), app.winfo_screenheight())))


canvas_frame = tk.Frame(app)
canvas_frame.place(relwidth=1, relheight=1)

canvas = tk.Canvas(canvas_frame, width=app.winfo_screenwidth(), height=app.winfo_screenheight())
canvas.pack()

canvas.create_image(0, 0, anchor=tk.NW, image=bg_image)

heading_frame = tk.Frame(app, bg="black")
heading_frame.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
heading_frame1 = tk.Frame(app, bg="black")
heading_frame1.place(relx=0.35, rely=0.3, anchor=tk.CENTER)
heading_frame2 = tk.Frame(app, bg="black")
heading_frame2.place(relx=0.7, rely=0.3, anchor=tk.CENTER)
heading_frame3 = tk.Frame(app, bg="black")
heading_frame3.place(relx=0.5, rely=0.8, anchor=tk.CENTER)


# original image
heading_label1 = tk.Label(heading_frame1, text="Original Image", font=("Helvetica", 20, "bold"), bg="black", fg="white")
heading_label1.pack()
# segmented image
heading_label2 = tk.Label(heading_frame2, text="Segmented Image", font=("Helvetica", 20, "bold"), bg="black", fg="white")
heading_label2.pack()

# Heading
heading_label = tk.Label(heading_frame, text="Glaucoma Eye Segmentation", font=("Helvetica", 20, "bold"), bg="black", fg="white")
heading_label.pack()
# Upload image button
upload_button = tk.Button(heading_frame, text="Upload Image", command=upload_image, font=("Helvetica", 12), bg="#008CBA", fg="white")
upload_button.pack(pady=10)

button_frame = tk.Frame(app, bg="#333")
button_frame.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

# Exit button
exit_button = tk.Button(button_frame, text="Exit", command=exit_app, font=("Helvetica", 12), bg="white", fg="black")
exit_button.pack()


app.mainloop()
