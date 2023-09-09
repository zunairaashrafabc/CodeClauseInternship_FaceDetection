import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")

        self.title_label = tk.Label(root, text="Face Detection App", font=("Helvetica", 16, "bold"))
        self.title_label.pack(pady=10)

        self.open_button = tk.Button(root, text="Upload Image", command=self.open_image, font=("Helvetica", 12))
        self.open_button.pack(pady=10)

        # Create a frame to hold the original image and detected face image
        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

        # Create labels to display images
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(side=tk.LEFT, padx=10)

        self.result_label = tk.Label(self.image_frame)
        self.result_label.pack(side=tk.RIGHT, padx=10)

        # Initialize PhotoImage objects
        self.photo = None
        self.result_photo = None

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = image.resize((400, 400), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)

            # Display the original image
            self.image_label.config(image=self.photo)

            result_image = self.detect_faces(file_path)
            self.result_photo = ImageTk.PhotoImage(result_image)

            # Display the detected face image (resized to match the original image size)
            self.result_label.config(image=self.result_photo)

    def detect_faces(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), thickness=10)  # Increase thickness for bolder rectangles

        result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_image)

        # Resize the result image to match the original image size
        result_image = result_image.resize((400, 400), Image.LANCZOS)
        return result_image

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
