import cv2
import os
import dlib
import tkinter as tk
from tkinter import Label, Entry, Button, StringVar, messagebox
from PIL import Image, ImageTk

# Function to get the person's name using a custom tkinter dialog
def get_person_name():
    def submit(event=None):
        nonlocal person_name
        person_name = name_var.get().strip().lower() # Convert entered name to lowercase

        # Check if a folder for this name already exists
        existing_folders = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
        if any(person_name in folder for folder in existing_folders):
            messagebox.showerror("Error", f"Name '{person_name}' already exists or is empty. Please choose a different name.")
            name_var.set("")  # Clear the entry field
        elif person_name:
            root.destroy()  # Close the dialog if the name is unique and non-empty

    def cancel():
        nonlocal person_name
        person_name = None  # Set name to None when cancelled
        root.destroy()  # Close the dialog

    person_name = None
    root = tk.Tk()
    root.title("Face Register")

    # Load an image using PIL
    image_path = "data/cover.jpg"  # Replace with your image path
    image = Image.open(image_path)
    image = image.resize((450, 200), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)

    # Create a Label to display the image
    img_label = Label(root, image=photo)
    img_label.pack()

    # Create a Label and Entry to get the person's name
    name_label = Label(root, text="Masukkan nama anda:")
    name_label.pack()
    name_var = StringVar()
    name_entry = Entry(root, textvariable=name_var)
    name_entry.pack()

    # Bind the Enter key to the submit function
    # Create a Button to submit the name
    root.bind("<Return>", submit)
    submit_button = Button(root, text="Submit", command=submit)
    submit_button.pack(side=tk.LEFT, padx=5, pady=5)

    # Create a Button to cancel
    cancel_button = Button(root, text="Cancel", command=cancel)
    cancel_button.pack(side=tk.RIGHT, padx=5, pady=5)

    # Center the window
    window_width = 350
    window_height = 280
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    root.mainloop()
    return person_name

# Set the directory to save images
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Get the person's name
person_name = get_person_name()
if not person_name:
    print("Operation cancelled or name already exists. Exiting program.")
    exit()

# Continue with creating the folder and capturing images only if name is provided
folder_count = sum(os.path.isdir(os.path.join(dataset_dir, item)) for item in os.listdir(dataset_dir))
folder_count_plus_one = folder_count + 1
person_dir = os.path.join(dataset_dir, f"person_{folder_count_plus_one}_{person_name}")
os.makedirs(person_dir, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(1)
imgBackground = cv2.imread("C:\\Users\\Verzen\\Documents\\Magang\\CobaKesini\\face-dev-nonalgorithm\\bg\\GB.png")

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()
count = 0

# Define the coordinates for the central rectangle
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
rect_x = int(frame_width * 0.35)
rect_y = int(frame_height * 0.25)
rect_w = int(frame_width * 0.3)
rect_h = int(frame_height * 0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector(frame)

    face_detected_in_rectangle = False
    for rect in faces:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

        # Check if the detected face is within the central rectangle
        if (x > rect_x and y > rect_y and
            x + w < rect_x + rect_w and
            y + h < rect_y + rect_h):

            face_detected_in_rectangle = True
            margin = int(0.4 * h)
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, frame_width)
            y2 = min(y + h + margin, frame_height)
            head = frame[y1:y2, x1:x2]

            if cv2.waitKey(1) & 0xFF == ord('h') or cv2.waitKey(1) & 0xFF == ord('H'):
                count += 1
                cv2.imwrite(os.path.join(person_dir, f"img_{person_name}_{count}.jpg"), head)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Gambar Wajah Disimpan: {count}/150", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if face_detected_in_rectangle:
        cv2.putText(frame, "Wajah terdeteksi", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif len(faces) > 1:
            print("Multiple faces detected. Please ensure only one face is visible.")
            cv2.putText(frame, "Banyak Wajah Terdeteksi!, tidak dapat menyimpan data!.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        print("Wajah tidak terdeteksi")
        cv2.putText(frame, "Wajah tidak terdeteksi, Tepatkan wajah di dalam kotak.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw the rectangle on the frame
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)

    # Display the frame
    imgBackground[140:140 + 480, 40:40 + 640] = frame
    cv2.imshow('Register Faces', imgBackground)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 150 or cv2.waitKey(1) & 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Captured {count} images for {person_name}.")
