import os
import cv2
import numpy as np
import pickle

DATA_DIR     = "dataset"        
TRAINER_FILE = "trainer.yml"    
LABELS_FILE  = "labels.pickle"  

def register_user(name: str, samples: int = 20):
    """Capture `samples` face images for `name`, then retrain the model."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cam          = cv2.VideoCapture(0)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    count = 0
    print(f"[+] Registering '{name}'.  Press 'q' to quit early.")
    while count < samples:
        ret, frame = cam.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{DATA_DIR}/{name}_{count}.jpg", face_img)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            break  

        cv2.putText(frame, f"Samples: {count}/{samples}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.imshow("Register", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    train_model()
    print(f"[+] Registration complete for '{name}'.")

def train_model():
    """Scan the dataset folder, assign numeric labels, train LBPH, and save."""
    recognizer   = cv2.face.LBPHFaceRecognizer_create()
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    current_id = 0
    label_ids  = {}
    x_train    = []
    y_labels   = []

    for fn in os.listdir(DATA_DIR):
        if not fn.lower().endswith(".jpg"):
            continue
        name = fn.split("_")[0]
        if name not in label_ids:
            label_ids[name] = current_id
            current_id += 1

        img_path = os.path.join(DATA_DIR, fn)
        img       = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces     = face_cascade.detectMultiScale(img)

        for (x,y,w,h) in faces:
            roi = img[y:y+h, x:x+w]
            x_train.append(roi)
            y_labels.append(label_ids[name])

    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_ids, f)
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(TRAINER_FILE)
    print("[*] Model trained and saved.")


def recognize_user(draw: bool = True, threshold: float = 50.0) -> str:
    """
    Open webcam, detect face, predict label, and return name.
    If draw=True, it will show a window with bounding boxes + name.
    """
    # load model + labels
    recognizer   = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    with open(LABELS_FILE, "rb") as f:
        label_ids = pickle.load(f)
    id_labels = {v:k for k,v in label_ids.items()}

    cam          = cv2.VideoCapture(0)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    name_found = "Unknown"
    print("[*] Starting recognition. Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            label_id, conf = recognizer.predict(roi)

            if conf < threshold:
                name_found = id_labels.get(label_id, "Unknown")
                color = (0,255,0)
            else:
                name_found = "Unknown"
                color = (0,0,255)

            if draw:
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, name_found, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if draw:
            cv2.imshow("Recognize", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or name_found != "Unknown":
            break

    cam.release()
    cv2.destroyAllWindows()
    return name_found

if __name__ == "__main__":
    mode = input("Enter mode (register/login): ").strip().lower()
    if mode == "register":
        user = input("Enter name to register: ").strip()
        register_user(user)
    elif mode == "login":
        user = recognize_user(draw=True)
        print(f"Detected User: {user}")
    else:
        print("Unknown mode. Use 'register' or 'login'.")
