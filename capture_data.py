import cv2
import os
import time

emotion = input("Enter emotion name: ")

# Create folders
train_path = f"dataset/train/{emotion}"
val_path = f"dataset/validation/{emotion}"
test_path = f"dataset/test/{emotion}"

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

count = 0

print("Press ESC to stop.")

while True:

    ret, frame = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(frame,
                      (x, y),
                      (x+w, y+h),
                      (255, 0, 0),
                      2)

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (48, 48))

        count += 1

        # Save images according to split
        if count <= 350:
            filename = os.path.join(train_path, f"{count}.jpg")

        elif count <= 425:
            filename = os.path.join(val_path, f"{count}.jpg")

        elif count <= 500:
            filename = os.path.join(test_path, f"{count}.jpg")

        else:
            break

        cv2.imwrite(filename, face)

        # delay to avoid duplicate frames
        time.sleep(0.2)

    cv2.imshow("Capturing Faces", frame)

    key = cv2.waitKey(1)

    if key == 27 or count >= 500:
        break

cam.release()
cv2.destroyAllWindows()

print("Dataset collection completed.")