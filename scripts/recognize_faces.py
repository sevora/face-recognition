# Import essential libraries 
import sys
import requests 
import cv2 
import numpy as np 
import imutils 

def main():
    if len(sys.argv) < 4:
        print("USAGE: python recognize_faces.py [Model Directory] [Labels Dictionary] [IP Address of Camera Device]")
        print("SAMPLE: python scripts/recognize_faces.py model.yml labels.txt http://192.168.100.156:8080/shot.jpg")
        return 1
    
    model_path = sys.argv[1]
    labels_path = sys.argv[2]
    camera_address = sys.argv[3]

    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)

    label_dictionary = {}

    with open(labels_path, "r") as file:
        for line in file.readlines():
            index, name = line.strip().lower().split(",")
            label_dictionary[index] = name

    while True: 
        response = requests.get(camera_address) 
        array = np.array(bytearray(response.content), dtype=np.uint8) 
        frame = imutils.resize(cv2.imdecode(array, -1), width=1000, height=1800) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Recognize the face
            label, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])
            name = "Unknown Face"

            if confidence < 100:
                name = label_dictionary[str(label)]

            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Android_Camera", frame) 

        # Press Esc key to exit 
        if cv2.waitKey(1) == 27: 
            break

    cv2.destroyAllWindows() 
    
if __name__ == "__main__":
    main()