import os, sys
import requests 
import cv2 
import numpy
import imutils 

# This is used to run the pre-existing Haar Cascade to gather training data from a camera.
# Instead of changing the variables via code, system arguments are used so that
# those values could be changed when executing this script from the terminal.
def main():
    if len(sys.argv) < 5:
        print("USAGE: python batch_capture.py [Label] [Count] [Directory] [IP Address of Camera Device]")
        print("SAMPLE: python scripts/batch_capture.py ralph 1000 training http://192.168.100.156:8080/shot.jpg")
        return 1
    
    image_label = sys.argv[1]
    save_directory = sys.argv[3]
    camera_address = sys.argv[4]
    
    count = 0
    image_limit = int(sys.argv[2])

    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    while True:
        response = requests.get(camera_address) 
        array = numpy.array(bytearray(response.content), dtype=numpy.uint8) 
        frame = imutils.resize(cv2.imdecode(array, -1) , width=1000, height=1800) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
 
        for (x, y, w, h) in faces:
            if w > 250 and h > 250:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{str(count)}/{str(image_limit)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(save_directory, f'{image_label}_{count}.jpg'), gray[y:y + h, x:x + w])
                count += 1
 
        cv2.imshow('Capture Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
        if count >= image_limit:
            print("Image limit reached!")
            break
 
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()