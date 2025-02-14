import os, sys
import cv2
import numpy

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_capture.py [Directory] [Save Path]")
        return 1
    
    train_directory = sys.argv[2]
    model_save_path = sys.argv[3]

    faces = []
    labels = []
     
    for file_name in os.listdir(train_directory):
        if file_name.endswith('.jpg'):
            name = file_name.split('_')[0]
            image = cv2.imread(os.path.join(train_directory, file_name))
            
            faces.append(image)
            labels.append(name)
 
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, numpy.array(labels))
 
    recognizer.save(model_save_path)
    return recognizer
 
if __name__ == "__main__":
    main()