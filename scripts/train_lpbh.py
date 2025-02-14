import os, sys
import cv2
import numpy

def main():
    if len(sys.argv) < 4:
        print("USAGE: python train_lpbh.py [Directory] [Label Dictionary] [Save Path]")
        print("SAMPLE: python train_lpbh.py training labels.txt model.yml")
        return 1
    
    train_directory = sys.argv[1]
    labels_path = sys.argv[2]
    model_save_path = sys.argv[3]

    faces = []
    labels = []
    label_dictionary = {}

    with open(labels_path, "r") as file:
        for line in file.readlines():
            index, name = line.strip().split(",")
            label_dictionary[name] = int(index)
     
    for file_name in os.listdir(train_directory):
        if file_name.endswith('.jpg'):
            name = file_name.split('_')[0]
            image = cv2.imread(os.path.join(train_directory, file_name))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces.append(gray)
            labels.append(label_dictionary[name])
 
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, numpy.array(labels))
    recognizer.save(model_save_path)
 
if __name__ == "__main__":
    main()