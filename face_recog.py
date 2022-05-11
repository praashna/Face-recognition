import numpy as np
import face_recognition as fr
import os
import cv2
import time

def main():

    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    known_face_encondings = []
    known_face_names= []
    for name in os.listdir("D:/face recog/myenv/known_faces"):
        for filename in os.listdir(f'D:/face recog/myenv/known_faces/{name}'):
            image = fr.load_image_file(f'D:/face recog/myenv/known_faces/{name}/{filename}')
            face_encondings = fr.face_encodings(image)[0]
            known_face_encondings.append(face_encondings)
            known_face_names.append(name)


    while True: 
        ret, frame = video_capture.read()

        rgb_frame = frame[:, :, ::-1]

        face_locations = fr.face_locations(rgb_frame)
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            matches = fr.compare_faces(known_face_encondings, face_encoding)

            name = "Unknown"

            face_distances = fr.face_distance(known_face_encondings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        

        cv2.imshow('Webcam_facerecognition', frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return name
            break

    video_capture.release()
    cv2.destroyAllWindows()



if __name__=='__main__':
    main()  