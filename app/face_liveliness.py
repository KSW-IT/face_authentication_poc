import numpy as np
import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh # To draw mash in the face
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)

def display_timer(image, remaining_time):
    """ Display a countdown timer on the image """
    cv2.putText(image, f"Time left: {remaining_time}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

def detect_head_movement(cap, face_mesh, direction, angle_threshold, instruction):
    timer_duration = 10  # Timer duration in seconds
    start_time = time.time()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture video frame")
            return False

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        img_h, img_w, img_c = image.shape

        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.putText(image, instruction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elapsed_time = time.time() - start_time
        remaining_time = max(0, int(timer_duration - elapsed_time))
        display_timer(image, remaining_time)

        if remaining_time == 0:
            return False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_2d = []
                face_3d = []
                nose_2d = None
                nose_3d = None

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 1:  # Nose tip
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    if idx in [33, 263, 61, 291, 199]:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
                if not success:
                    print("Failed to solve PnP problem")
                    return False

                rmat, _ = cv2.Rodrigues(rotation_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                x_angle = angles[0] * 360
                y_angle = angles[1] * 360

                mp_drawing.draw_landmarks(image=image,
                                          landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=drawing_spec,
                                          connection_drawing_spec=drawing_spec)

                if nose_2d and nose_3d:
                    nose_3d_projection, _ = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0]), int(nose_2d[1] - x_angle * 10))
                    cv2.line(image, p1, p2, (255, 0, 0), 3)

                if (direction == "up" and x_angle > angle_threshold) or \
                   (direction == "down" and x_angle < -8) or \
                   (direction == "left" and y_angle < angle_threshold) or \
                   (direction == "right" and y_angle > angle_threshold):
                    return True

        cv2.imshow('Head Pose Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    return False

def show_initial_instruction(cap, instruction):
    start_time = time.time()
    while time.time() - start_time < 1:
        elapsed_time = time.time() - start_time
        remaining_time = max(0, int(20 - elapsed_time))
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture initial video frame")
            return
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        cv2.putText(frame, instruction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # put text in the frame
        display_timer(frame, remaining_time)
        cv2.imshow('Head Pose Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # show image in the frame , 'Head Pose Detection' is title
        if cv2.waitKey(1) & 0xFF == 27:
            break

def active_liveliness_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to access webcam")
        return

    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        show_initial_instruction(cap, "Verify your head movement")
        if detect_head_movement(cap, face_mesh, direction="up", angle_threshold=10, instruction="Move your head Up"):
            show_initial_instruction(cap, "Move your head Down")
            if detect_head_movement(cap, face_mesh, direction="down", angle_threshold=-10, instruction="Move your head Down"):
                show_initial_instruction(cap, "Move your head Left")
                if detect_head_movement(cap, face_mesh, direction="left", angle_threshold=-10, instruction="Move your head Left"):
                    show_initial_instruction(cap, "Move your head Right")
                    if detect_head_movement(cap, face_mesh, direction="right", angle_threshold=10, instruction="Move your head Right"):
                        print("Verification successful!")
                    else:
                        print("Verification failed for Move your head Right")
                else:
                    print("Verification failed for Move your head Left")
            else:
                print("Verification failed for Move your head Down")
        else:
            print("Verification failed for Move your head Up")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    active_liveliness_detection()
