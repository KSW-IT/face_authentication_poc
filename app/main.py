
from datetime import datetime
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.schema import ChangePassword, UserLogin, UserRegister
from .database import create_user, get_user2, init_db, save_user, get_all_users, get_user, save_user2, update_password
from .utils import save_temp_file, generate_embedding, compare_faces
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import os
import logging 

#adding packeages for checking liveliness
import numpy as np
import cv2
import mediapipe as mp
import time
#Finish

#1version
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect
from starlette.websockets import WebSocketDisconnect
import base64

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize database
init_db()
@app.get("/test")
async def test():
    return JSONResponse(content={"message":"Hi Buddy,how are you"})

@app.post("/login",response_model=dict)
async def login(user: UserLogin):
    logging.info(f"Calling login api name {user.email} , password {user.password}") 
    try:
        db_user= get_user2(user.email)
        user_id = db_user[0][0]
        name = db_user[0][1]
        email = db_user[0][2]
        password = db_user[0][3]
        create_timestamp = db_user[0][5]
        expire_timestamp = db_user[0][6]
        logging.info(f" {user_id}, {name}, {email}, {create_timestamp}, {expire_timestamp} ")
        if len(db_user) ==0:
            logging.info(f"No user found") 
            return JSONResponse(content={"message": "No user found","code":"2"})
        
        password =db_user[0][3] 
        logging.info(f"Value {password}") 
        if password == user.password:
            expire_time = datetime.strptime(expire_timestamp, "%Y-%m-%d %H:%M:%S")
            current_time = datetime.now()
            if current_time > expire_time:
        
             logging.info(f"Password expired.") 
             return JSONResponse(content={"message": "Password expired. Please reset password.","code":"3"})
            else:
             logging.info(f"Successfully login") 
             return JSONResponse(content={"message": "Login successfull","code":"1"})


        else:
            logging.info(f"Incorrect password") 
            return JSONResponse(content={"message": "Incorrect password ","code":"2"})



        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.post("/registerUser",response_model=dict)
async def login(user: UserRegister):
    logging.info(f"Calling registerUser api name {user.username} , password {user.password}") 
   
    try:
        create_user(user.username,user.email,user.password)
        return JSONResponse(content={"message": "User register successfully","code":"1"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))




@app.post("/register")
async def register(name: str =Form(...), email: str=Form(...), file: UploadFile = File(...)):
    logging.info("Calling register") 
    try:
        # Save the uploaded file temporarily
        file_path = save_temp_file(file) 
        
        
        # Generate facial embedding
        embedding = generate_embedding(file_path)
        
        # Save to databases
        users = get_user(email)
        #users2 = get_user(embedding)
        nos =len(users)
        logging.info(f" no of same users {nos}")
        #logging.info(f" no of same users according embedding {nos}")
        if nos >0 :
            return JSONResponse(content={"message": "User already exists","code":"2"})
        else:
            save_user(name, email, embedding)
             # Clean up
            os.remove(file_path)
            return JSONResponse(content={"message": "User registered successfully","code":"1"})
            
       
        
        
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/register2")
async def register2(email: str=Form(...), file: UploadFile = File(...)):
    logging.info("Calling register") 
    try:
        # Save the uploaded file temporarily
        file_path = save_temp_file(file) 
        
        
        # Generate facial embedding
        embedding = generate_embedding(file_path)
        
        # Save to databases
        users = get_user2(email)
        #users2 = get_user(embedding)
        nos =len(users)
        logging.info(f" no of same users {nos}")
        #logging.info(f" no of same users according embedding {nos}")
        
        if nos >0 :
            embeddedFromdb =users[0][4] 

            if embeddedFromdb == None: # for time being lets register only the user who is not create face embedded
                logging.info(f" No face is registered yet. So new face registration accepted") 
                save_user2(email, embedding)
                return JSONResponse(content={"message": "Face is registered succesfully","code":"1"})

            else:
                logging.info(f" Face is already registered. Re-register face is not allowed right now") 
                
                return JSONResponse(content={"message": "Face is already registered. Re-register face is not allowed right now","code":"2"})

            
           
        else:
            return JSONResponse(content={"message": "No user found","code":"2"})
        
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/changePassword",response_model=dict)
async def changePassword(changePassword: ChangePassword):

    logging.info(f"Calling changePassword api name {changePassword.email}") 
    try:
        update_password(changePassword.email,changePassword.newPassword)
        return JSONResponse(content={"message": "Password changed successfulyy","code":"1"})
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))





@app.post("/authenticate")
async def authenticate(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = save_temp_file(file)
        
        # Generate facial embedding for the input image
        input_embedding = generate_embedding(file_path)
        
        # Compare with registered faces
        users = get_all_users()
        # status code = 1:authenticate, 2: authentication fail,
        for user in users:
            user_id, user_name, user_email, user_embedding = user
            
            # Compare embeddings
            if compare_faces(input_embedding, user_embedding):
                os.remove(file_path)
                return JSONResponse(content={"message": "Authentication successful", "user_id": user_id, "user_name": user_name,"email":user_email,"code":"1"})
        
        os.remove(file_path)
        return JSONResponse(content={"message": "Authentication failed", "code":2})
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

# Here I wil add the liveliness code 
mp_face_mesh = mp.solutions.face_mesh # To draw mash in the face
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)

def detect_head_movement(cap, face_mesh, direction, angle_threshold):
    """Detects head movement and returns True if the movement meets the threshold."""
    start_time = time.time()
    timer_duration = 10

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            return False

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        img_h, img_w, img_c = image.shape
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        elapsed_time = time.time() - start_time
        if elapsed_time > timer_duration:
            return False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_2d = []
                face_3d = []
                for idx, lm in enumerate(face_landmarks.landmark):
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

                success, rotation_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
                if not success:
                    return False

                rmat, _ = cv2.Rodrigues(rotation_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                x_angle = angles[0] * 360
                y_angle = angles[1] * 360

                if (direction == "up" and x_angle > angle_threshold) or \
                   (direction == "down" and x_angle < -8) or \
                   (direction == "left" and y_angle < angle_threshold) or \
                   (direction == "right" and y_angle > angle_threshold):
                    return True

    return False


@app.get("/liveliness")
def liveliness_check():

    """Runs the full liveliness check sequence and returns the result."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return JSONResponse(content={"status": "error", "message": "Webcam not accessible"}, status_code=500)

    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        if detect_head_movement(cap, face_mesh, "up", 10):
            if detect_head_movement(cap, face_mesh, "down", -10):
                if detect_head_movement(cap, face_mesh, "left", -10):
                    if detect_head_movement(cap, face_mesh, "right", 10):
                        return JSONResponse(content={"status": "success", "message": "Liveliness check passed"})
                    return JSONResponse(content={"status": "error", "message": "Failed at move right"})
                return JSONResponse(content={"status": "error", "message": "Failed at move left"})
            return JSONResponse(content={"status": "error", "message": "Failed at move down"})
        return JSONResponse(content={"status": "error", "message": "Failed at move up"})
    finally:
        cap.release()
        cv2.destroyAllWindows()


#below codes are approach using preview camera and socket

#1st version
# 1st version is not properly working as of now.
cap = cv2.VideoCapture(0)
@app.get("/")
async def get():
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    start_time = time.time()
    timer_duration = 10  # Seconds

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            img_h, img_w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image=frame, 
                                              landmark_list=face_landmarks, 
                                              connections=mp_face_mesh.FACEMESH_CONTOURS,
                                              landmark_drawing_spec=drawing_spec,
                                              connection_drawing_spec=drawing_spec)

            elapsed_time = int(time.time() - start_time)
            remaining_time = max(0, timer_duration - elapsed_time)
            cv2.putText(frame, f"Time left: {remaining_time}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            await websocket.send_bytes(buffer.tobytes())

            if remaining_time == 0:
                await websocket.send_text("Liveliness check failed")
                break

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    finally:
        cap.release()


#2nd Version
def detect_liveliness(image_np):
    """ Process the image and detect head movement """
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        return True  # Face detected (for simplicity)
    return False  # No face detected
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            image_data = base64.b64decode(data.split(",")[1])
            np_arr = np.frombuffer(image_data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            is_live = detect_liveliness(image_np)
            await websocket.send_json({"liveliness": is_live})
        except Exception as e:
            await websocket.close()
            break