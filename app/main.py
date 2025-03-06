
from datetime import datetime
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.schema import ChangePassword, UserLogin, UserRegister, Response
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

import requests
import httpx

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

@app.get("/new_test")
async def testAPI():
    return {"message": "Hello world"}

@app.get("/test")
async def test():
    return JSONResponse(content={"message":"Hi Buddy,how are you"})


@app.post("/register2",response_model=Response )
async def register2(email: str = Form(...), password: str = Form(...), file: UploadFile = File(...)):
    logging.info("Calling register")
    try:
        # Save the uploaded file temporarily
        file_path = save_temp_file(file)

        # Generate facial embedding
        embedding = generate_embedding(file_path)

        # Save to databases
        users = get_user2(email)
        # users2 = get_user(embedding)
        nos = len(users)
        logging.info(f" no of same users {nos}")


        logging.info(f" No face is registered yet. So new face registration accepted")
        save_user2(email, embedding)

        # EXTRA CODE
        # await submitDataToSpringAPI()
        return await submitDataToSpringAPI(email)
        # ---- EXTRA CODE

        # return JSONResponse(content={"message": "Face is registered succesfully", "code": "1"})




    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/authenticate",response_model=dict)
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
            user_id, user_email, user_embedding = user

            # Compare embeddings
            if compare_faces(input_embedding, user_embedding):
                os.remove(file_path)
                #SUCCESS EXTRA CODE
                req = requests.get('http://localhost:8080/fromPythonMC/connect')
                return JSONResponse(
                    content={"message": "Authentication successful","user_name":user_email,  "code": "1"})

        os.remove(file_path)
        return JSONResponse(content={"message": "Authentication failed", "code": "2"})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def submitDataToSpringAPI(email):
    url = "http://localhost:8080/fromPythonMC/faceRegister"
    json_body = {"userEmail": email}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=json_body)

    return response.json()







###########################################################Unused codes#################################################

@app.post("/login",response_model=dict)
async def login(user: UserLogin):
    logging.info(f"Calling login api name {user.email} , password {user.password}") 
    try:
        db_user= get_user2(user.email)
        user_id = db_user[0][0]
        name = db_user[0][1]
        email = db_user[0][2]
        password = db_user[0][3]
        embedding = db_user[0][4]
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
             if embedding == None:
                 logging.info(f"Face is not registered")
                 return JSONResponse(content={"message": "Face is not registered","code":"4"})
             else:
                  logging.info(f"Successfully login") 
                  return JSONResponse(content={"message": "Login successfull","code":"1"})
             #return JSONResponse(content={"message": "Login successfull","code":"1"})
                 
                 
            


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



@app.put("/changePassword",response_model=dict)
async def changePassword(changePassword: ChangePassword):

    logging.info(f"Calling changePassword api name {changePassword.email}") 
    try:
        update_password(changePassword.email,changePassword.newPassword)
        return JSONResponse(content={"message": "Password changed successfulyy","code":"1"})
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))






    

