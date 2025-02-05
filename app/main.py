from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from .database import init_db, save_user, get_all_users, get_user
from .utils import save_temp_file, generate_embedding, compare_faces
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import os
import logging 

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