from deepface import DeepFace
import os
import logging

def save_temp_file(file):
    file_path = f"app/temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path

def generate_embedding(file_path):
    return DeepFace.represent(img_path=file_path, model_name="Facenet")[0]["embedding"]

def compare_faces(embedding1, embedding2):
    result = DeepFace.verify(embedding1, eval(embedding2), model_name="Facenet",enforce_detection=False,anti_spoofing=True)
   # ver, dis, thres, model, dis_met,fa, time  = result
  # logging.info(f" verified: {ver}, distance: {dis}, threshold: {thres}, model: {model}, distanc_met: {dis_met}, facial_area: {fa}, time: {time}")
    return result["verified"]