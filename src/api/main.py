from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "สวัสดี FastAPI"}

"http://127.0.0.1:8000/predict?pgnum=20&sode=lh" 
