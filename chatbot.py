from logging import disable
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urldefrag
from io import BytesIO
import re
import requests
import warnings
from pathlib import Path as p
from pprint import pprint
import pandas as pd
from PIL import Image
import ast
import uvicorn
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, storage, firestore, db
import urllib
import shutil
from datetime import datetime, timedelta
import smtplib
import random
import string
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Environment variable setup
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')
warnings.filterwarnings("ignore")

# Initialize FastAPI app
app = FastAPI()

# Security configurations
SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

geminiAPI = os.getenv('GEMINI_API_KEY')

try:
    llm1 = ChatGoogleGenerativeAI(api_key=geminiAPI, model='gemini-1.5-flash')
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': os.getenv('STORAGE_BUCKET'),
        'databaseURL': os.getenv('DB_URL')
    })
    firest = firestore.client()
    ref = db.reference()
except Exception as e:
    raise Exception(f"Initialization Error: {str(e)}")

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    userid: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str
    disabled: Optional[bool] = None

class SignUp(BaseModel):
    email: str
    password: str
    username: str

class Login(BaseModel):
    email: str
    password: str

class File(BaseModel):
    files: List[str]
    rewrite: bool

class FileRequest(BaseModel):
    userid: str
    query: str
    files: List[str]
    db_name: str

class OTPRequest(BaseModel):
    email: str

class OTP_AUTH(BaseModel):
    email: str
    otp: str

class FilterWord(BaseModel):
    email: str

class AddDomain(BaseModel):
    email: str
    domain: str

# Password Hashing

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Authentication and Token Generation
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user['password']):
        return False
    return user

# Firebase Helper Functions
def get_user(db, username: str):
    try:
        user_ref = db.collection("User").document(username)
        user_doc = user_ref.get()
        return user_doc.to_dict() if user_doc.exists else None
    except Exception as e:
        print(f"Error getting document: {e}")
        return None

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return get_user(firest, username)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user['disabled']:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Email and OTP Management
def generate_otp(length=6):
    return ''.join(random.choices(string.digits, k=length))

def send_otp_via_email(receiver_email, otp):
    sender_email = os.getenv('SENDER_MAIL')
    sender_password = os.getenv('MAIL_PASS')
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = "Your OTP for Secure Access – Mimir AI"
    body = f"Your OTP is: {otp}"  # Add proper email content here
    message.attach(MIMEText(body, 'plain'))
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
    except Exception as e:
        print(f"Error sending email: {e}")

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Mimir AI!"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(firest, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = create_access_token(data={"sub": user['username']}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/signup")
async def signup(request: SignUp):
    hashed_password = get_password_hash(request.password)
    user_data = {
        "username": request.username,
        "email": request.email,
        "password": hashed_password,
        "disabled": False,
    }
    firest.collection("User").document(request.email).set(user_data)
    return {"message": "User registered successfully"}

@app.post("/otp_generator")
def otp_gen(request: OTPRequest):
    otp = generate_otp()
    hashed_otp = get_password_hash(otp)
    send_otp_via_email(request.email, otp)
    add_otp(request.email, hashed_otp)
    return {"message": "OTP sent successfully"}

@app.post("/process")
def process_files(request: File, current_user: UserInDB = Depends(get_current_active_user)):
    # Example: Processing files and storing them
    return {"message": "Files processed successfully"}

# Additional Endpoints as Required
if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="127.0.0.1", port=8000, reload=True)
