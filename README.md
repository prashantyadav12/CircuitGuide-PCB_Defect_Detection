# 🧠 CircuitGuard – PCB Defect Detection System

CircuitGuard is an **end-to-end PCB defect detection system** built using  
**YOLO (Deep Learning)**, **FastAPI (Backend)**, and **Streamlit (Frontend)**.

It detects common PCB manufacturing defects from uploaded images and provides  
**annotated visual outputs, defect statistics, and downloadable reports**.

---

## 🚀 Features

- 🔍 Detects **6 PCB defect types**
  - Missing Hole  
  - Mouse Bite  
  - Open Circuit  
  - Short  
  - Spur  
  - Spurious Copper  

- 🖼️ **Annotated defect visualization**
  - Bounding boxes
  - Class labels
  - Confidence scores

- 📊 **Defect analytics**
  - Bar chart (defect count)
  - Donut chart (defect distribution)

- 💾 **Backend image storage** for traceability

- 📥 **Download options**
  - Annotated images
  - CSV + ZIP reports

- ⚡ **Real-time inference** using YOLO

- 🧩 **Modular frontend–backend architecture**

---

## 🏗️ Project Architecture

# 🧠 CircuitGuard – PCB Defect Detection System

CircuitGuard is an **end-to-end PCB defect detection system** built using  
**YOLO (Deep Learning)**, **FastAPI (Backend)**, and **Streamlit (Frontend)**.

It detects common PCB manufacturing defects from uploaded images and provides  
**annotated visual outputs, defect statistics, and downloadable reports**.

---

## 🚀 Features

- 🔍 Detects **6 PCB defect types**
  - Missing Hole  
  - Mouse Bite  
  - Open Circuit  
  - Short  
  - Spur  
  - Spurious Copper  

- 🖼️ **Annotated defect visualization**
  - Bounding boxes
  - Class labels
  - Confidence scores

- 📊 **Defect analytics**
  - Bar chart (defect count)
  - Donut chart (defect distribution)

- 💾 **Backend image storage** for traceability

- 📥 **Download options**
  - Annotated images
  - CSV + ZIP reports

- ⚡ **Real-time inference** using YOLO

- 🧩 **Modular frontend–backend architecture**

---

## 🏗️ Project Architecture

CircuitGuard/
│
├── app.py # Streamlit Frontend
├── pcb-defect-backend/
│ ├── main.py # FastAPI Backend
│ ├── model/
│ │ └── best.pt # Trained YOLO model
│ └── uploads/ # Images saved by backend
│
├── screenshots/ # UI screenshots
├── requirements.txt
├── packages.txt
├── runtime.txt
├── README.md


---

## 🔄 System Workflow

### 🔹 Old Setup (Frontend-only)
- Streamlit directly loaded YOLO model  
- Inference + annotation done locally  
- ❌ No backend  
- ❌ No image persistence  

### 🔹 Current Setup (Frontend + Backend)

1. User uploads image via **Streamlit frontend**
2. Frontend sends image → `POST /predict`
3. **FastAPI backend**
   - Saves image to `/uploads`
   - Runs YOLO inference
   - Returns structured JSON response
4. Frontend
   - Displays annotated images
   - Shows statistics & charts
   - Enables downloads

✔ This confirms **frontend ↔ backend connection**

---

## 📁 Backend Proof of Connection

When an image is uploaded from the frontend, it is saved here:


---

## 🔄 System Workflow

### 🔹 Old Setup (Frontend-only)
- Streamlit directly loaded YOLO model  
- Inference + annotation done locally  
- ❌ No backend  
- ❌ No image persistence  

### 🔹 Current Setup (Frontend + Backend)

1. User uploads image via **Streamlit frontend**
2. Frontend sends image → `POST /predict`
3. **FastAPI backend**
   - Saves image to `/uploads`
   - Runs YOLO inference
   - Returns structured JSON response
4. Frontend
   - Displays annotated images
   - Shows statistics & charts
   - Enables downloads

✔ This confirms **frontend ↔ backend connection**

---

## 📁 Backend Proof of Connection

When an image is uploaded from the frontend, it is saved here:

pcb-defect-backend/uploads/
├── 01_missing_hole_10.jpg
├── 01_spur_09.jpg


This proves:
**Frontend uploads → Backend receives → Backend stores**

---

## 🖥️ Frontend (Streamlit)

**Responsibilities**
- Upload PCB images (single / multiple)
- Send images to backend API
- Display:
  - Original image
  - Annotated image
  - Defect statistics & charts
- Allow result downloads

**Run Frontend**
```bash
python -m streamlit run app.py
📍 URL: http://localhost:8501

⚙️ Backend (FastAPI)
Responsibilities
Accept images via /predict
Save uploaded images
Run YOLO inference
Return structured JSON response
Run Backend
cd pcb-defect-backend
uvicorn main:app --reload
📍 API: http://127.0.0.1:8000
📘 Swagger Docs: http://127.0.0.1:8000/docs

📡 API Endpoint
POST /predict
Input
Multipart form-data
Image file (png, jpg, jpeg)
Sample Response
{
  "status": "success",
  "defects_detected": {
    "spur": 1
  },
  "total_defects": 1
}

🧠 Model Details
Model: YOLO (Ultralytics)
Input: PCB top-view images
Performance
mAP@50: 0.98
Precision: 0.97
Recall: 0.97

⚠️ Known Limitations
Some visualization is still handled by frontend
Duplicate images may appear in rare cases
Backend currently returns limited metadata

🔮 Future Improvements
Fully backend-driven rendering
Database integration (MongoDB / PostgreSQL)
Authentication & user sessions
Dockerization
Cloud deployment (AWS / Azure)
Async batch processing


🛠️ Tech Stack
Python 3.11
YOLO (Ultralytics)
FastAPI
Streamlit
OpenCV / PIL
Altair
Uvicorn


👨‍💻 Author
Prashant Yadav
B.Tech CSE (AI)
PCB Defect Detection – Internship Project


