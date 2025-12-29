🧠 CircuitGuard – PCB Defect Detection System
CircuitGuard is an end-to-end PCB defect detection system built using YOLO (Deep Learning), FastAPI (Backend), and Streamlit (Frontend).
It detects and visualizes common PCB manufacturing defects from uploaded images and provides annotated outputs, statistics, and downloadable results.
🚀 Features
🔍 Detects 6 PCB defect types
Missing hole
Mouse bite
Open circuit
Short
Spur
Spurious copper
🖼️ Annotated defect visualization (bounding boxes + confidence)
📊 Defect statistics (bar chart + donut chart)
📁 Backend image storage for traceability
⬇️ Download annotated images and CSV/ZIP reports
⚡ Real-time inference using YOLO
🌐 Modular frontend–backend architecture
🏗️ Project Architecture
CircuitGuard/
│
├── app.py                     # Streamlit Frontend
├── pcb-defect-backend/
│   ├── main.py                # FastAPI Backend
│   ├── model/
│   │   └── best.pt            # Trained YOLO model
│   ├── uploads/               # Images saved by backend
│   └── __pycache__/
│
├── requirements.txt
├── packages.txt
├── runtime.txt
├── README.md
└── screenshots/
🔄 System Workflow (IMPORTANT)
🔹 Before Backend (Old Setup)
Streamlit frontend directly loaded YOLO model
Inference + annotation happened inside frontend
No backend involvement
No image persistence
🔹 After Backend Integration (Current Setup)
Frontend (Streamlit)
   ↓  HTTP POST /predict
Backend (FastAPI)
   ↓  Saves image to /uploads
   ↓  Runs YOLO inference
   ↓  Returns JSON response
Frontend
   ↓  Displays annotated results
✔ This proves frontend ↔ backend connection
🧪 Backend Proof of Connection
When an image is uploaded from frontend:
It is saved here:
pcb-defect-backend/uploads/
Example:
uploads/
 ├── 01_missing_hole_10.jpg
 ├── 01_spur_09.jpg
This confirms:
Frontend uploads → Backend receives → Backend stores
🖥️ Frontend (Streamlit)
Responsibilities
Image upload (single / multiple)
Sends images to backend API
Displays:
Original image
Annotated image
Defect count
Charts & statistics
Allows downloads (image / ZIP)
Run Frontend
python -m streamlit run app.py
Frontend runs on:
http://localhost:8501
⚙️ Backend (FastAPI)
Responsibilities
Accept image via /predict endpoint
Save image to disk
Run YOLO inference
Return structured JSON response
Run Backend
cd pcb-defect-backend
uvicorn main:app --reload
Backend runs on:
http://127.0.0.1:8000
Swagger Docs:
http://127.0.0.1:8000/docs
📡 API Endpoint
POST /predict
Input
Multipart form-data
Image file (png, jpg, jpeg)
Response
{
  "status": "success",
  "defects_detected": {
    "spur": 1
  },
  "total_defects": 1
}
📊 Visual Outputs
Annotated PCB images (bounding boxes + labels)
Bar chart: defect count
Donut chart: defect distribution
Detailed per-image defect tables
🧠 Model Details
Model: YOLO (Ultralytics)
Input: PCB top-view images
Performance:
mAP@50: 0.98
Precision: 0.97
Recall: 0.97
⚠️ Known Limitations
Frontend still does some local processing for visualization
Duplicate images may appear if both local + backend results are rendered
Backend currently returns limited metadata (can be extended)
🔮 Future Improvements
Full backend-driven annotation rendering
Database integration (MongoDB / PostgreSQL)
Authentication & user sessions
Dockerization
Cloud deployment (AWS / Azure)
Async batch processing
🧑‍💻 Tech Stack
Python 3.11
YOLO (Ultralytics)
FastAPI
Streamlit
OpenCV / PIL
Altair
Uvicorn
👨‍🎓 Author
Prashant Yadav
B.Tech CSE (AI)
PCB Defect Detection – Internship Project




