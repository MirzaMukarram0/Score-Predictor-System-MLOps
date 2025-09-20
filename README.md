# 🎯 Score Predictor System (MLOps)

This project is a simple **Flask-based ML web service** that predicts student scores using a trained regression model.  
It also includes a **CI/CD pipeline** with GitHub Actions, Docker, and Jenkins.

---

## 🚀 Features
- **Flask API** with `/predict` endpoint.
- Returns predictions as a single **float** for given input features.
- **Unit tests** with `pytest`.
- **Code quality checks** with `flake8`.
- **Dockerized** for consistent deployment.
- **Jenkins pipeline** for automated builds and pushes to DockerHub.
- **GitHub Actions** for linting and pull request checks.

---

Score-Predictor-System-MLOps/
│
├── app/
│ └── server.py # Flask server
│
├── tests/
│ └── test_server.py # Pytest unit tests
│
├── datasets/
│ └── student_model.pkl # Pretrained model (dummy/regression)
│
├── Dockerfile # Container build instructions
├── Jenkinsfile # CI/CD pipeline for Jenkins
├── requirements.txt # Python dependencies
├── .github/workflows/ # GitHub Actions workflows
└── README.md # Project documentation


---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Score-Predictor-System-MLOps.git
cd Score-Predictor-System-MLOps

2. Create Virtual Environment
python -m venv venv
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Run Flask Server
python app/server.py


Server will run on http://0.0.0.0:5000
