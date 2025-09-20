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

## 📦 Project Structure
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

python -m venv venv
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
python app/server.py

🧪 Running Tests

Run all tests with:

pytest -v

Check linting with:

flake8 app tests

Docker
Build Docker Image
docker build -t score-predictor .

Run Container
docker run -p 5000:5000 score-predictor


Then test the API at:
http://localhost:5000/predict

🔄 CI/CD
GitHub Actions

Runs flake8 automatically on dev branch.

Runs pytest on pull requests to test and main.

Jenkins

Jenkinsfile builds Docker image, runs tests, and pushes image to DockerHub.

Uses dockerhub-creds credentials.

Sends email notifications on failure/success.

📌 Tech Stack

Python 3.10+
Flask
scikit-learn
pytest
flake8
Docker
Jenkins
GitHub Actions