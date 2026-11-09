# Instamart Customer Intelligence & Support Copilot

## 1. Project Overview

The Instamart Customer Intelligence & Support Copilot is a Python-based data analytics and AI-powered customer support application.

The project combines customer segmentation, churn prediction, support ticket classification, policy-based retrieval, and an AI refund assistant.

It also provides a FastAPI backend and a React frontend to interact with the application.

## 2. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- SQLite
- FastAPI
- Uvicorn
- React
- Vite
- JavaScript
- Sentence Transformers

## 3. Project Modules

### Part A – Data Analysis and Dashboard

- Customer, product, and order data analysis.
- Sales analysis by city.
- Data visualization.
- Sales dashboard generation.

### Part B – Customer Intelligence

- Customer segmentation using machine learning.
- K-Means clustering.
- Customer churn prediction.
- Customer risk analysis.
- Retention insights.

### Part C – Support Ticket Intelligence

- Support ticket category prediction.
- Sentiment analysis.
- Urgency and priority classification.
- Machine learning-based ticket intelligence.

### Part D – AI Refund Agent

- Refund policy document processing.
- Semantic search using Sentence Transformers.
- Policy retrieval using embeddings.
- AI-generated customer support responses.
- Human escalation for critical complaints.

## 4. Backend – FastAPI

The backend provides REST API endpoints for application interaction.

Main endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome |
| GET | `/health` | Health check |
| GET | `/api/info` | Application information |
| POST | `/api/complaints` | Process customer complaints |

API documentation is available at `/docs` when the backend is running.

## 5. Frontend – React

The React frontend provides a user interface to submit customer complaints and view the AI-generated support response.

The frontend communicates with the FastAPI backend.

## 6. Project Structure

```text
Instamart_Copilot/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── database/
│
├── frontend/
│
├── outputs/
│
├── reports/
│
├── src/
│   ├── api/
│   ├── part_a/
│   ├── part_b/
│   ├── part_c/
│   └── part_d/
│
├── requirements.txt
├── generate_data.py
├── .gitignore
└── README.md
```

## 7. Installation and Execution

### Step 1 – Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Instamart_Copilot
```

### Step 2 – Create Virtual Environment

```bash
python -m venv venv
```

Activate on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Start FastAPI Backend

From the project root:

```bash
uvicorn src.api.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### Step 5 – Start React Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the frontend URL shown by Vite, usually:

```text
http://localhost:5173
```

## 8. Testing

The application has been tested with sample customer complaints, including:

- Damaged product complaints.
- Missing item complaints.
- Payment failure complaints.

The API returns complaint classification, sentiment, urgency, priority, policy retrieval information, AI response, and human escalation status.

## 9. Limitations and Next Steps

- The application uses synthetic project data.
- Machine learning predictions depend on the training dataset.
- AI-generated responses are intended as support suggestions.
- Real payment processing and actual refund execution are not connected.
- Further improvements can include authentication, deployment, and integration with real customer systems.

## 10. Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates how data analytics, machine learning, semantic retrieval, FastAPI, and React can work together to improve customer support and decision-making.