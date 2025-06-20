# Medical Admissions AI Assistant - Setup & Deployment Guide

## Quick Start (Local Testing)

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv medical_admissions_env
source medical_admissions_env/bin/activate  # On Windows: .\medical_admissions_env\Scripts\activate

# Install dependencies
pip install streamlit pandas numpy plotly openai scikit-learn PyPDF2 openpyxl joblib
```

### 2. Create the Application File
Save the Streamlit app code as `medical_admissions_app.py`

### 3. Run Locally
```bash
streamlit run medical_admissions_app.py
```

The app will open at `http://localhost:8501`

## Production Deployment Options

### Option 1: Streamlit Cloud (Fastest - Recommended for POC)

1. **Push to GitHub**
```bash
# Create requirements.txt
echo "streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
plotly==5.17.0
openai==1.3.0
scikit-learn==1.3.0
PyPDF2==3.0.1
openpyxl==3.1.2
joblib==1.3.0" > requirements.txt

# Create .gitignore
echo "*.pyc
__pycache__/
.env
*.xlsx
*.pdf
*.csv" > .gitignore

# Push to GitHub
git init
git add .
git commit -m "Initial medical admissions app"
git remote add origin YOUR_GITHUB_REPO
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect your GitHub account
- Select your repository
- Deploy!

3. **Add Secrets**
In Streamlit Cloud settings, add:
```toml
[secrets]
openai_api_key = "your-openai-api-key"
```

### Option 2: AWS EC2 with Docker

1. **Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY medical_admissions_app.py .
COPY models/ ./models/

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "medical_admissions_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Deploy Script**
```bash
# Build and push to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_URI
docker build -t medical-admissions .
docker tag medical-admissions:latest YOUR_ECR_URI/medical-admissions:latest
docker push YOUR_ECR_URI/medical-admissions:latest

# Deploy on EC2
ssh ec2-user@YOUR_EC2_IP
docker pull YOUR_ECR_URI/medical-admissions:latest
docker run -d -p 80:8501 --env OPENAI_API_KEY=$OPENAI_API_KEY YOUR_ECR_URI/medical-admissions:latest
```

### Option 3: Azure App Service

1. **Prepare for Azure**
```bash
# Create startup command
echo "streamlit run medical_admissions_app.py --server.port 8000 --server.address 0.0.0.0" > startup.txt
```

2. **Deploy with Azure CLI**
```bash
# Create resource group
az group create --name MedicalAdmissionsRG --location eastus

# Create app service plan
az appservice plan create --name MedicalAdmissionsPlan --resource-group MedicalAdmissionsRG --sku B1 --is-linux

# Create web app
az webapp create --resource-group MedicalAdmissionsRG --plan MedicalAdmissionsPlan --name medical-admissions-app --runtime "PYTHON:3.9"

# Configure startup
az webapp config set --resource-group MedicalAdmissionsRG --name medical-admissions-app --startup-file startup.txt

# Set environment variables
az webapp config appsettings set --resource-group MedicalAdmissionsRG --name medical-admissions-app --settings OPENAI_API_KEY="your-key"

# Deploy code
az webapp deployment source config --name medical-admissions-app --resource-group MedicalAdmissionsRG --repo-url YOUR_GITHUB_REPO --branch main
```

## Enhanced Production Architecture

### Full-Scale Implementation
```
┌─────────────────────────────────────────────────────────────┐
│                        Load Balancer                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
┌───────▼────────┐                        ┌────────▼────────┐
│   Web Server   │                        │   Web Server    │
│  (React App)   │                        │  (React App)    │
└───────┬────────┘                        └────────┬────────┘
        │                                           │
        └─────────────────┬─────────────────────────┘
                          │
                ┌─────────▼──────────┐
                │   API Gateway      │
                │   (Rate Limiting)  │
                └─────────┬──────────┘
                          │
        ┌─────────────────┴─────────────────────┐
        │                                       │
┌───────▼────────┐                    ┌────────▼────────┐
│  FastAPI       │                    │   FastAPI       │
│  Backend #1    │                    │   Backend #2    │
└───────┬────────┘                    └────────┬────────┘
        │                                       │
        └────────────┬──────────────────────────┘
                     │
           ┌─────────┴──────────┐
           │                    │
     ┌─────▼──────┐      ┌──────▼─────┐
     │   Redis    │      │ PostgreSQL │
     │   Queue    │      │  Database  │
     └────────────┘      └────────────┘
```

### Backend API (FastAPI)
```python
# api/main.py
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import List
import uuid

app = FastAPI(title="Medical Admissions API")

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/process-batch")
async def process_batch(
    applicants_file: UploadFile,
    essays_file: UploadFile,
    background_tasks: BackgroundTasks
):
    # Generate batch ID
    batch_id = str(uuid.uuid4())
    
    # Queue for processing
    background_tasks.add_task(
        process_applications_async,
        batch_id,
        applicants_file,
        essays_file
    )
    
    return {"batch_id": batch_id, "status": "processing"}

@app.get("/api/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    # Check processing status
    status = await check_batch_status(batch_id)
    return status

@app.get("/api/batch/{batch_id}/results")
async def get_results(batch_id: str):
    # Return processed results
    results = await get_batch_results(batch_id)
    return results
```

### React Frontend
```javascript
// frontend/src/App.js
import React, { useState } from 'react';
import { Upload, Process, Results } from './components';

function App() {
  const [batchId, setBatchId] = useState(null);
  const [results, setResults] = useState(null);
  
  const handleUpload = async (files) => {
    const formData = new FormData();
    formData.append('applicants_file', files.applicants);
    formData.append('essays_file', files.essays);
    
    const response = await fetch('/api/process-batch', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    setBatchId(data.batch_id);
    pollForResults(data.batch_id);
  };
  
  const pollForResults = async (id) => {
    const interval = setInterval(async () => {
      const response = await fetch(`/api/batch/${id}/status`);
      const status = await response.json();
      
      if (status.status === 'complete') {
        clearInterval(interval);
        const resultsResponse = await fetch(`/api/batch/${id}/results`);
        const results = await resultsResponse.json();
        setResults(results);
      }
    }, 5000);
  };
  
  return (
    <div className="App">
      {!results ? (
        <Upload onUpload={handleUpload} />
      ) : (
        <Results data={results} />
      )}
    </div>
  );
}
```

## Security & Compliance

### 1. Data Security
```python
# Encryption at rest
from cryptography.fernet import Fernet

class SecureStorage:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt_file(self, file_path):
        with open(file_path, 'rb') as f:
            encrypted = self.cipher.encrypt(f.read())
        return encrypted
    
    def decrypt_file(self, encrypted_data):
        return self.cipher.decrypt(encrypted_data)
```

### 2. Access Control
```python
# Role-based access
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Verify token and permissions
    user = verify_token(token)
    if not user.has_permission("review_applications"):
        raise HTTPException(status_code=403)
    return user
```

### 3. Audit Logging
```python
import logging
from datetime import datetime

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger("audit")
        
    def log_access(self, user_id, action, resource):
        self.logger.info({
            "timestamp": datetime.utcnow(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "ip_address": get_client_ip()
        })
```

## Performance Optimization

### 1. Batch Processing
```python
# Process in batches to optimize API calls
def process_essays_batch(essays: List[str], batch_size: int = 5):
    results = []
    
    for i in range(0, len(essays), batch_size):
        batch = essays[i:i + batch_size]
        # Process batch with OpenAI
        batch_results = parallel_process(batch)
        results.extend(batch_results)
    
    return results
```

### 2. Caching
```python
import redis
import hashlib

class FeatureCache:
    def __init__(self):
        self.redis_client = redis.Redis()
    
    def get_cached_features(self, text):
        key = hashlib.md5(text.encode()).hexdigest()
        cached = self.redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    def cache_features(self, text, features):
        key = hashlib.md5(text.encode()).hexdigest()
        self.redis_client.setex(
            key, 
            86400,  # 24 hour cache
            json.dumps(features)
        )
```

## Cost Optimization

### OpenAI API Usage
- **GPT-4**: ~$0.03 per application (1K tokens in, 100 tokens out)
- **Batch of 500 applicants**: ~$15
- **Annual cost (5000 applicants)**: ~$150

### Optimization Strategies
1. Cache repeated essays
2. Use GPT-3.5-turbo for initial screening
3. Only use GPT-4 for borderline cases
4. Batch API calls

## Next Steps

### Phase 1: Proof of Concept (Current)
- ✅ Streamlit app with basic functionality
- ✅ OpenAI integration for text analysis
- ✅ Visualization of results
- ✅ Export capabilities

### Phase 2: Production MVP
- [ ] Add authentication
- [ ] Implement proper ML model training
- [ ] Add database for result storage
- [ ] Enhanced error handling
- [ ] Batch processing optimization

### Phase 3: Enterprise Features
- [ ] Multi-user support with roles
- [ ] Advanced analytics dashboard
- [ ] Integration with existing systems
- [ ] Automated report generation
- [ ] Historical comparison tools

### Phase 4: Advanced AI
- [ ] Fine-tune custom LLM
- [ ] Active learning from reviewer feedback
- [ ] Bias detection and mitigation
- [ ] Explainable AI features

## Support & Maintenance

### Monitoring
```python
# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "openai_status": check_openai_connection(),
        "database_status": check_db_connection()
    }
```

### Error Handling
```python
# Graceful error handling
try:
    result = process_application(data)
except openai.RateLimitError:
    # Queue for retry
    await queue_for_retry(data)
except Exception as e:
    # Log and alert
    logger.error(f"Processing failed: {str(e)}")
    send_alert_to_admin(e)
```

This setup provides a complete path from proof of concept to production deployment!
