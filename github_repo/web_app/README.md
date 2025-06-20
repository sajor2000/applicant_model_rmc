# Rush AI Admissions Web Application

A Streamlit-based web interface for processing medical school applications through the AI model.

## Features

- 🔐 Secure authentication
- 📤 Batch file upload and processing
- 📊 Real-time results visualization
- 📈 Analytics dashboard
- 📥 Export functionality
- 🎨 Rush University branding

## Running the Application

### Local Development

```bash
streamlit run app.py
```

### Configuration

The app uses environment variables from `.env`:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`

### Pages

1. **Dashboard**: Overview and quick stats
2. **Process Applications**: Upload and process files
3. **View Results**: Filter, search, and analyze results
4. **Settings**: Configuration and model info

## File Structure

- `app.py` - Main Streamlit application
- `processor.py` - Application processing logic  
- `results.py` - Results display module
- `static/` - CSS and images
- `templates/` - HTML templates (if needed)

## Deployment

### Option 1: Local Server
Best for internal use within Rush network.

### Option 2: Azure App Service
For cloud deployment with proper authentication.

### Option 3: Streamlit Cloud
Quick deployment option for demos.

## Security

- Implement proper authentication (LDAP/SSO)
- Use HTTPS for all connections
- Follow HIPAA guidelines for data handling
- Regular security audits