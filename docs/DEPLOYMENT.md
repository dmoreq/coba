# COBA-Web Deployment Guide

This guide covers deploying the coba-web platform (frontend + backend) locally and to production.

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Docker Deployment](#docker-deployment)
3. [Environment Variables](#environment-variables)
4. [Database Setup](#database-setup)
5. [CI/CD Integration](#cicd-integration)
6. [Monitoring and Logs](#monitoring-and-logs)
7. [Troubleshooting](#troubleshooting)

---

## Local Development Setup

### Prerequisites

- **Node.js** 20+ (check with `node --version`)
- **Python** 3.10+ (check with `python3 --version`)
- **pip** (Python package manager)
- **npm** or **pnpm** (Node package manager)

### Backend Setup

```bash
cd coba-web/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at **http://localhost:8000**

API documentation (Swagger UI) at **http://localhost:8000/docs**

### Frontend Setup

```bash
cd coba-web/frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at **http://localhost:3000**

### Full Stack Development

Run both in parallel (separate terminals):

```bash
# Terminal 1: Backend
cd coba-web/backend
source venv/bin/activate
python -m uvicorn main:app --reload

# Terminal 2: Frontend
cd coba-web/frontend
npm run dev
```

Then open **http://localhost:3000** in your browser.

---

## Docker Deployment

### Build Docker Images

#### Backend Image

```bash
cd coba-web/backend
docker build -t coba-web-backend:latest .
```

#### Frontend Image

```bash
cd coba-web/frontend
docker build -t coba-web-frontend:latest .
```

### Run with Docker Compose

```bash
# From project root
docker-compose up -d
```

This starts:
- Backend on **http://localhost:8000**
- Frontend on **http://localhost:3000**

### Stop Services

```bash
docker-compose down
```

---

## Environment Variables

### Backend (.env file in coba-web/backend/)

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# CORS (comma-separated origins)
CORS_ORIGINS=http://localhost:3000,http://localhost

# Session Management
SESSION_CLEANUP_INTERVAL=3600  # Cleanup interval in seconds
SESSION_TIMEOUT=86400          # Session timeout in seconds

# Logging
LOG_LEVEL=INFO

# Optional: Database
DATABASE_URL=sqlite:///./coba.db
```

### Frontend (.env.local file in coba-web/frontend/)

```bash
# API endpoint
NEXT_PUBLIC_API_URL=http://localhost:8000

# Feature flags
NEXT_PUBLIC_ENABLE_PROGRESS_TRACKING=true
NEXT_PUBLIC_ENABLE_QUIZZES=true

# Analytics (optional)
NEXT_PUBLIC_GOOGLE_ANALYTICS_ID=
```

---

## Database Setup

### SQLite (Default, for local development)

SQLite databases are created automatically in the backend directory. No setup needed.

```bash
# To reset database (delete all data):
rm coba-web/backend/coba.db
```

### PostgreSQL (For production)

```bash
# Install PostgreSQL 14+
# Create database:
createdb coba_web

# Update .env:
DATABASE_URL=postgresql://user:password@localhost:5432/coba_web

# Run migrations:
cd coba-web/backend
alembic upgrade head
```

---

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy COBA-Web

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      # Backend tests
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: cd coba-web/backend && pip install -r requirements.txt && pytest

      # Frontend tests
      - uses: actions/setup-node@v3
        with:
          node-version: '20'
      - run: cd coba-web/frontend && npm install && npm run test

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      # Build backend image
      - uses: docker/build-push-action@v4
        with:
          context: coba-web/backend
          push: true
          tags: ghcr.io/${{ github.repository }}/backend:${{ github.sha }}

      # Build frontend image
      - uses: docker/build-push-action@v4
        with:
          context: coba-web/frontend
          push: true
          tags: ghcr.io/${{ github.repository }}/frontend:${{ github.sha }}
```

---

## Monitoring and Logs

### Backend Logs

```bash
# Development (shows live logs)
python -m uvicorn main:app --reload

# Production (use logging service)
# Logs to: /var/log/coba-web/backend.log
tail -f /var/log/coba-web/backend.log
```

### Frontend Logs

```bash
# Browser console (F12 Developer Tools)
# Check Application > Logs for client-side errors

# Production logs via vercel/netlify dashboard
```

### Health Checks

```bash
# Backend health
curl http://localhost:8000/api/health

# Response: {"status": "ok"}
```

---

## Troubleshooting

### Issue: "Cannot connect to backend"

**Solution:**
1. Verify backend is running: `curl http://localhost:8000/api/health`
2. Check CORS origins in `.env`
3. Verify `NEXT_PUBLIC_API_URL` in frontend `.env.local`

### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python -m uvicorn main:app --port 8001
```

### Issue: "Module not found" (Python)

**Solution:**
```bash
# Ensure virtual environment is activated
source coba-web/backend/venv/bin/activate
pip install -r requirements.txt
```

### Issue: "npm install fails"

**Solution:**
```bash
cd coba-web/frontend
rm -rf node_modules package-lock.json
npm install
```

---

## Production Deployment Checklist

- [ ] Set `DEBUG=false` in backend
- [ ] Set `NODE_ENV=production` in frontend
- [ ] Configure PostgreSQL database
- [ ] Set up HTTPS/SSL certificates
- [ ] Configure CORS for production domain
- [ ] Set up CI/CD pipeline
- [ ] Enable monitoring and alerting
- [ ] Set up log aggregation (ELK, Datadog, etc.)
- [ ] Configure session cleanup
- [ ] Set up backups for database
- [ ] Load test before going live
- [ ] Plan rollback procedure

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review GitHub Issues: https://github.com/dmoreq/coba
3. Join discussions: https://github.com/dmoreq/coba/discussions
