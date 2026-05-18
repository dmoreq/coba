# Deployment Guide

This guide covers running COBA Web locally for development and deploying to production.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Production Deployment](#production-deployment)
3. [Environment Configuration](#environment-configuration)
4. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- **Node.js** 20+ (`node --version`)
- **Python** 3.10+ (`python3 --version`)
- **Git**

### Backend Setup

The backend is a FastAPI service (Python).

```bash
cd web/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start development server (auto-reload)
python3 -m uvicorn app.main:app --reload --port 8000
```

Server runs at: **http://localhost:8000**
Swagger API docs: **http://localhost:8000/docs**

### Frontend Setup

The frontend is a Next.js 16 app (TypeScript + React 19).

```bash
cd web/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Server runs at: **http://localhost:3000**

### Full Stack (Recommended)

Run both in parallel (separate terminals):

```bash
# Terminal 1: Backend
cd web/backend
source venv/bin/activate
python3 -m uvicorn app.main:app --reload

# Terminal 2: Frontend
cd web/frontend
npm run dev
```

Then open **http://localhost:3000** and start learning.

---

## Running Tests

### Backend Tests

```bash
cd web/backend

# Run all tests
pytest

# Run with coverage (target: 90%)
pytest --cov=app --cov-report=term-missing

# Run single file
pytest tests/unit/test_models.py -v
```

All tests must pass and maintain 90% coverage.

### Frontend Tests

```bash
cd web/frontend

# Run unit tests (Vitest)
npm test

# Run in watch mode
npm run test:watch

# Run with coverage
npm test -- --coverage

# Run integration tests
npm run test:integration
```

Target: 80%+ coverage on critical paths.

---

## Production Deployment

### Option 1: Frontend on Vercel (Recommended)

Vercel is the easiest option for Next.js apps.

1. **Push to GitHub:**
   ```bash
   git push origin main
   ```

2. **Deploy on Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repo
   - Set `Root Directory` to `web/frontend`
   - Click "Deploy"

3. **Configure environment variables:**
   - Set `NEXT_PUBLIC_API_URL` to your backend URL (see Option 1b below)

Vercel will auto-deploy on every push to `main`.

### Option 1b: Backend on Railway / Render / Fly.io

Pick one:

#### Railway (Easiest)
1. Go to [railway.app](https://railway.app)
2. Create new project → GitHub repo
3. Select `web/backend` directory
4. Set environment: `PORT=8000`
5. Deploy

Get the backend URL (e.g., `https://my-app.railway.app`), then set in Vercel:
- Variable: `NEXT_PUBLIC_API_URL`
- Value: `https://my-app.railway.app`

#### Render
1. Go to [render.com](https://render.com)
2. New Web Service → GitHub repo
3. Set `Root Directory` to `web/backend`
4. Runtime: Python 3.10
5. Build command: `pip install -r requirements.txt`
6. Start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
7. Deploy

#### Fly.io
1. Install `flyctl` CLI
2. Run: `flyctl launch` from `web/backend/`
3. Follow prompts to generate `Dockerfile`
4. Deploy: `flyctl deploy`

### Option 2: Self-Hosted on VPS (Linux)

For full control, deploy on a VPS (AWS EC2, DigitalOcean, Linode, etc.).

#### Backend (Systemd Service)

1. **SSH into your server:**
   ```bash
   ssh ubuntu@your-server-ip
   ```

2. **Clone repo and setup:**
   ```bash
   git clone https://github.com/your-username/coba.git
   cd coba/web/backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Create systemd service file:**
   ```bash
   sudo tee /etc/systemd/system/coba-backend.service > /dev/null <<EOF
   [Unit]
   Description=COBA Web Backend
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/coba/web/backend
   ExecStart=/home/ubuntu/coba/web/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   EOF
   ```

4. **Start service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable coba-backend
   sudo systemctl start coba-backend
   ```

5. **Check status:**
   ```bash
   sudo systemctl status coba-backend
   ```

#### Frontend (Static Export + Nginx)

1. **Build Next.js app:**
   ```bash
   cd coba/web/frontend
   npm install
   npm run build
   ```

2. **Install Nginx:**
   ```bash
   sudo apt-get update
   sudo apt-get install nginx
   ```

3. **Configure Nginx:**
   ```bash
   sudo tee /etc/nginx/sites-available/coba > /dev/null <<EOF
   server {
       listen 80;
       server_name your-domain.com;

       root /home/ubuntu/coba/web/frontend/.next/static;

       location / {
           try_files \$uri \$uri/ /404.html;
       }

       location /api/ {
           proxy_pass http://localhost:8000;
           proxy_http_version 1.1;
       }
   }
   EOF
   ```

4. **Enable and start:**
   ```bash
   sudo ln -s /etc/nginx/sites-available/coba /etc/nginx/sites-enabled/
   sudo systemctl restart nginx
   ```

5. **SSL (Let's Encrypt):**
   ```bash
   sudo apt-get install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

---

## Environment Configuration

### Backend (`web/backend/.env` — optional, all have defaults)

```bash
# API binding
API_HOST=0.0.0.0
API_PORT=8000

# CORS origins (comma-separated)
# If not set, defaults to http://localhost:3000
CORS_ORIGINS=http://localhost:3000,https://your-domain.com

# Session cleanup (seconds)
SESSION_CLEANUP_INTERVAL=1800      # 30 minutes
SESSION_TIMEOUT=86400              # 24 hours

# Logging
LOG_LEVEL=INFO                      # INFO | DEBUG | WARNING
```

### Frontend (`web/frontend/.env.local` — optional, only NEXT_PUBLIC_API_URL needed for prod)

```bash
# Backend API endpoint (required for production)
NEXT_PUBLIC_API_URL=https://your-api-url.com

# Analytics (optional)
NEXT_PUBLIC_GOOGLE_ANALYTICS_ID=
```

---

## Monitoring

### Backend Health Check

```bash
curl http://localhost:8000/api/health
# Returns: {"status": "ok"}
```

Use this endpoint to monitor uptime.

### Logs

**Development:**
```bash
# Backend logs appear in terminal where uvicorn was started
# Frontend logs appear in terminal where npm run dev was started
```

**Production (Systemd):**
```bash
# View recent logs
sudo journalctl -u coba-backend -n 50

# Follow live logs
sudo journalctl -u coba-backend -f

# Full logs with timestamps
sudo journalctl -u coba-backend --no-pager | head -100
```

**Browser Console Errors:**
- Open DevTools (F12)
- Check Console tab for client-side errors
- Check Network tab to debug API calls

---

## Database (None — In-Memory Sessions)

⚠️ **Important:** The current version has **no persistent database**. All bandit sessions live in server memory and are lost on restart.

### For Production, You Would Add:

To persist sessions across restarts, extend `BanditSessionService`:

1. Add SQLAlchemy models (`app/models/db.py`)
2. Initialize PostgreSQL or SQLite
3. Replace `self._sessions: dict` with database queries
4. Add database migration management (Alembic)

**Not needed for v1.0**, but planned for v2.0 when user accounts are added.

---

## Troubleshooting

### "Cannot connect to backend"

1. **Is backend running?**
   ```bash
   curl http://localhost:8000/api/health
   ```
   Should return `{"status": "ok"}`

2. **Check CORS origins in backend:**
   ```bash
   # Backend .env should include frontend URL
   CORS_ORIGINS=http://localhost:3000
   ```

3. **Check frontend environment:**
   ```bash
   # web/frontend/.env.local
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

4. **Restart both servers** (clear caches)

### "Port 8000 already in use"

```bash
# Find process on port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
uvicorn app.main:app --port 8001
```

### "Module not found" (Python)

```bash
# Ensure venv is activated
source web/backend/venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### "npm install fails"

```bash
cd web/frontend
rm -rf node_modules package-lock.json
npm install
```

### "ModuleNotFoundError: coba"

The backend imports `from coba.bandit` — this is the COBA library. It must be installed:

```bash
cd web/backend
pip install -r requirements.txt
```

If still failing, check that the system `coba` package is available:
```bash
python3 -c "import coba; print(coba.__file__)"
```

### Tests Failing

**Frontend:**
```bash
npm test 2>&1 | head -50  # See first errors
```

**Backend:**
```bash
pytest -v --tb=short    # Verbose with short tracebacks
```

---

## Checklist for Production

- [ ] Backend .env configured with correct CORS origins
- [ ] Frontend .env.local has `NEXT_PUBLIC_API_URL` set to production API
- [ ] All tests passing locally: `npm test` (frontend), `pytest` (backend)
- [ ] Build succeeds: `npm run build`
- [ ] Backend logs show no errors on startup
- [ ] Health check endpoint responds: `curl /api/health`
- [ ] Frontend loads and can create a session
- [ ] At least one lesson can be stepped through
- [ ] Mobile view tested (375px width)
- [ ] Dark mode toggle works
- [ ] Keyboard shortcuts work (Space, ←/→, 1/2/3)

---

## Questions?

1. Check backend logs: `journalctl -u coba-backend -f`
2. Check browser console (F12) for frontend errors
3. Test API manually: `curl http://localhost:8000/docs` (Swagger)
4. See `CONTRIBUTING.md` for code style and development workflow
5. Open a GitHub Issue if deployment is blocked
