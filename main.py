"""
EpigrafIA Backend - FastAPI Server
=====================================
API para detección de idioma usando Deep Learning

Endpoints:
- POST /api/analyze - Analiza audio y devuelve predicciones
- GET /api/health - Estado del servidor
- GET /api/models/status - Estado de los modelos cargados
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.predict import Predictor

# ========================================
# Configuración de logging
# ========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# Configuración de entorno
# ========================================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Fuerza TensorFlow a usar solo CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce logs de TensorFlow

# ========================================
# Variables globales
# ========================================
predictor: Optional[Predictor] = None

# ========================================
# Lifecycle events
# ========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events"""
    global predictor
    
    # Startup
    logger.info("🚀 Starting EpigrafIA API...")
    try:
        predictor = Predictor()
        logger.info("✅ Models loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 EpigrafIA API shutting down...")
    if predictor:
        predictor.cleanup()

# ========================================
# FastAPI App
# ========================================
app = FastAPI(
    title="EpigrafIA API",
    version="1.0.0",
    lifespan=lifespan
)

# ========================================
# CORS Configuration
# ========================================
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:4321"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Routes
# ========================================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "EpigrafIA API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "POST /api/analyze",
            "health": "GET /api/health",
            "models": "GET /api/models/status",
            "docs": "GET /api/docs"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": predictor is not None
    }

@app.get("/api/models/status")
async def models_status():
    """Get models loading status"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "language_model": "loaded",
        "image_model": "loaded",
        "status": "ready"
    }

@app.post("/api/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    """
    Analyze audio file and predict language
    
    Args:
        audio: Audio file (wav, mp3, etc.)
    
    Returns:
        JSON with language predictions
    """
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please try again later."
        )
    
    try:
        # Read audio file
        audio_bytes = await audio.read()
        
        # Predict
        result = predictor.predict(audio_bytes)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

@app.get("/api/docs")
async def api_docs():
    """API documentation"""
    return {
        "title": "EpigrafIA API Documentation",
        "version": "1.0.0",
        "endpoints": [
            {
                "path": "/api/analyze",
                "method": "POST",
                "description": "Analyze audio and predict language",
                "parameters": {
                    "audio": "Audio file (multipart/form-data)"
                },
                "response": {
                    "language": {
                        "detected": "string",
                        "confidence": "float",
                        "probabilities": "object"
                    }
                }
            },
            {
                "path": "/api/health",
                "method": "GET",
                "description": "Check API health status"
            },
            {
                "path": "/api/models/status",
                "method": "GET",
                "description": "Check models loading status"
            }
        ]
    }

# ========================================
# Error handlers
# ========================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )