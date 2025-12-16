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
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.predict import AudioPredictor

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
predictor: Optional[AudioPredictor] = None

# Mapeo de índices a idiomas
LANGUAGE_NAMES = {
    0: "español",
    1: "inglés",
    2: "francés",
    3: "alemán"
}

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
        # Buscar modelo
        model_path = Path("models/language_model.keras")
        if not model_path.exists():
            logger.error(f"❌ Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Cargar modelo
        predictor = AudioPredictor(language_model_path=model_path)
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
    "http://localhost:3000,http://localhost:4321,https://epigrafiafrontend-fe01u5o3d-adrianberlinchesayala25s-projects.vercel.app,https://*.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Temporal - permite todos los orígenes
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
        "models_loaded": predictor is not None and predictor.models_loaded
    }

@app.get("/api/models/status")
async def models_status():
    """Get models loading status"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "language_model": "loaded" if predictor.language_model else "not loaded",
        "status": "ready" if predictor.models_loaded else "not ready"
    }

@app.post("/api/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    """
    Analyze audio file and predict language
    
    Args:
        audio: Audio file (wav, mp3, webm, etc.)
    
    Returns:
        JSON with language predictions
    """
    if not predictor or not predictor.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please try again later."
        )
    
    try:
        # Read audio file
        logger.info(f"📥 Receiving audio: {audio.filename} ({audio.content_type})")
        audio_bytes = await audio.read()
        logger.info(f"📊 Audio size: {len(audio_bytes)} bytes")
        
        # Predict
        result = predictor.predict_language(audio_bytes)
        
        # Format response
        language_idx = result['language_prediction']
        language_name = LANGUAGE_NAMES.get(language_idx, f"unknown_{language_idx}")
        
        # Build probabilities dict
        probabilities = {}
        for idx, prob in enumerate(result['language_probabilities']):
            lang_name = LANGUAGE_NAMES.get(idx, f"unknown_{idx}")
            probabilities[lang_name] = float(prob)
        
        formatted_result = {
            "language": {
                "detected": language_name,
                "confidence": float(result['language_confidence']),
                "probabilities": probabilities
            },
            "status": "success"
        }
        
        logger.info(f"✅ Prediction: {language_name} (confidence: {result['language_confidence']:.2%})")
        
        return JSONResponse(content=formatted_result)
        
    except Exception as e:
        logger.error(f"❌ Error processing audio: {e}", exc_info=True)
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
        "description": "API para detección de idioma en audio",
        "supported_languages": list(LANGUAGE_NAMES.values()),
        "endpoints": [
            {
                "path": "/api/analyze",
                "method": "POST",
                "description": "Analyze audio and predict language",
                "parameters": {
                    "audio": "Audio file (multipart/form-data) - WAV, MP3, WebM, etc."
                },
                "response": {
                    "language": {
                        "detected": "string (español, inglés, francés, alemán)",
                        "confidence": "float (0-1)",
                        "probabilities": "object with all language probabilities"
                    },
                    "status": "string"
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
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )