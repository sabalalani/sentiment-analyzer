from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import pipeline
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
sentiment_pipeline = None


def load_model():
    """Load the sentiment analysis model"""
    global sentiment_pipeline
    try:
        logger.info("Loading sentiment analysis model...")
        sentiment_pipeline = pipeline("sentiment-analysis")
        logger.info("✅ Sentiment model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting up sentiment analysis application...")
    load_model()
    yield
    # Shutdown
    logger.info("🛑 Shutting down application...")


# Initialize FastAPI app
app = FastAPI(
    title="AI Sentiment Analyzer",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    label: str


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML"""
    return FileResponse('templates/index.html')


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": sentiment_pipeline is not None}


@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of text"""
    try:
        if sentiment_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        input_text = request.text.strip()

        if not input_text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Analyze sentiment
        results = sentiment_pipeline(input_text)
        result = results[0]

        sentiment_label = result['label']
        confidence = round(result['score'] * 100, 2)

        return SentimentResponse(
            sentiment=sentiment_label,
            confidence=confidence,
            label=result['label']
        )

    except Exception as e:
        logger.error(f"❌ Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)