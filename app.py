from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
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
        # Using a lightweight model for faster loading
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name
        )
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
    description="Real-time sentiment analysis for text",
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
    model_name: str = "Roberta-Twitter"


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
        logger.info(f"📝 Analyzing sentiment for: {len(request.text)} chars")

        if sentiment_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        input_text = request.text.strip()

        if not input_text:
            raise HTTPException(status_code=400, detail="No text provided")

        if len(input_text) > 1000:
            raise HTTPException(status_code=400, detail="Text too long. Maximum 1000 characters.")

        # Analyze sentiment
        results = sentiment_pipeline(input_text)
        result = results[0]

        # Map labels to more user-friendly terms
        label_mapping = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }

        sentiment_label = label_mapping.get(result['label'], result['label'])
        confidence = round(result['score'] * 100, 2)

        logger.info(f"✅ Sentiment result: {sentiment_label} ({confidence}%)")

        return SentimentResponse(
            sentiment=sentiment_label,
            confidence=confidence,
            label=result['label']
        )

    except Exception as e:
        logger.error(f"❌ Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)