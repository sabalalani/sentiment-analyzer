import gradio as gr
from transformers import pipeline

# Load the sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")


def analyze_sentiment(text):
    """Analyze sentiment of the input text"""
    if not text or not text.strip():
        return "❌ Please enter some text to analyze."

    if len(text) > 1000:
        return "❌ Text too long. Please keep it under 1000 characters."

    try:
        # Analyze sentiment
        results = sentiment_pipeline(text)
        result = results[0]

        sentiment = result['label']
        confidence = result['score'] * 100

        # Create output based on sentiment
        if sentiment == "POSITIVE":
            emoji = "😊"
        elif sentiment == "NEGATIVE":
            emoji = "😞"
        else:
            emoji = "😐"

        output = f"""
        {emoji} **Sentiment: {sentiment}**

        📊 **Confidence: {confidence:.1f}%**

        ---
        *Analyzed using AI model*
        """

        return output

    except Exception as e:
        return f"❌ Error analyzing sentiment: {str(e)}"


# Create the Gradio interface
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        label="📝 Enter Text to Analyze",
        placeholder="Type your text here...",
        lines=3,
        info="Maximum 1000 characters"
    ),
    outputs=gr.Markdown(label="📊 Sentiment Analysis Result"),
    title="🧠 AI Sentiment Analyzer",
    description="Analyze emotions and sentiments in text using advanced AI models",
    examples=[
        ["I absolutely love this product! It's amazing!"],
        ["This is the worst service I've ever experienced."],
        ["The package arrived on time and works as expected."]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()