import gradio as gr
from transformers import pipeline
import time


# Load the sentiment analysis model
def load_model():
    """Load the sentiment analysis model"""
    print("Loading sentiment analysis model...")
    model = pipeline("sentiment-analysis")
    print("✅ Model loaded successfully!")
    return model


# Initialize the model
sentiment_pipeline = load_model()


def analyze_sentiment(text):
    """Analyze sentiment of the input text"""
    if not text or not text.strip():
        return {
            "sentiment": "N/A",
            "confidence": 0,
            "output": "❌ Please enter some text to analyze."
        }

    if len(text) > 1000:
        return {
            "sentiment": "N/A",
            "confidence": 0,
            "output": "❌ Text too long. Please keep it under 1000 characters."
        }

    try:
        # Analyze sentiment
        results = sentiment_pipeline(text)
        result = results[0]

        sentiment = result['label']
        confidence = result['score'] * 100

        # Create output based on sentiment
        if sentiment == "POSITIVE":
            emoji = "😊"
            color = "green"
        elif sentiment == "NEGATIVE":
            emoji = "😞"
            color = "red"
        else:
            emoji = "😐"
            color = "yellow"

        output = f"""
        {emoji} **Sentiment: {sentiment}**

        📊 **Confidence: {confidence:.1f}%**

        ---
        *Analyzed using RoBERTa model*
        """

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "output": output
        }

    except Exception as e:
        return {
            "sentiment": "ERROR",
            "confidence": 0,
            "output": f"❌ Error analyzing sentiment: {str(e)}"
        }


# Create the Gradio interface
with gr.Blocks(
        theme=gr.themes.Soft(),
        title="AI Sentiment Analyzer",
        css="""
    .positive { border-left: 5px solid #28a745; padding: 10px; background: #f8fff9; }
    .negative { border-left: 5px solid #dc3545; padding: 10px; background: #fff8f8; }
    .neutral { border-left: 5px solid #ffc107; padding: 10px; background: #fffef8; }
    """
) as demo:
    gr.Markdown(
        """
        # 🧠 AI Sentiment Analyzer
        **Analyze emotions and sentiments in text using advanced AI models**

        Enter any text below to get instant sentiment analysis!
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="📝 Input Text",
                placeholder="Enter your text here (reviews, comments, tweets, etc.)...",
                lines=4,
                max_lines=6,
                info="Maximum 1000 characters"
            )

            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        with gr.Column(scale=1):
            sentiment_output = gr.Markdown(
                label="📊 Analysis Results",
                value="*Results will appear here after analysis...*"
            )

            # Hidden outputs for data
            sentiment_label = gr.Textbox(visible=False)
            confidence_score = gr.Number(visible=False)

    gr.Markdown("---")

    gr.Markdown("### 💡 Example Texts to Try")

    with gr.Row():
        ex_positive = gr.Examples(
            examples=[
                ["I absolutely love this product! It has exceeded all my expectations and the quality is outstanding."]],
            inputs=text_input,
            label="Positive Example",
            examples_per_page=1
        )
        ex_negative = gr.Examples(
            examples=[
                ["This is the worst service I've ever experienced. The product broke immediately and customer support was terrible."]],
            inputs=text_input,
            label="Negative Example",
            examples_per_page=1
        )
        ex_neutral = gr.Examples(
            examples=[["The package arrived on time. The product seems to work as described in the specifications."]],
            inputs=text_input,
            label="Neutral Example",
            examples_per_page=1
        )

    # Connect the button
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=[sentiment_label, confidence_score, sentiment_output]
    )


    # Clear button
    def clear_all():
        return "", 0, "*Results will appear here after analysis...*"


    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[sentiment_label, confidence_score, sentiment_output]
    )

    # Also allow Enter key to submit
    text_input.submit(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=[sentiment_label, confidence_score, sentiment_output]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )