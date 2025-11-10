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
        return "❌ Please enter some text to analyze."

    if len(text) > 1000:
        return "❌ Text too long. Please keep it under 1000 characters."

    try:
        # Show loading state
        time.sleep(0.5)  # Small delay to show loading

        # Analyze sentiment
        results = sentiment_pipeline(text)
        result = results[0]

        sentiment = result['label']
        confidence = result['score'] * 100

        # Create a nice output format
        if sentiment == "POSITIVE":
            emoji = "😊"
            color = "#28a745"
        elif sentiment == "NEGATIVE":
            emoji = "😞"
            color = "#dc3545"
        else:
            emoji = "😐"
            color = "#ffc107"

        output = f"""
        {emoji} **Sentiment: {sentiment}**

        📊 **Confidence: {confidence:.1f}%**

        ---
        *Analyzed using RoBERTa model*
        """

        return output

    except Exception as e:
        return f"❌ Error analyzing sentiment: {str(e)}"


# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="AI Sentiment Analyzer") as demo:
    gr.Markdown(
        """
        # 🧠 AI Sentiment Analyzer
        **Analyze emotions and sentiments in text using advanced AI models**

        Enter any text below to get instant sentiment analysis!
        """
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="📝 Input Text",
                placeholder="Enter your text here (reviews, comments, tweets, etc.)...",
                lines=4,
                max_lines=6,
                info="Try our examples below or enter your own text!"
            )

            analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")

        with gr.Column():
            output_text = gr.Markdown(
                label="📊 Analysis Results",
                value="*Results will appear here after analysis...*"
            )

    gr.Markdown("---")

    with gr.Row():
        gr.Markdown("### 💡 Example Texts to Try")

    with gr.Row():
        with gr.Column():
            ex_positive = gr.Examples(
                examples=[
                    ["I absolutely love this product! It has exceeded all my expectations and the quality is outstanding."]],
                inputs=text_input,
                label="Positive Example"
            )
        with gr.Column():
            ex_negative = gr.Examples(
                examples=[
                    ["This is the worst service I've ever experienced. The product broke immediately and customer support was terrible."]],
                inputs=text_input,
                label="Negative Example"
            )
        with gr.Column():
            ex_neutral = gr.Examples(
                examples=[
                    ["The package arrived on time. The product seems to work as described in the specifications."]],
                inputs=text_input,
                label="Neutral Example"
            )

    # Connect the button
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=output_text
    )

    # Also allow Enter key to submit
    text_input.submit(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=output_text
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )