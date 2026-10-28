import gradio as gr
import torch
from transformers import (
    BertForSequenceClassification, 
    BertTokenizer,
    CLIPProcessor, 
    CLIPModel,
    pipeline
)
from PIL import Image
import pytesseract
import numpy as np
import cv2

class BertClipMemeAnalyzer:
    def __init__(self):
        # BERT for sentiment analysis
        self.bert_sentiment = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
        # CLIP for zero-shot image classification
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Define meme categories for CLIP
        self.meme_categories = [
            "a hateful meme",
            "a non-hateful meme", 
            "a funny meme",
            "an offensive meme",
            "a wholesome meme",
            "a sarcastic meme",
            "a political meme",
            "a neutral meme"
        ]
        
    def extract_text_from_image(self, image):
        """Extract text using OCR"""
        try:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            text = pytesseract.image_to_string(enhanced)
            return text.strip()
        except:
            return ""
    
    def classify_with_clip(self, image):
        """Use CLIP to classify meme type"""
        inputs = self.clip_processor(
            text=self.meme_categories,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        results = []
        for i, (category, prob) in enumerate(zip(self.meme_categories, probs[0])):
            results.append({
                'category': category.replace("a ", "").replace(" meme", ""),
                'score': prob.item()
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def analyze_meme(self, text_input, image):
        output = "## 🎭 Meme Analysis Results\n\n"
        
        extracted_text = ""
        if image is not None:
            extracted_text = self.extract_text_from_image(image)
            if extracted_text:
                output += f"### 📝 Extracted Text (OCR):\n`{extracted_text}`\n\n"
        
        full_text = text_input or ""
        if extracted_text:
            full_text = f"{full_text} {extracted_text}".strip()
        
        if full_text:
            output += "### 🧠 BERT Sentiment Analysis:\n"
            sentiment_results = self.bert_sentiment(full_text)
            
            label_map = {
                '1 star': 'Very Negative',
                '2 stars': 'Negative',
                '3 stars': 'Neutral',
                '4 stars': 'Positive',
                '5 stars': 'Very Positive',
                'POSITIVE': 'Positive',
                'NEGATIVE': 'Negative',
                'NEUTRAL': 'Neutral'
            }
            
            sentiment = sentiment_results[0]
            mapped_label = label_map.get(sentiment['label'], sentiment['label'])
            output += f"**{mapped_label}** (Confidence: {sentiment['score']:.1%})\n\n"
        else:
            output += "### 🧠 BERT Sentiment Analysis:\n"
            output += "No text provided for sentiment analysis\n\n"
        
        if image is not None:
            output += "### 🖼️ CLIP Meme Classification:\n"
            clip_results = self.classify_with_clip(image)
            
            top = clip_results[0]
            output += f"**Primary Classification: {top['category'].title()}** ({top['score']:.1%})\n\n"
            
            hateful_score = next(r['score'] for r in clip_results if 'hateful' in r['category'])
            non_hateful_score = next(r['score'] for r in clip_results if 'non-hateful' in r['category'])
            
            if hateful_score > non_hateful_score:
                output += f"⚠️ **Hate Detection: Potentially Hateful** ({hateful_score:.1%})\n\n"
            else:
                output += f"✅ **Hate Detection: Non-hateful** ({non_hateful_score:.1%})\n\n"
            
            output += "**All Classifications:**\n"
            for result in clip_results:
                bar = "█" * int(result['score'] * 20)
                output += f"- {result['category'].title()}: {bar} {result['score']:.1%}\n"
        else:
            output += "### 🖼️ CLIP Meme Classification:\n"
            output += "No image provided for classification\n\n"
        
        output += "\n### 📊 Summary:\n"
        if full_text:
            output += f"- Text analyzed: {len(full_text.split())} words\n"
        if image:
            output += f"- Image analyzed: ✓\n"
        if extracted_text:
            output += f"- OCR successful: ✓\n"
        
        return output

if __name__ == "__main__":
    analyzer = BertClipMemeAnalyzer()
    demo = gr.Interface(
        fn=analyzer.analyze_meme,
        inputs=[
            gr.Textbox(
                label="📝 Text Input (Optional)",
                placeholder="Enter meme text or leave empty for OCR...",
                lines=2
            ),
            gr.Image(
                label="🖼️ Upload Meme",
                type="pil"
            )
        ],
        outputs=gr.Markdown(label="Analysis Results"),
        title="🎭 BERT + CLIP Meme Analyzer",
        description="Baseline Zero-Shot Analyzer using BERT + CLIP",
        theme=gr.themes.Soft()
    )
    demo.launch()
