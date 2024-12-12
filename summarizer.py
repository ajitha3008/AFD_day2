import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize the text summarizer with a pre-trained model.
        
        Args:
            model_name (str): Hugging Face model for summarization.
                              Default is BART large CNN model.
        """
        try:
            # Load the tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Create summarization pipeline
            self.summarizer = pipeline(
                "summarization", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
        except Exception as e:
            print(f"Error initializing summarizer: {e}")
            raise

    def summarize(self, 
                  text, 
                  max_length=150, 
                  min_length=50, 
                  do_sample=False):
        """
        Generate a summary of the input text.
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum length of summary
            min_length (int): Minimum length of summary
            do_sample (bool): Whether to use sampling for summary generation
        
        Returns:
            str: Generated summary
        """
        # Validate input
        if not text or len(text.strip()) == 0:
            return "No text provided for summarization."
        
        # Check text length
        if len(text) < 100:
            return "Text is too short to summarize effectively."
        
        try:
            # Generate summary
            summary = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=do_sample
            )[0]['summary_text']
            
            return summary
        
        except Exception as e:
            return f"Error during summarization: {e}"

    def batch_summarize(self, texts, **kwargs):
        """
        Summarize multiple texts in a batch.
        
        Args:
            texts (list): List of texts to summarize
            **kwargs: Additional arguments for summarization
        
        Returns:
            list: List of summaries
        """
        return [self.summarize(text, **kwargs) for text in texts]

def main():
    # Example usage
    summarizer = TextSummarizer()
    
    # Sample long text
    sample_text = """
    Artificial Intelligence (AI) is transforming nearly every aspect of modern society. 
    From healthcare and finance to transportation and entertainment, AI technologies 
    are revolutionizing how we work, live, and interact. Machine learning algorithms 
    can now detect diseases with remarkable accuracy, predict market trends, 
    drive autonomous vehicles, and even create art and music. 
    
    The rapid advancement of AI raises both exciting possibilities and critical ethical 
    considerations. While AI promises unprecedented efficiency and innovation, 
    it also introduces complex challenges related to privacy, job displacement, 
    and the potential for algorithmic bias. Researchers and policymakers are 
    increasingly focused on developing frameworks that ensure AI technologies 
    are developed and deployed responsibly, with a focus on transparency, 
    fairness, and human-centric design.
    """
    
    # Generate summary
    summary = summarizer.summarize(sample_text)
    print("Original Text:", sample_text)
    print("Original Text Length:", len(sample_text))
    print("Summary Length:", len(summary))
    print("\nSummary:\n", summary)

if __name__ == "__main__":
    main()
