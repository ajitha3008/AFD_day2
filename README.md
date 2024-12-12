# Pre requisites:
1) Python
2) 2) pip - command line tool for python

# To install necessary libraries:
1) pip3 install torch
2) pip3 install transformers

Note: Please disconnect from any VPN connections before running the program. Executing the program for first time might take lot of time. But subsequent execution will be quick.  

# Intent:
In this activity, you’ll summarize a big paragraph into a smaller one

# Text Summarization Code Explanation

This Python program uses the **Hugging Face Transformers library** to implement a text summarization tool. Below is a detailed explanation of the program, its structure, and its functionality.

## Dependencies
The program imports the following modules:

- **torch**: For managing hardware device compatibility (GPU or CPU).
- **transformers**: Specifically, it imports:
  - `pipeline`: Simplifies the creation of pre-trained model pipelines for tasks like summarization.
  - `AutoModelForSeq2SeqLM`: Loads pre-trained sequence-to-sequence models (e.g., BART).
  - `AutoTokenizer`: Loads the corresponding tokenizer for the chosen model.

## Class: `TextSummarizer`
This class encapsulates the text summarization functionality.

### 1. **Initialization (`__init__` method)**

#### Description:
The constructor initializes the summarizer by:
- Loading a pre-trained model (`facebook/bart-large-cnn`) for summarization.
- Creating a summarization pipeline using the loaded model and tokenizer.

#### Parameters:
- `model_name`: The Hugging Face model name. Default is `facebook/bart-large-cnn`, which is a large-scale pre-trained BART model for text summarization.

#### Implementation:
1. **Model and Tokenizer Loading**:
   - The `AutoTokenizer` and `AutoModelForSeq2SeqLM` are used to load the pre-trained model and tokenizer.
   - If GPU is available, the pipeline is set to run on the GPU (`device=0`). Otherwise, it runs on the CPU (`device=-1`).

2. **Error Handling**:
   - If the initialization fails, the program raises an exception with a relevant error message.

---
### 2. **`summarize` Method**

#### Description:
Generates a summary for the input text using the summarization pipeline.

#### Parameters:
- `text` (str): The input text to summarize.
- `max_length` (int): The maximum length of the generated summary (default: 150).
- `min_length` (int): The minimum length of the generated summary (default: 50).
- `do_sample` (bool): Whether to use sampling for summary generation (default: `False`).

#### Steps:
1. **Input Validation**:
   - Checks if the input text is empty or too short (less than 100 characters).
   - Returns an appropriate error message for invalid input.

2. **Summary Generation**:
   - Uses the summarization pipeline to generate a summary based on the specified parameters.
   - Handles exceptions and returns a relevant error message if summarization fails.

#### Output:
- Returns the generated summary as a string.

---
### 3. **`batch_summarize` Method**

#### Description:
Summarizes multiple texts in a batch.

#### Parameters:
- `texts` (list): A list of input texts to summarize.
- `**kwargs`: Additional keyword arguments passed to the `summarize` method (e.g., `max_length`, `min_length`, `do_sample`).

#### Implementation:
- Uses list comprehension to call the `summarize` method for each text in the input list.
- Returns a list of generated summaries.

---
## Main Functionality
The program demonstrates the usage of the `TextSummarizer` class in the `main` function:

### Steps in `main`:
1. **Instance Creation**:
   - An instance of `TextSummarizer` is created.

2. **Sample Text**:
   - A sample text discussing AI's impact on modern society is provided.

3. **Summarization**:
   - The `summarize` method is called with the sample text.

4. **Output**:
   - Prints the original text, its length, the summary, and the summary length.

---
## Code Execution
To execute the program, the `main` function is invoked if the script is run as the main module (`if __name__ == "__main__":`).

---
## Key Features of the Program
1. **Pre-Trained Model Usage**:
   - Leverages a state-of-the-art pre-trained model (`facebook/bart-large-cnn`) for high-quality summarization.
2. **Device Compatibility**:
   - Automatically detects and utilizes GPU if available, otherwise defaults to CPU.
3. **Robust Error Handling**:
   - Ensures that initialization and summarization processes handle exceptions gracefully.
4. **Flexibility**:
   - Supports customizable summarization parameters and batch processing of texts.

---
## Example Output
### Input:

```text
Artificial Intelligence (AI) is transforming nearly every aspect of modern society. From healthcare and finance to transportation and entertainment, AI technologies are revolutionizing how we work, live, and interact. Machine learning algorithms can now detect diseases with remarkable accuracy, predict market trends, drive autonomous vehicles, and even create art and music.

The rapid advancement of AI raises both exciting possibilities and critical ethical considerations. While AI promises unprecedented efficiency and innovation, it also introduces complex challenges related to privacy, job displacement, and the potential for algorithmic bias. Researchers and policymakers are increasingly focused on developing frameworks that ensure AI technologies are developed and deployed responsibly, with a focus on transparency, fairness, and human-centric design.
```

### Output:

- **Summary:**

```text
Artificial Intelligence (AI) is revolutionizing various fields like healthcare, finance, and entertainment by enabling remarkable innovations such as disease detection and autonomous vehicles. However, these advancements come with challenges like privacy concerns, job displacement, and ethical issues. Efforts are being made to ensure responsible AI development focusing on transparency, fairness, and human-centric designs.
```

