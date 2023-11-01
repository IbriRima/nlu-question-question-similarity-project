# Question Answering NLU
This project is a question-answering system that leverages natural language processing (NLP) techniques and machine learning models to provide answers to user-provided questions. It combines the power of pre-trained NLP models and sentence embeddings to find the most relevant answers from a pre created dataset based on Tunisian governmental websites.
## How to Use
1. Clone the repository to your local machine.
2. Install the required dependencies and libraries, as specified in the Installation section.
3. Run the script by providing your question and the path to a CSV file containing questions and answers as command-line arguments.
   For example:
<pre>
<code>
```python your_script.py --question "Your question here" --data-file "preprocessed_dataset.csv"
```
</code>
</pre>
   
5. The script will analyze the user question, identify the most relevant questions from the dataset, and generate a meaningful answer based on the provided context.

## Installation
All packages needed in this project may be found at the requirement.txt file you will need simply to execute the following command on your terminal:
Markdown:

<pre>
<code>
```python
  pip install -r requirements.txt
  ```
</code>
</pre>

## Configuration:
You can configure the script by modifying the list of NLP models used, adjusting the maximum number of tokens for answer generation, and specifying other settings within the script.

## Authors
- Ben Salah Jihed
- Ben Ghnia Souhir
- Ibri Rima
