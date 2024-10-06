# End-to-end-Medical-Chatbot-using-Llama2
# STEPS TO RUN

```bash
conda create -n mchat python=3.8 -y
```

```bash
conda activate mchatbot
```
```bash
python app.py
```

# Medical Chatbot

## Overview
The Medical Chatbot is a custom-built AI-powered conversational agent designed to provide users with accurate medical information and assist in healthcare decision-making. This chatbot uses a Retrieval-Augmented Generation (RAG) approach to retrieve relevant data from a curated medical knowledge base, generating informative and context-aware responses.

## Features
- **Medical Information**: Provides insights into various medical conditions, symptoms, and treatments.
- **Medication Guidance**: Suggests medications and treatments for common ailments.
- **Symptom Checker**: Helps users identify potential health issues based on their symptoms.
- **Custom RAG Implementation**: Developed a unique RAG system to ensure contextually relevant responses based on user queries.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Flask (for web framework)
  - LangChain (for implementing RAG)
  - Pinecone (for vector storage and retrieval)
  - Hugging Face Transformers (for pre-trained language models)
  - NLTK / SpaCy (for natural language processing)

## Implementation
The following steps outline the creation and setup process:

1. **Setup**:
   - Install Python and required libraries.

2. **Create the Knowledge Base**:
   - Collect the dataset
   - Gather medical data and resources relevant to user queries.
   - Store the data in a vector database  Pinecone for efficient retrieval.

3. **Develop the RAG System**:
   - Implement the Retrieval-Augmented Generation approach to fetch relevant data from the knowledge base based on user inputs and generate responses.

4. **Run the Application**:
   ```bash
   python app.py


5  **📍 Workflow Breakdown**:
User Query: Users input health-related questions.

Llama 2 - Language Model: The query is processed by Llama 2, generating context-aware medical responses using its vast knowledge.

Pinecone - Knowledge Retrieval: For more specific information, Pinecone efficiently searches indexed medical data for relevant insights.

Response Delivery: The chatbot delivers an accurate, comprehensive answer to the user’s query.'

Continuous Learning: The chatbot improves its responses over time, ensuring it provides up-to-date medical information.
