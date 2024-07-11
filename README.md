# Research Paper Analysis 🐼

## Overview

This project is designed to analyze research papers using advanced language models and vector embedding techniques. The main objective is to provide users with the ability to query a set of research papers and receive accurate, context-based responses.

## Features

- **Query Research Papers**: Ask questions about research papers and get precise answers based on the context.
- **Document Embedding**: Utilizes vector embeddings for efficient document retrieval.
- **Document Similarity Search**: Find relevant document chunks similar to the queried context.
- **Summarize Research Papers**: Get concise summaries of research papers.

## Technologies Used

- **Language**: Python
- **Libraries**: 
  - Streamlit
  - Langchain
  - dotenv
  - PyPDF2
- **Model Used**: ChatGroq (Llama3-70b-8192)
- **APIs Used**: OpenAI API, GROQ API

## Installation

1. **Clone the repository**:
    \`\`\`bash
    git clone https://github.com/yourusername/research-paper-analysis.git
    cd research-paper-analysis
    \`\`\`

2. **Create and activate a virtual environment**:
    \`\`\`bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use \`env\\Scripts\\activate\`
    \`\`\`

3. **Install the required dependencies**:
    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`

4. **Set up environment variables**:
    - Create a \`.env\` file in the project directory and add your API keys:
    \`\`\`env
    OPENAI_API_KEY=your_openai_api_key
    GROQ_API_KEY=your_groq_api_key
    \`\`\`

## Usage

1. **Run the Streamlit app**:
    \`\`\`bash
    streamlit run app.py
    \`\`\`

2. **Interact with the app**:
    - Upload research papers to the specified directory.
    - Use the provided interface to query, read, and summarize the research papers.

## How It Works

### Vector Embedding and ObjectBox Vectorstore DB

1. **Data Ingestion**: Load research papers using \`PyPDFDirectoryLoader\`.
2. **Text Splitting**: Split documents into manageable chunks using \`RecursiveCharacterTextSplitter\`.
3. **Vector Embedding**: Embed the document chunks using \`OpenAIEmbeddings\`.
4. **Vector Storage**: Store the embeddings in an \`ObjectBox\` vector store for efficient retrieval.

### Query and Retrieval

1. **User Input**: Enter a query related to the research papers.
2. **Retrieval Chain**: Combine document chains and retrieval mechanisms to find relevant document sections.
3. **Response Generation**: Generate answers based on the context using the ChatGroq model.

### Summarize Research Papers

1. **Summarization**: Generate concise summaries of the research papers, highlighting key contributions and findings.

## Screenshots

![Screenshot of App](objectbox\screenshot1.png)
![Screenshot of App](objectbox\screenshot2.png)

## Project Information

**Created by:** Sai Koushik Gandikota  
**LinkedIn:** [https://www.linkedin.com/in/gandikota-sai-koushik/](https://www.linkedin.com/in/gandikota-sai-koushik/)  
**Email:** saikoushik.gsk@gmail.com  

### Technologies Used:

- **Language:** Python
- **Libraries:** Streamlit, Langchain, dotenv, PyPDF2
- **Model Used:** ChatGroq (Llama3-70b-8192)
- **APIs Used:** OpenAI API, GROQ API

### Project Report:

This project is designed to analyze research papers using advanced language models and vector embedding techniques. The main objective is to provide users with the ability to query a set of research papers and receive accurate, context-based responses.

The project leverages the Langchain framework, which facilitates the integration of various language models and embeddings. Specifically, it uses the ChatGroq model (Llama3-70b-8192) to generate responses based on the context of the research papers. The OpenAI API and GROQ API are used to power the language model and embeddings, ensuring high-quality and relevant responses.

The data ingestion process involves loading research papers from a specified directory using the PyPDFDirectoryLoader, followed by splitting the documents into manageable chunks with the RecursiveCharacterTextSplitter. These chunks are then embedded using the OpenAIEmbeddings and stored in an ObjectBox vector store, which enables efficient retrieval of relevant document sections.

Users can interact with the system through a Streamlit interface, where they can input their queries, read documents, and get summaries of the research papers. The system processes the input, retrieves relevant document sections, and generates accurate answers to the users' questions. Additionally, the project includes functionality to summarize the database of research papers, providing a concise overview of each paper's main contributions.

Overall, this project demonstrates the effective use of advanced language models and vector embeddings to enhance research paper analysis and information retrieval.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.