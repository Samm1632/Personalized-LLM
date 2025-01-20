# SkyNet: AI-Powered PDF Processing and Query Tool

SkyNet is a robust, AI-driven application that enables users to upload PDFs, process their content into vectorized embeddings, and query the content using natural language questions. Powered by **LangChain**, **ChromaDB**, and **OpenAI's GPT-3.5-turbo**, this app offers an intuitive interface for exploring document knowledge.

---

## **Key Features**
- **PDF Processing**:
  - Upload a PDF, extract text, and split it into manageable chunks for analysis.
  - Use **LangChain** to split content into chunks while preserving context.
  
- **Vector Embedding**:
  - Generate embeddings using OpenAI’s `text-embedding-ada-002` model.
  - Store embeddings persistently in **ChromaDB** for efficient similarity search.

- **Natural Language Querying**:
  - Input questions in plain English to query the processed PDF content.
  - Retrieve the most relevant chunks and generate accurate answers using OpenAI’s GPT-3.5-turbo.

- **Interactive Interface**:
  - Built using **Streamlit** for a smooth, real-time user experience.
  - Provides clear feedback and error handling.
