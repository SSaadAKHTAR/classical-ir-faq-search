# Classical IR FAQ Search (Interactive Document Q&A System)

A Python-based interactive Question & Answer system that uses Classical Information Retrieval techniques (TF-IDF, BM25) to retrieve relevant passages from documents and answer user queries.

## Project Members

| Name | Roll Number |
|------|-------------|
| Kinza Fatima | 22SP-032-CS |
| Subul Raza | 22SP-041-CS |
| Rehma Rehan | 22SP-022-CS |
| Saad Akhtar | 22SP-029-CS |

## Project Description

This system implements a "Retrieve-and-Read" architecture from scratch, using **only the Python Standard Library**.

### Key Features:
-   **Document Loading**: Load text from snippets, files, or JSON.
-   **Indexing**: Uses Inverted Index, TF-IDF weights, and BM25 ranking.
-   **Retrieval**: Finds the most relevant sentences/paragraphs for a query.
-   **Reading/Answering**: Extracts precise answers using regex-based pattern matching (Who, What, When, How Many, etc.).
-   **FAQ Generation**: Automatically generates Frequently Asked Questions from key sentences in the text.
-   **Interactive CLI**: Professional, robust command-line interface.

## Prerequisites

-   Python 3.6+
-   No external libraries required.

## Demo Reproducibility Steps

Follow these steps to run the project and verify its functionality:

1.  **Open a terminal** in the project directory.

2.  **Run the application**:
    ```bash
    python main.py
    ```

3.  **Load a Document**:
    -   When prompted, choose **Option 1** to load the sample "Machine Learning" document.
    -   *Or* choose Option 3 to load a custom text file.

4.  **Ask a Question**:
    -   Select **Option 2** from the main menu.
    -   Try questions like:
        -   "What is machine learning?"
        -   "How many types of machine learning are there?"
        -   "Who uses recommendation systems?"
    -   The system will display the answer. You can ask follow-up questions immediately.

5.  **Generate FAQs**:
    -   Select **Option 3** from the main menu.
    -   Enter `5` to generate 5 automatic Q&A pairs from the content.
