<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology">
  </a>
</p>

<h1 align="center"><b>Natural Language Processing</b></h>

## GROUP MEMBERS
| #   | Student ID| Name              | Email                  |
|-----|-----------|-------------------|------------------------|
| 1   | 23520822  | Tran Tuan Kiet   | 23520774@gm.uit.edu.vn |
| 2   | 23520872  | Nguyen Thang Loi  | 23520872@gm.uit.edu.vn |
| 3   | 23521527  | Nguyen My Thong   | 23521527@gm.uit.edu.vn |

## COURSE INTRODUCTION
* **Course Name:** Natural Language Processing
* **Course ID:** CS221
* **Class ID:** CS221.Q12
* **Year:** 2025 - 2026
* **Lecturer**: Ph.D Nguyen Thi Quy 

# S³: Semantic Signal Separation Demo

This project is a Flask-based web application that demonstrates the **Semantic Signal Separation (S³)** method for text analysis, as implemented in the [turftopic](https://github.com/x-tabdeveloping/turftopic) library.

The application allows users to input a text corpus, train an S³ model to discover "Semantic Axes" (topics with opposing poles), and visualize the vocabulary on a "Concept Compass" scatter plot.

## Features

*   **Corpus Entry**: Upload a text file (`.txt`) containing documents.
*   **Text Preprocessing**: Automatic lowercasing and removal of special characters.
*   **S³ Model Training**: Uses `SemanticSignalSeparation` with `all-MiniLM-L6-v2` encoder and "combined" feature importance.
*   **Semantic Axes Display**: Shows discovered topics with their top positive and negative terms (e.g., "Deep Learning vs. Algorithm").
*   **Concept Compass**: Interactive 2D scatter plot (using Plotly.js) to visualize how words project onto selected semantic axes.

## Prerequisites

*   Python 3.8 or higher
*   pip (Python package installer)

## Installation

1.  Clone the repository or download the source code.
2.  Navigate to the project directory:
    ```bash
    cd path/to/project
    ```
3.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```
4.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Start the Flask application:
    ```bash
    python app.py
    ```
2.  Open your web browser and go to:
    ```
    http://127.0.0.1:5000
    ```
3.  **Upload your text corpus** (a `.txt` file) using the file input. Ensure the file has at least 10 lines of text for meaningful results.
4.  Click **"Train S³ Model"**.
    *   *Note: The first run might take a moment to download the embedding model.*
5.  Once trained, explore the **Semantic Axes** in the sidebar.
6.  Use the **Concept Compass** to visualize the relationship between words by selecting different axes for the X and Y dimensions.

## Technologies Used

*   **Backend**: Flask, Python
*   **NLP/Modeling**: [turftopic](https://github.com/x-tabdeveloping/turftopic), scikit-learn, sentence-transformers
*   **Frontend**: HTML, Bootstrap 5, Plotly.js
*   **Data Handling**: pandas, numpy

## License

[MIT](LICENSE)
