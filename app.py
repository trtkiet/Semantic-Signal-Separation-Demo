from flask import Flask, render_template, request, jsonify
from turftopic import SemanticSignalSeparation
import pandas as pd
import numpy as np
import re

app = Flask(__name__)

# Global variable to store the trained model and corpus
# In a production app, this should be stored in a database or session with proper management
current_model = None
current_corpus = None

def preprocess_text(text):
    # Basic preprocessing: lowercase, remove special characters (keep only alphanumeric and spaces)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global current_model, current_corpus
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        text = file.read().decode('utf-8')
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 400
    
    # Split text into documents (one per line as per plan)
    # Preprocess each line
    corpus = [preprocess_text(line.strip()) for line in text.split('\n') if line.strip()]
    # Filter out empty lines after preprocessing
    corpus = [doc for doc in corpus if doc.strip()]
    
    if not corpus:
        return jsonify({'error': 'No valid text found after preprocessing. Please check your input.'}), 400

    if len(corpus) < 10: # Basic check to ensure enough data
        return jsonify({'error': 'Please provide at least 10 documents (lines) for meaningful analysis.'}), 400

    current_corpus = corpus
    
    # try:
    # print(f"Corpus: {corpus}")
    # Initialize and train the model
    # Plan: feature_importance="combined", encoder="all-MiniLM-L6-v2"
    # We set n_components to 10 for the demo, but this could be configurable
    current_model = SemanticSignalSeparation(
        n_components=20, 
        feature_importance="combined", 
        encoder="all-MiniLM-L6-v2",
        max_iter=500
    )
    
    current_model.fit(corpus)
    print(f"Corpus size: {len(corpus)}")
    print("Model trained successfully.")
    # Extract topics
    # We need positive and negative terms.
    # turftopic's get_topics usually returns top words. 
    # For S3, the axes have positive and negative poles.
    # We can access the components_ directly to get top positive and negative.
    
    vocab = current_model.get_vocab()
    components = current_model.components_ # shape: (n_topics, n_vocab)
    print(f"Shape Vocab: {len(vocab)}")
    print(f"Shape Components: {components.shape}")
    #current_model.print_topics(top_k=10)
    
    topics_summary = []
    for i, component in enumerate(components):
        # Create a dataframe for this component
        comp_series = pd.Series(component, index=vocab)
        
        # Top positive
        top_pos = comp_series.nlargest(10).index.tolist()
        # Top negative (smallest values)
        top_neg = comp_series.nsmallest(10).index.tolist()

        topics_summary.append({
            'id': i,
            'name': f"Axis {i}",
            'positive': top_pos,
            'negative': top_neg
        })
        
    return jsonify({
        'message': 'Model trained successfully',
        'topics': topics_summary,
        'n_documents': len(corpus)
    })
        
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    global current_model
    
    if current_model is None:
        return jsonify({'error': 'Model not trained yet'}), 400
        
    data = request.json
    axis_x_id = int(data.get('axis_x', 0))
    axis_y_id = int(data.get('axis_y', 1))
    
    # try:
    vocab = current_model.get_vocab()
    components = current_model.components_
    
    # Get scores for the requested axes
    scores_x = components[axis_x_id]
    scores_y = components[axis_y_id]
    
    # We want to plot words. Sending all words might be too much if the vocab is huge.
    # Let's filter for "significant" words on either axis.
    # Or just send the top N words for these axes combined.
    # For a demo, let's send the top 200 words that have the highest absolute magnitude on either axis.
    
    # Calculate magnitude for filtering
    magnitude = np.abs(scores_x) + np.abs(scores_y)
    
    # Get indices of top 200 words by magnitude
    top_indices = np.argsort(magnitude)[-200:]
    
    plot_data = []
    for idx in top_indices:
        plot_data.append({
            'word': vocab[idx],
            'x': float(scores_x[idx]),
            'y': float(scores_y[idx]),
            'magnitude': float(magnitude[idx])
        })
        
    # Helper to get label
    def get_axis_label(component, vocab):
        comp_series = pd.Series(component, index=vocab)
        top_pos = comp_series.nlargest(1).index[0]
        top_neg = comp_series.nsmallest(1).index[0]
        return f"{top_pos} vs. {top_neg}"

    label_x = get_axis_label(scores_x, vocab)
    label_y = get_axis_label(scores_y, vocab)

    return jsonify({
        'plot_data': plot_data,
        'axis_x_label': label_x,
        'axis_y_label': label_y
    })
    
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
