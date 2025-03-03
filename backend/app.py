from flask import Flask, request, jsonify
from flask_cors import CORS
from qa_chain import QAChain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize QA chain
print("Initializing QA chain...")
qa_chain = QAChain()
print("QA chain initialized!")

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        # Get answer from QA chain
        result = qa_chain.answer(question)
        
        return jsonify({
            'answer': result['answer'],
            'sources': [
                {
                    'source': src['metadata']['source'].upper(),
                    'content': src['content'][:200] + '...'
                }
                for src in result['sources']
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Changed port to 5001 