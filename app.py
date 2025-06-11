from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_md")


def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content
    return text.strip()

def analyze_resume(resume_text, job_desc):
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_desc)

    resume_clean = " ".join([token.lemma_ for token in resume_doc if not token.is_stop])
    job_clean = " ".join([token.lemma_ for token in job_doc if not token.is_stop])

    docs = [resume_clean, job_clean]
    tfidf = TfidfVectorizer().fit_transform(docs)
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        files = request.files.getlist('resumes')
        job_desc = request.form['job_description']

        results = []
        for file in files:
            resume_text = extract_text_from_pdf(file)
            score = analyze_resume(resume_text, job_desc)
            results.append({
                'filename': file.filename,
                'match_score': score
            })

        return jsonify({
            'message': 'Resumes analyzed successfully.',
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
