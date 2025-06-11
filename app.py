from flask import Flask, request, jsonify, render_template_string
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

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Resume Analyzer</title>
        <style>
            body { font-family: Arial; text-align: center; padding: 20px; }
            textarea, input { width: 80%; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Smart Resume Analyzer</h1>
        <form id="analyzeForm">
            <label>Job Description:</label><br>
            <textarea name="job_description" rows="5" required></textarea><br>
            <label>Select Resume(s) (PDF):</label><br>
            <input type="file" name="resumes" multiple accept=".pdf" required><br><br>
            <button type="submit">Analyze</button>
        </form>
        <div id="results" style="margin-top: 20px;"></div>

        <script>
            const form = document.getElementById('analyzeForm');
            const resultDiv = document.getElementById('results');

            form.onsubmit = async (e) => {
                e.preventDefault();
                resultDiv.innerHTML = "Analyzing...";

                const formData = new FormData(form);

                try {
                    const res = await fetch("/analyze", {
                        method: "POST",
                        body: formData
                    });
                    const data = await res.json();

                    if (data.error) {
                        resultDiv.innerHTML = "<p style='color:red;'>Error: " + data.error + "</p>";
                    } else {
                        let html = "<h3>Analysis Results:</h3><ul>";
                        data.results.forEach(item => {
                            html += `<li><strong>${item.filename}</strong>: ${item.match_score}% match</li>`;
                        });
                        html += "</ul>";
                        resultDiv.innerHTML = html;
                    }
                } catch (err) {
                    resultDiv.innerHTML = "<p style='color:red;'>Request failed</p>";
                }
            };
        </script>
    </body>
    </html>
    """)

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
