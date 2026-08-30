# AI Study Buddy (Streamlit + Groq)

A Streamlit app that uses the free Groq API to:

- Explain your notes, or any topic, at a level you pick (Like I'm 10 / high school / university)
- Generate a practice quiz that adapts to your weak topics and past scores, and explains the specific misconception behind each wrong answer — play it Kahoot-style (one question at a time, 20-second timer, speed-based scoring) or classic (all questions on one page)
- Build a revision plan that emphasizes topics you've struggled with
- Act as a doubt-solving chatbot — with a Socratic mode that guides you to the answer instead of just telling you
- Generate flip-through flashcards
- **Feynman mode** — you explain a concept in your own words, and the AI scores your understanding, points out what's missing, and flags misconceptions
- **Interactive concept map** — a draggable node diagram of how the ideas in your notes connect
- Track your progress: study streak, quiz score history, and topics to review
- Extract key topics using TF-IDF (scikit-learn) — this part is local ML and makes no API call

Notes can be pasted directly or uploaded as a PDF, DOCX, or TXT file.

## Setup

1. Get a free Groq API key: https://console.groq.com → sign up → API Keys → Create key

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your key so you don't have to paste it every time — copy `.env.example` to `.env` and put your real key inside:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   ```
   GROQ_API_KEY=your_actual_key_here
   ```
   `.env` is already in `.gitignore`, so it never gets committed if you push this to GitHub.

4. Run the app:
   ```bash
   streamlit run app.py
   ```

5. Open `http://localhost:8501` in your browser. The key loads automatically from `.env` — the sidebar field is only needed if you skip step 3.

## Files

- `app.py` — the Streamlit UI and all tab logic
- `utils.py` — Groq API calls, file text extraction, TF-IDF keyword extraction
- `db.py` — local SQLite database for quiz history, weak topics, and streaks (creates `study_buddy.db` automatically on first run — this file is gitignored, so your progress stays local)

## Demo tips (for a hackathon video)

The features most worth showing on camera, in order of impact:

1. **Concept map** — paste a chunk of notes, hit generate, and drag the nodes around live. This is the most visual, most "wow" moment in the app.
2. **Kahoot-style quiz** — hit "Generate quiz," pick the Kahoot-style option, and answer live on camera. The countdown timer and speed-based score jumping up make this feel like a real game, not a form.
3. **Feynman mode** — type a rough, imperfect explanation of something on camera, and show the AI catching the specific gap or misconception. This demonstrates real pedagogy, not just a chatbot wrapper.
4. **Misconception-aware quiz** — deliberately pick a wrong answer and show that the feedback explains *why* that wrong answer is tempting, not just "incorrect."
5. **Socratic doubt chat** — ask a question in Socratic mode and show it asking a guiding question back instead of answering directly.
6. **Progress dashboard** — take two quizzes back to back so the score-history chart and streak actually show data.

## Important: never hardcode your API key in app.py

Don't paste your real key directly into the source code (e.g. as a default value in `os.environ.get(...)`). If you ever share, commit, or push that file, your key goes with it and anyone can use your free quota. Always load it from `.env` or the sidebar input instead.

## Notes

- To change the model, edit `MODEL` in `utils.py`. As of August 2026, Groq's free/available models include `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, and `openai/gpt-oss-20b`. Groq retired `llama-3.3-70b-versatile` and `llama-3.1-70b-versatile` — check console.groq.com/docs/models for the current list if you hit a `model_not_found` error.
- Progress data (quiz scores, streaks) lives only in your local `study_buddy.db` file — delete it to reset.
- The quiz, flashcards, Feynman feedback, and concept map are all returned as JSON, so Groq's `response_format: json_object` mode is used throughout.
- The concept map renders using vis-network, loaded from a CDN — it needs an internet connection in the browser, same as the rest of the app.

## About IBM Bob and Kahoot (for the hackathon writeup)

- **IBM Bob** is IBM's IDE-integrated AI coding assistant, not an API you call at runtime from a student-facing app. If your hackathon provisioned you a Bob account, the "meaningful use of Bob" requirement is satisfied by actually using it as your coding tool while building this project (planning, generating code, extending features) — not by wiring it into `app.py`. Capture that usage (screen recording, session report) for your submission.
- **Kahoot** has no public API for creating or hosting quizzes — only a paid enterprise reports API, plus unofficial reverse-engineered bots that violate Kahoot's terms of service. The "(like Kahoot integration)" note in most project briefs describes the *feel* of the quiz (timed, gamified, engaging), not a literal integration requirement. The Kahoot-style quiz mode in this app delivers that feel natively, with no external dependency.