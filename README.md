# 🌙 Luna Loves You

<div align="center">

![Luna, a smiling crescent moon holding a flower beneath a pastel night sky](assets/luna-mascot.png)

### A gentle, CSV-grounded menstrual health education chatbot

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Atkiya/Luna-Loves-You)

**Repository:** [Atkiya/Luna-Loves-You](https://github.com/Atkiya/Luna-Loves-You) · **Live app:** https://luna-loves-you.onrender.com/

</div>

---

## About Luna

Luna is a Flask-based chatbot that answers menstrual-health education questions using a curated question-and-answer CSV as its knowledge source. It retrieves relevant entries from the dataset and sends those references, together with the user's question, to a configured hosted language model. If every configured provider fails, Luna can return the best-matching stored answer when the retrieval score is high enough.

Luna is intended for general education. It does not diagnose medical conditions or provide medication doses.

## How it works

1. **Load the knowledge base.** Luna reads `menst_merged.csv` and identifies columns whose headers contain `Question` and `Answer`.
2. **Build the retrieval index.** Scikit-learn creates word-level and character-level TF-IDF representations of the stored questions.
3. **Retrieve relevant references.** For each message, Luna finds the closest questions and selects up to `TOP_ANSWERS` distinct stored answers.
4. **Generate a response.** The application tries the configured LLM providers in the order specified by `PROVIDER_ORDER`.
5. **Use a fallback when needed.** If all LLM providers fail, Luna returns a stored answer only when its retrieval score meets `FALLBACK_MIN`; otherwise, it asks the user to rephrase or consult a healthcare professional.

## Features

- CSV-based retrieval-augmented generation (RAG).
- Combined word and character TF-IDF retrieval to help match related wording and spelling variations.
- Groq, Gemini, and OpenRouter provider options using chat-completions-compatible APIs.
- Provider fallback and temporary cooldown handling for rate limits.
- A stored-answer fallback when the LLM services are unavailable.
- Request limits to reduce excessive use of hosted API quotas.
- A `/health` endpoint with dataset and provider diagnostics.
- A responsive, moon-and-pastel-themed chat interface.

## Project structure

```text
Luna-Loves-You/
├── app.py
├── menst_merged.csv
├── requirements.txt
├── render.yaml
└── assets/
    └── luna-mascot.png
```

Keep `menst_merged.csv` in the repository root beside `app.py`, unless you configure a different path with `DATA_FILE`.

## Dataset format

The CSV must include a question column and an answer column. The header names can contain additional wording, as long as they include `question` and `answer` respectively.

Example:

| Question | Answer |
|---|---|
| What is a menstrual cycle? | A curated educational answer from the knowledge base. |
| Why might a period be irregular? | A curated educational answer from the knowledge base. |

Only non-empty question-and-answer rows are indexed. Duplicate question-and-answer pairs are removed during loading.

## LLM configuration

Luna reads API credentials from environment variables. Do not put API keys in `app.py`, `README.md`, `render.yaml`, or any committed file.

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key | Not set |
| `GROQ_MODEL` | Groq model identifier | `openai/gpt-oss-120b` |
| `GEMINI_API_KEY` | Gemini API key | Not set |
| `GEMINI_MODEL` | Gemini model identifier | Not set; required to activate Gemini |
| `OPENROUTER_API_KEY` | OpenRouter API key | Not set |
| `OPENROUTER_MODEL` | OpenRouter model identifier | Not set; required to activate OpenRouter |
| `PROVIDER_ORDER` | Comma-separated provider order | `groq,gemini,openrouter` |

A provider is activated only when both its API key and model identifier are configured. Provider availability, pricing, and usage limits depend on the provider and model selected. The app can still use its CSV fallback when no LLM provider is active, provided the dataset loads successfully.

## Retrieval and usage settings

| Variable | Purpose | Default |
|---|---|---:|
| `DATA_FILE` | Path to the CSV dataset | `menst_merged.csv` beside `app.py` |
| `MIN_SCORE` | Minimum retrieval score before an LLM request is attempted | `0.10` |
| `FALLBACK_MIN` | Minimum score required to return a stored answer after LLM failure | `0.45` |
| `TOP_ANSWERS` | Maximum number of distinct references sent to the LLM | `5` |
| `PER_IP_PER_MIN` | Maximum requests from one IP address per minute | `6` |
| `DAILY_CAP` | Application-wide daily request limit | `150` |
| `LLM_TIMEOUT` | Timeout for each provider request, in seconds | `35` |

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Atkiya/Luna-Loves-You.git
cd Luna-Loves-You
```

### 2. Install dependencies

```bash
python -m venv .venv
```

Activate the virtual environment, then run:

```bash
pip install -r requirements.txt
```

### 3. Configure an LLM provider

For Groq, set these environment variables before starting the app:

```text
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
PROVIDER_ORDER=groq
```

On macOS or Linux, export them in your terminal. On Windows PowerShell, use `$env:VARIABLE_NAME = "value"`. Keep the real API key private.

### 4. Start Luna

```bash
python app.py
```

Open `http://localhost:10000` in your browser.

## Deploy to Render

The repository includes a `render.yaml` Blueprint configuration. You can use the button at the top of this README to create a Render service from the GitHub repository.

For a manual setup, create a **Web Service** connected to this repository and use:

| Render setting | Value |
|---|---|
| Runtime | Python |
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn app:app` |
| Health check path | `/health` |

Add `GROQ_API_KEY` in the Render Environment section. The Blueprint also sets the default Groq model, provider order, and CSV filename. Add Gemini or OpenRouter credentials and their model identifiers there only when you want to enable those providers.

After the first successful deployment, copy the service's `onrender.com` URL from the Render Dashboard and replace the **Live app** placeholder near the top of this README. The live URL is assigned by Render and cannot be determined from the source code alone.

## Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Opens the Luna chat interface. |
| `GET` | `/health` | Reports dataset readiness, CSV details, configured providers, cooldowns, and recent provider errors. |
| `POST` | `/chat` | Accepts a JSON message and returns Luna's text response. |

Example request:

```bash
curl -X POST http://localhost:10000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is a menstrual cycle?"}'
```

## Health check

Open `/health` on your local or deployed app. Check these fields when troubleshooting:

- `ready`: whether the CSV was loaded and the retrieval index was built.
- `questions`: number of valid question-and-answer rows indexed.
- `csv_exists`: whether the configured dataset path exists.
- `providers_active`: providers with both an API key and model configured.
- `last_error`: most recent provider errors recorded by the app.

If `ready` is `false`, inspect `error` and the Render service logs before troubleshooting the LLM connection.

## Safety and privacy

Luna provides general educational information and is not a substitute for medical care. It is designed not to diagnose conditions or give medication doses. People experiencing severe or worrying symptoms should seek medical care; emergencies require urgent local help.

Questions are sent to the selected third-party LLM provider when generation is enabled. Do not enter names, contact details, identifying information, or private medical records. Review the dataset and provider privacy terms before making the service publicly available.

## License

Add the license you intend to use for this project before redistribution. Until a license is included in the repository, users should not assume that the source code or dataset is licensed for reuse.
