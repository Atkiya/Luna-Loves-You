# 🌙 Luna Loves You

<div align="center">

![Luna, a smiling crescent moon holding a flower beneath a pastel night sky](assets/luna-mascot.png)

### A gentle, CSV-grounded menstrual health education chatbot


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



## Safety and privacy

Luna provides general educational information and is not a substitute for medical care. It is designed not to diagnose conditions or give medication doses. People experiencing severe or worrying symptoms should seek medical care; emergencies require urgent local help.

Questions are sent to the selected third-party LLM provider when generation is enabled. Do not enter names, contact details, identifying information, or private medical records. Review the dataset and provider privacy terms before making the service publicly available.

## License

Add the license you intend to use for this project before redistribution. Until a license is included in the repository, users should not assume that the source code or dataset is licensed for reuse.
