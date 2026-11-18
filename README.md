# Wine Class Prediction — Streamlit deployment

A standalone Streamlit app using scikit-learn's built-in Wine dataset. **No CSV, API keys, model files, or database required.**
 live demo-https://7j3ihsu54x6e7x3tuurwta.streamlit.app/
## Run locally


```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a **new GitHub repository** and upload **the contents of this folder** (not the enclosing ZIP).
2. Visit https://share.streamlit.io/ and create an app from that repository.
3. Set **Main file path** to `app.py` and deploy.
4. No secrets or environment variables are needed.

## Notes

The model trains automatically from the built-in Wine dataset and is cached. It predicts **three wine cultivar classes**, not wine quality. Test accuracy is calculated using a held-out stratified split. If your original Wine project predicts *quality* rather than *class*, the original dataset and target must be supplied to faithfully recreate it.
