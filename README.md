# 📄 PDF 분석기

PDF를 업로드하면 OpenAI 모델로 요약·분석·질의응답을 제공하는 Streamlit 앱입니다.

## 로컬 실행

```bash
pip install -r requirements.txt
cp .env.example .env   # .env에 OPENAI_API_KEY 입력
streamlit run app.py
```

## Streamlit Cloud 배포

1. 이 저장소를 GitHub에 push (`.env`는 `.gitignore`로 제외됨)
2. https://share.streamlit.io → New app → Main file path: `app.py`
3. Advanced settings → Secrets에 입력:
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-6.1-terra"
   ```
