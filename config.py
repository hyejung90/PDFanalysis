"""설정 로딩: st.secrets(Streamlit Cloud) → .env / 환경 변수(로컬) 순서."""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gpt-6.1-terra"


def get_config(key: str, default: str | None = None) -> str | None:
    """설정 값을 st.secrets → os.environ 순서로 찾는다."""
    try:
        if key in st.secrets:
            value = st.secrets[key]
            if value:
                return str(value)
    except Exception:
        # 로컬에 secrets.toml이 없으면 st.secrets 접근 시 예외가 발생할 수 있다.
        pass
    value = os.getenv(key)
    return value if value else default


def get_api_key() -> str | None:
    return get_config("OPENAI_API_KEY")


def get_model() -> str:
    return get_config("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
