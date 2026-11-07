"""OpenAI 호출, 프롬프트, 에러 처리."""

import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import OpenAI

from utils.config import get_api_key, get_model
from utils.pdf_parser import chunk_label

MAX_RETRIES = 3
MAX_PARALLEL = 4


class LLMError(Exception):
    """사용자에게 그대로 보여줄 수 있는 한국어 에러."""


# =====================================================================
# 프롬프트
# =====================================================================

SYSTEM_PROMPT = (
    "당신은 PDF 문서를 정확하게 분석하는 전문 분석가입니다. "
    "제공된 문서 내용에만 근거해서 답변하고, 문서에 없는 내용은 추측하지 마세요. "
    "결과는 Markdown 형식으로 작성하세요."
)

LANGUAGE_INSTRUCTIONS = {
    "한국어": "반드시 한국어로 작성하세요.",
    "English": "Write your entire response in English.",
}

ANALYSIS_TYPES: dict[str, str | None] = {
    "요약": "문서 전체 내용을 3~5개 문단으로 요약하세요. 문서의 목적, 주요 내용, 결론이 드러나야 합니다.",
    "핵심 포인트": "문서의 핵심 내용을 5~10개의 bullet point로 정리하세요. 각 항목은 한두 문장으로 쓰고, 관련 페이지를 (p.N) 형식으로 표기하세요.",
    "키워드 추출": "문서의 주요 키워드/개념 10개를 뽑아 Markdown 표(| 키워드 | 설명 |)로 정리하세요. 설명은 문서 내용에 근거해 한 줄로 쓰세요.",
    "구조 분석": "문서의 목차와 섹션 구성을 파악해 계층형 목록으로 정리하고, 각 섹션의 역할과 주요 내용을 한 줄로 설명하세요.",
    "사용자 정의": None,  # 사용자가 직접 입력한 지시문 사용
}

MAP_PROMPT = """다음은 문서 '{file_name}'의 일부({pages})입니다.

최종 작업: {instruction}

이 최종 작업에 필요한 정보를 이 부분에서 빠짐없이, 간결하게 추출해 정리하세요.
수치, 고유명사, 결론은 그대로 유지하고, 각 내용 옆에 [p.N] 페이지 표시를 남기세요.

---
{text}
---"""

REDUCE_PROMPT = """다음은 문서 '{file_name}'(총 {num_parts}개 부분)의 각 부분에서 추출한 정보입니다.

이 정보를 종합해서 다음 작업을 수행하세요.
작업: {instruction}
{language}

---
{notes}
---"""

SINGLE_PROMPT = """다음은 문서 '{file_name}'의 내용입니다. 텍스트 안의 [p.N]은 페이지 번호입니다.

작업: {instruction}
{language}

---
{text}
---"""

QA_SYSTEM_PROMPT = """당신은 사용자가 업로드한 PDF 문서에 대한 질문에 답하는 도우미입니다.

규칙:
1. 아래 [참고 문서] 내용에만 근거해서 답변하세요. 일반 상식이나 추측으로 보충하지 마세요.
2. 답변의 각 내용 뒤에 출처를 [파일명 p.N] 형식으로 표기하세요.
3. 참고 문서에서 답을 찾을 수 없으면 다른 말 없이 "{not_found}"라고만 답하세요.
4. {language}

[참고 문서]
{context}"""

NOT_FOUND = {
    "한국어": "문서에서 찾을 수 없습니다.",
    "English": "I could not find this in the documents.",
}


def is_not_found(answer: str) -> bool:
    """답변이 '문서에서 찾을 수 없음' 응답인지 판별한다(어느 언어든)."""
    text = answer.strip().rstrip(".").strip()
    return any(
        text == phrase.rstrip(".") or (text.startswith(phrase.rstrip(".")) and len(text) < len(phrase) + 10)
        for phrase in NOT_FOUND.values()
    )

COMPARE_PROMPT = """다음은 여러 문서를 각각 요약한 내용입니다.

{summaries}

위 문서들을 비교 분석해서 아래 형식의 Markdown으로 작성하세요.

## 공통점
- 문서들이 공유하는 주제, 관점, 내용

## 차이점
문서별 차이를 Markdown 표로 정리하세요. 첫 열은 비교 항목(주제, 목적, 대상, 주요 내용, 결론 등), 나머지 열은 각 문서입니다.

## 종합 의견
- 문서들을 함께 읽었을 때의 시사점과 종합 평가

{language}"""


# =====================================================================
# 호출 / 에러 처리
# =====================================================================


def get_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key:
        raise LLMError("OpenAI API 키가 설정되지 않았습니다. `.env` 또는 Secrets에 OPENAI_API_KEY를 설정해주세요.")
    # 재시도는 직접 처리하므로 SDK 자동 재시도는 끈다.
    return OpenAI(api_key=api_key, timeout=120, max_retries=0)


def _is_quota_error(e: Exception) -> bool:
    return getattr(e, "code", None) == "insufficient_quota" or "insufficient_quota" in str(e)


def _is_retryable(e: Exception) -> bool:
    if isinstance(e, openai.RateLimitError):
        return not _is_quota_error(e)
    return isinstance(
        e, (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)
    )


def to_friendly_error(e: Exception) -> LLMError:
    if isinstance(e, LLMError):
        return e
    if isinstance(e, openai.AuthenticationError):
        return LLMError("API 키가 올바르지 않습니다. OPENAI_API_KEY를 확인해주세요. (401)")
    if isinstance(e, openai.PermissionDeniedError):
        return LLMError("이 API 키로는 요청한 기능을 사용할 권한이 없습니다. (403)")
    if isinstance(e, openai.NotFoundError):
        return LLMError(
            f"모델 '{get_model()}'을(를) 찾을 수 없거나 사용 권한이 없습니다. "
            "OPENAI_MODEL 설정 또는 계정 권한을 확인해주세요. (404)"
        )
    if isinstance(e, openai.RateLimitError):
        if _is_quota_error(e):
            return LLMError("OpenAI API 사용 한도(크레딧)를 초과했습니다. 결제 및 사용량을 확인해주세요. (429)")
        return LLMError("요청이 너무 많습니다. 잠시 후 다시 시도해주세요. (429)")
    if isinstance(e, openai.APITimeoutError):
        return LLMError("응답 시간이 초과되었습니다. 사이드바에서 청크 크기를 줄이거나 잠시 후 다시 시도해주세요.")
    if isinstance(e, openai.APIConnectionError):
        return LLMError("OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인해주세요.")
    if isinstance(e, openai.BadRequestError):
        msg = str(e)
        if "context_length" in msg or "maximum context" in msg or "too long" in msg:
            return LLMError("입력이 모델의 처리 한도를 초과했습니다. 사이드바에서 청크 크기를 줄여주세요.")
        return LLMError(f"요청이 거부되었습니다. (400) {getattr(e, 'message', '')}")
    if isinstance(e, openai.APIStatusError):
        return LLMError(f"OpenAI 서버 오류가 발생했습니다. ({e.status_code}) 잠시 후 다시 시도해주세요.")
    return LLMError(f"알 수 없는 오류가 발생했습니다: {type(e).__name__}")


def _with_retry(fn: Callable):
    """429/타임아웃/연결/서버 오류는 지수 백오프(1, 2, 4초)로 재시도한다."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except openai.OpenAIError as e:
            if attempt < MAX_RETRIES and _is_retryable(e):
                time.sleep(2**attempt)
                continue
            raise to_friendly_error(e) from e


def chat(messages: list[dict]) -> str:
    client = get_client()
    response = _with_retry(
        lambda: client.chat.completions.create(model=get_model(), messages=messages)
    )
    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise LLMError("모델이 빈 응답을 반환했습니다. 다시 시도해주세요.")
    return content


def chat_stream(messages: list[dict]) -> Iterator[str]:
    client = get_client()
    stream = _with_retry(
        lambda: client.chat.completions.create(
            model=get_model(), messages=messages, stream=True
        )
    )
    try:
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content
    except openai.OpenAIError as e:
        raise to_friendly_error(e) from e


def _ask(prompt: str) -> str:
    return chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])


# =====================================================================
# 기능별 함수
# =====================================================================


def analyze_document(
    chunks: list[dict],
    instruction: str,
    language: str,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """청크가 1개면 바로 분석, 여러 개면 Map-Reduce(청크별 추출 → 통합)."""
    if not chunks:
        raise LLMError("분석할 텍스트가 없습니다.")
    file_name = chunks[0]["file_name"]
    lang = LANGUAGE_INSTRUCTIONS[language]

    if len(chunks) == 1:
        result = _ask(
            SINGLE_PROMPT.format(
                file_name=file_name, instruction=instruction, language=lang, text=chunks[0]["text"]
            )
        )
        if progress:
            progress(1, 1)
        return result

    total = len(chunks) + 1
    notes: list[str | None] = [None] * len(chunks)

    def map_one(i: int) -> tuple[int, str]:
        c = chunks[i]
        pages = chunk_label(c).removeprefix(file_name).strip()
        return i, _ask(
            MAP_PROMPT.format(file_name=file_name, pages=pages, instruction=instruction, text=c["text"])
        )

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = [pool.submit(map_one, i) for i in range(len(chunks))]
        try:
            for f in as_completed(futures):
                i, note = f.result()
                notes[i] = note
                done += 1
                if progress:
                    progress(done, total)
        except Exception:
            for f in futures:
                f.cancel()
            raise

    joined = "\n\n".join(
        f"### 부분 {i + 1} ({chunk_label(chunks[i])})\n{n}" for i, n in enumerate(notes)
    )
    result = _ask(
        REDUCE_PROMPT.format(
            file_name=file_name, num_parts=len(chunks), instruction=instruction, language=lang, notes=joined
        )
    )
    if progress:
        progress(total, total)
    return result


def answer_question_stream(
    question: str,
    context_chunks: list[dict],
    history: list[dict],
    language: str,
) -> Iterator[str]:
    context = "\n\n".join(f"[문서: {chunk_label(c)}]\n{c['text']}" for c in context_chunks)
    system = QA_SYSTEM_PROMPT.format(
        not_found=NOT_FOUND[language], language=LANGUAGE_INSTRUCTIONS[language], context=context
    )
    messages = [{"role": "system", "content": system}]
    # 후속 질문을 이해할 수 있도록 최근 대화 6개만 포함
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    messages.append({"role": "user", "content": question})
    return chat_stream(messages)


def compare_documents(summaries: dict[str, str], language: str) -> str:
    joined = "\n\n".join(f"### 문서: {name}\n{summary}" for name, summary in summaries.items())
    return _ask(COMPARE_PROMPT.format(summaries=joined, language=LANGUAGE_INSTRUCTIONS[language]))
