"""PDF 텍스트 추출 및 청크 분할."""

import logging
import re
from dataclasses import dataclass, field
from io import BytesIO

from pypdf import PasswordType, PdfReader
from pypdf.errors import DependencyError, PdfReadError


# 손상 파일 등에서 pypdf가 남기는 경고("invalid pdf header" 등)는 사용자에게 이미
# 안내하므로 서버 로그에는 출력하지 않는다.
logging.getLogger("pypdf").setLevel(logging.ERROR)

MISSING_CRYPTO_MSG = (
    "AES로 암호화된 PDF를 열려면 cryptography 패키지가 필요합니다. "
    "(requirements.txt에 cryptography가 있는지 확인)"
)


@dataclass
class ParsedPDF:
    file_name: str
    file_hash: str
    size_bytes: int
    pages: list[dict] = field(default_factory=list)  # [{"file_name", "page", "text"}]
    num_pages: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def empty_pages(self) -> list[int]:
        return [p["page"] for p in self.pages if not p["text"].strip()]

    @property
    def has_text(self) -> bool:
        return any(p["text"].strip() for p in self.pages)


def extract_pdf(file_name: str, file_hash: str, data: bytes) -> ParsedPDF:
    """PDF 바이트에서 페이지별 텍스트를 추출한다. 파일은 디스크에 저장하지 않는다."""
    result = ParsedPDF(file_name=file_name, file_hash=file_hash, size_bytes=len(data))

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            # 소유자 암호만 걸린 PDF(열람 제한 없음)는 빈 암호로 열린다.
            # AES 암호화 PDF는 cryptography 패키지가 있어야 해제할 수 있다.
            try:
                decrypted = reader.decrypt("")
            except DependencyError:
                result.error = MISSING_CRYPTO_MSG
                return result
            except Exception:
                decrypted = PasswordType.NOT_DECRYPTED
            if decrypted == PasswordType.NOT_DECRYPTED:
                result.error = "열람 암호로 보호된 PDF라서 읽을 수 없습니다. 암호를 해제한 뒤 업로드해주세요."
                return result
        result.num_pages = len(reader.pages)
    except DependencyError:
        result.error = MISSING_CRYPTO_MSG
        return result
    except PdfReadError:
        result.error = "PDF 파일이 손상되었거나 올바른 PDF 형식이 아닙니다."
        return result
    except Exception as e:
        result.error = f"PDF를 여는 중 오류가 발생했습니다: {type(e).__name__}"
        return result

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
            result.warnings.append(f"{i}페이지에서 텍스트를 추출하지 못했습니다.")
        result.pages.append({"file_name": file_name, "page": i, "text": text})

    if result.num_pages == 0:
        result.error = "페이지가 없는 PDF입니다."
    elif not result.has_text:
        result.warnings.append(
            "텍스트를 추출할 수 없습니다. 스캔 이미지 PDF일 수 있습니다."
        )
    elif result.empty_pages:
        result.warnings.append(
            f"텍스트가 없는 페이지 {len(result.empty_pages)}개 "
            f"(p.{', '.join(map(str, result.empty_pages[:10]))}"
            f"{' 등' if len(result.empty_pages) > 10 else ''}) — 스캔 이미지일 수 있음"
        )

    return result


def chunk_document(doc: ParsedPDF, chunk_size: int) -> list[dict]:
    """페이지 텍스트를 chunk_size(문자 수) 이하의 청크로 묶는다.

    각 청크 텍스트에는 [p.N] 표시가 들어가 모델이 페이지를 인용할 수 있다.
    반환: [{"file_name", "page_start", "page_end", "text"}]
    """
    chunks: list[dict] = []
    buf: list[str] = []
    buf_len = 0
    start = end = None

    def flush():
        nonlocal buf, buf_len, start, end
        if buf:
            chunks.append(
                {
                    "file_name": doc.file_name,
                    "page_start": start,
                    "page_end": end,
                    "text": "\n".join(buf),
                }
            )
        buf, buf_len, start, end = [], 0, None, None

    for p in doc.pages:
        text = p["text"].strip()
        if not text:
            continue
        # 한 페이지가 chunk_size보다 길면 잘라서 넣는다.
        for i in range(0, len(text), chunk_size):
            piece = f"[p.{p['page']}]\n{text[i:i + chunk_size]}"
            if buf and buf_len + len(piece) > chunk_size:
                flush()
            if start is None:
                start = p["page"]
            end = p["page"]
            buf.append(piece)
            buf_len += len(piece)
    flush()
    return chunks


def limit_chunks(chunks: list[dict], max_chunks: int) -> list[dict]:
    """청크가 너무 많으면 문서 전체에 고르게 퍼지도록 max_chunks개를 고른다."""
    if len(chunks) <= max_chunks:
        return chunks
    if max_chunks == 1:
        return chunks[:1]
    step = (len(chunks) - 1) / (max_chunks - 1)
    return [chunks[round(i * step)] for i in range(max_chunks)]


def chunk_label(chunk: dict) -> str:
    s, e = chunk["page_start"], chunk["page_end"]
    pages = f"p.{s}" if s == e else f"p.{s}-{e}"
    return f"{chunk['file_name']} {pages}"


# ---- 질의응답용 키워드 검색 ----

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_HANGUL_RE = re.compile(r"[가-힣]")
_STOPWORDS = {
    "무엇", "어떤", "어떻게", "있나요", "있어", "알려줘", "알려주세요", "설명해줘",
    "설명", "대해", "대한", "그리고", "what", "how", "the", "and", "is", "are",
}


def _query_terms(question: str) -> set[str]:
    words = [w.lower() for w in _TOKEN_RE.findall(question)]
    terms: set[str] = set()
    for w in words:
        if w in _STOPWORDS or len(w) < 2:
            continue
        terms.add(w)
        # 한국어는 조사가 붙으므로 2글자 조각(bigram)도 함께 검색한다.
        if _HANGUL_RE.search(w) and len(w) > 2:
            terms.update(w[i:i + 2] for i in range(len(w) - 1))
    return terms - _STOPWORDS


def search_chunks(chunks: list[dict], question: str, top_k: int = 5) -> list[dict]:
    """질문 키워드와 많이 겹치는 청크를 top_k개 반환한다."""
    if len(chunks) <= top_k:
        return chunks
    terms = _query_terms(question)
    scored = []
    for idx, c in enumerate(chunks):
        low = c["text"].lower()
        score = sum(min(low.count(t), 5) * len(t) for t in terms)
        scored.append((score, idx, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = [(idx, c) for score, idx, c in scored[:top_k] if score > 0]
    if not top:
        # 일치하는 키워드가 없으면 각 문서 앞부분을 근거로 사용
        return chunks[:top_k]
    return [c for _, c in sorted(top, key=lambda x: x[0])]
