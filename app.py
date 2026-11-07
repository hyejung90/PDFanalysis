import hashlib
import html
from datetime import datetime

import streamlit as st

from utils.config import get_api_key, get_model
from utils.llm import (
    ANALYSIS_TYPES,
    LLMError,
    analyze_document,
    answer_question_stream,
    compare_documents,
    is_not_found,
)
from utils.markdown_fix import fix_markdown
from utils.pdf_parser import (
    ParsedPDF,
    chunk_document,
    chunk_label,
    extract_pdf,
    limit_chunks,
    search_chunks,
)

st.set_page_config(page_title="PDF 분석기", page_icon="📄", layout="wide")

TOP_K = 5  # 질의응답 시 참고할 청크 수
CARDS_PER_ROW = 3

# 최소한의 스타일: 본문 폭 제한, 카드 안 긴 파일명 줄바꿈,
# 결과 표는 좁은 화면에서 가로 스크롤 + 첫 열(항목명)이 한 글자씩 끊기지 않게
st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] { max-width: 1100px; padding-top: 2.5rem; }
    .file-name { font-weight: 600; overflow-wrap: anywhere; line-height: 1.4; }
    [data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; }
    [data-testid="stMarkdownContainer"] th:first-child,
    [data-testid="stMarkdownContainer"] td:first-child { min-width: 5.5em; word-break: keep-all; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=50)
def parse_pdf_cached(file_hash: str, file_name: str, _data: bytes) -> ParsedPDF:
    # 캐시 키는 file_hash + file_name. 이름이 같아도 내용이 다르면 따로 처리된다.
    return extract_pdf(file_name, file_hash, _data)


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def init_state():
    st.session_state.setdefault("analysis_results", {})  # key -> markdown
    st.session_state.setdefault("compare_results", {})  # key -> markdown
    st.session_state.setdefault("messages", [])  # [{"role", "content", "sources"}]


def analysis_key(doc: ParsedPDF, analysis_type: str, instruction: str) -> tuple:
    return (doc.file_hash, analysis_type, instruction, language, chunk_size, max_chunks)


def run_analysis(doc: ParsedPDF, analysis_type: str, instruction: str, label: str) -> str:
    """캐시된 결과가 없을 때만 API를 호출한다."""
    key = analysis_key(doc, analysis_type, instruction)
    results = st.session_state.analysis_results
    if key in results:
        return results[key]

    chunks = analysis_chunks[doc.file_hash]
    bar = st.progress(0.0, text=f"{label} — {doc.file_name} 분석 중...")

    def on_progress(done: int, total: int):
        bar.progress(done / total, text=f"{label} — {doc.file_name} 분석 중... ({done}/{total})")

    try:
        result = analyze_document(chunks, instruction, language, on_progress)
    finally:
        bar.empty()
    result = fix_markdown(result)
    results[key] = result
    return result


def download_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M}.md"


def render_file_card(d: ParsedPDF):
    with st.container(border=True):
        if not d.ok:
            icon, status = "❌", "오류"
        elif d.warnings:
            icon, status = "⚠️", "경고"
        else:
            icon, status = "✅", "정상"
        # 파일명은 사용자 입력이므로 HTML 이스케이프 필수
        st.markdown(
            f'<div class="file-name">📄 {html.escape(d.file_name)}</div>', unsafe_allow_html=True
        )
        pages = f"{d.num_pages}페이지 · " if d.ok else ""
        st.caption(f"{pages}{format_size(d.size_bytes)} · {icon} {status}")
        if d.error:
            st.caption(f"❌ {d.error}")
        for w in d.warnings:
            st.caption(f"⚠️ {w}")


def render_welcome():
    st.markdown("#### 이렇게 사용하세요")
    steps = [
        ("1️⃣ PDF 업로드", "위 영역에 분석할 PDF를 끌어다 놓으세요. 여러 개를 한 번에 올릴 수 있습니다."),
        ("2️⃣ 분석 설정", "왼쪽 사이드바에서 응답 언어와 분석 유형(요약, 핵심 포인트 등)을 고르세요."),
        ("3️⃣ 결과 확인", "분석 결과를 보고, 질의응답 탭에서 질문하거나 여러 문서를 비교해보세요."),
    ]
    for col, (title, desc) in zip(st.columns(3), steps):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)


# ---------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------

init_state()
api_key = get_api_key()
model = get_model()

with st.sidebar:
    st.markdown("### 🤖 모델")
    st.markdown(f"`{model}`")
    if api_key:
        st.success("API 키가 연결되었습니다.", icon="✅")
    else:
        st.error("API 키가 없습니다.", icon="🔑")

    st.divider()
    st.markdown("### 🔍 분석 설정")
    language = st.radio("응답 언어", ["한국어", "English"], horizontal=True)
    analysis_type = st.selectbox("분석 유형", list(ANALYSIS_TYPES.keys()))

    st.divider()
    st.markdown("### ⚙️ 처리 설정")
    chunk_size = st.slider(
        "청크 크기 (글자 수)", 2000, 12000, 6000, step=1000,
        help="문서를 이 크기로 나눠 모델에 보냅니다. 오류가 나면 줄여보세요.",
    )
    max_chunks = st.slider(
        "문서당 최대 청크 수", 1, 50, 20,
        help="긴 문서의 API 호출 수(비용)를 제한합니다. 초과하면 문서 전체에서 고르게 골라 분석합니다.",
    )

    st.divider()
    st.markdown("### 💬 대화")
    # st.rerun()을 쓰면 아직 그려지지 않은 file_uploader 상태가 지워져 업로드 파일이 사라진다.
    # 콜백은 스크립트 실행 전에 처리되므로 다른 위젯 상태에 영향이 없다.
    st.button(
        "대화 초기화",
        icon="🗑️",
        use_container_width=True,
        on_click=lambda: st.session_state.update(messages=[]),
    )


# ---------------------------------------------------------------------
# 헤더 & 업로드
# ---------------------------------------------------------------------

st.title("📄 PDF 분석기")
st.caption(f"PDF를 올리면 `{model}`이(가) 요약·분석하고, 문서 내용에 대한 질문에 답합니다.")

if not api_key:
    st.warning(
        "OpenAI API 키가 설정되지 않았습니다. 로컬에서는 `.env` 파일에, "
        "Streamlit Cloud에서는 Secrets에 `OPENAI_API_KEY`를 설정해주세요. "
        "API 키가 없으면 분석 기능을 사용할 수 없습니다.",
        icon="🔑",
    )

uploaded_files = st.file_uploader(
    "PDF 파일을 업로드하세요 (여러 개 가능)",
    type=["pdf"],
    accept_multiple_files=True,
)

parsed: list[ParsedPDF] = []
skipped_duplicates: list[str] = []
# 업로드 파일별(file_id) 추출 결과. 위젯을 조작할 때마다 파일을 다시 읽고 해시하지 않도록 한다.
upload_cache: dict[str, ParsedPDF] = st.session_state.setdefault("upload_cache", {})
current_ids = {f.file_id for f in uploaded_files or []}
for stale in set(upload_cache) - current_ids:
    del upload_cache[stale]

if uploaded_files:
    with st.spinner("PDF에서 텍스트를 추출하는 중..."):
        seen = set()
        for f in uploaded_files:
            doc = upload_cache.get(f.file_id)
            if doc is None:
                data = f.getvalue()
                doc = parse_pdf_cached(hashlib.sha256(data).hexdigest(), f.name, data)
                upload_cache[f.file_id] = doc
            if doc.file_hash in seen:
                skipped_duplicates.append(f.name)
                continue
            seen.add(doc.file_hash)
            parsed.append(doc)

if not parsed:
    render_welcome()
    st.stop()


# ---------------------------------------------------------------------
# 파일 목록
# ---------------------------------------------------------------------

st.markdown(f"#### 업로드한 파일 ({len(parsed)}개)")
n_cols = min(CARDS_PER_ROW, len(parsed))
for i in range(0, len(parsed), n_cols):
    row = parsed[i:i + n_cols]
    for col, d in zip(st.columns(n_cols), row):
        with col:
            render_file_card(d)

for name in skipped_duplicates:
    st.info(f"**{name}**: 같은 내용의 파일이 이미 업로드되어 건너뜁니다.", icon="ℹ️")

# 분석 가능한 문서 = 오류 없고 텍스트가 있는 문서
docs = [d for d in parsed if d.ok and d.has_text]
all_chunks = {d.file_hash: chunk_document(d, chunk_size) for d in docs}
analysis_chunks = {h: limit_chunks(c, max_chunks) for h, c in all_chunks.items()}

for d in docs:
    total, used = len(all_chunks[d.file_hash]), len(analysis_chunks[d.file_hash])
    if used < total:
        st.caption(
            f"ℹ️ {d.file_name}: 전체 {total}개 청크 중 {used}개만 분석에 사용합니다 "
            "(사이드바의 '문서당 최대 청크 수'로 조정)."
        )

can_call = bool(api_key) and bool(docs)


# ---------------------------------------------------------------------
# 탭
# ---------------------------------------------------------------------

tab_analysis, tab_qa, tab_compare = st.tabs(["📊 분석", "💬 질의응답", "⚖️ 문서 비교"])

# ---- 분석 (F-03) ----
with tab_analysis:
    if not docs:
        st.info("텍스트를 읽을 수 있는 PDF가 없습니다. 다른 파일을 업로드해주세요.", icon="📭")
    else:
        custom_instruction = ""
        if analysis_type == "사용자 정의":
            custom_instruction = st.text_area(
                "분석 지시문",
                placeholder="예: 이 문서에서 고객에게 불리할 수 있는 조항을 찾아 정리해줘",
            ).strip()
        else:
            st.caption(f"**{analysis_type}** · {ANALYSIS_TYPES[analysis_type]}")
        instruction = custom_instruction or ANALYSIS_TYPES[analysis_type] or ""

        names = {d.file_hash: d.file_name for d in docs}
        selected = st.multiselect(
            "분석할 문서",
            options=list(names.keys()),
            default=list(names.keys()),
            format_func=lambda h: names[h],
        )
        selected_docs = [d for d in docs if d.file_hash in selected]

        # 지시문 유무로 버튼을 비활성화하면, 입력 직후(포커스가 아직 입력란에 있을 때)
        # 값이 반영되기 전이라 첫 클릭이 무시된다. 클릭 시점에 검사한다.
        run = st.button(
            f"'{analysis_type}' 분석 실행",
            type="primary",
            icon="▶️",
            disabled=not can_call or not selected_docs,
        )
        if run and not instruction:
            st.warning("분석 지시문을 입력해주세요.", icon="✏️")
        elif run:
            try:
                for d in selected_docs:
                    run_analysis(d, analysis_type, instruction, analysis_type)
                st.success("분석이 완료되었습니다.", icon="✅")
            except LLMError as e:
                st.error(str(e), icon="🚨")

        export = []
        for d in selected_docs:
            result = st.session_state.analysis_results.get(analysis_key(d, analysis_type, instruction))
            if result:
                with st.expander(f"📑 {d.file_name}", expanded=True):
                    st.markdown(result)
                export.append(f"## {d.file_name}\n\n{result}")

        if export:
            header = f"# PDF 분석 결과 — {analysis_type}\n\n"
            if analysis_type == "사용자 정의":
                header += f"> 지시문: {instruction}\n\n"
            st.download_button(
                "분석 결과 다운로드 (.md)",
                data=header + "\n\n".join(export),
                file_name=download_name("pdf_analysis"),
                mime="text/markdown",
                icon="📥",
            )

# ---- 질의응답 (F-04) ----
with tab_qa:
    if not docs:
        st.info("텍스트를 읽을 수 있는 PDF가 없습니다. 다른 파일을 업로드해주세요.", icon="📭")

    messages_box = st.container()
    with messages_box:
        if docs and not st.session_state.messages:
            st.caption(
                "💡 업로드한 문서 내용에 대해 질문해보세요. 답변에는 근거가 된 문서와 페이지가 표시됩니다. "
                "예) \"이 문서의 결론은 무엇인가요?\""
            )
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("sources"):
                    st.caption("📎 참고: " + " · ".join(m["sources"]))

    question = st.chat_input(
        "문서에 대해 질문하세요",
        disabled=not can_call,
    )
    if question:
        chunks = [c for d in docs for c in all_chunks[d.file_hash]]
        context = search_chunks(chunks, question, TOP_K)
        sources = [chunk_label(c) for c in context]
        history = list(st.session_state.messages)

        with messages_box:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    answer = st.write_stream(
                        answer_question_stream(question, context, history, language)
                    )
                    answer = answer if isinstance(answer, str) else "".join(map(str, answer))
                    if not answer.strip():
                        answer = None
                        st.error("모델이 빈 응답을 반환했습니다. 다시 시도해주세요.", icon="🚨")
                    elif is_not_found(answer):
                        # 근거가 없다는 답변에는 출처를 붙이지 않는다.
                        sources = []
                    else:
                        st.caption("📎 참고: " + " · ".join(sources))
                except LLMError as e:
                    answer = None
                    st.error(str(e), icon="🚨")

        if answer:
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append(
                {"role": "assistant", "content": fix_markdown(answer), "sources": sources}
            )

    if st.session_state.messages:
        def to_md(m: dict) -> str:
            if m["role"] == "user":
                return f"### Q. {m['content']}"
            # 답변이 목록·표로 시작해도 형식이 깨지지 않도록 본문은 별도 문단으로
            text = f"**A.**\n\n{m['content']}"
            if m.get("sources"):
                text += f"\n\n_참고: {' · '.join(m['sources'])}_"
            return text

        chat_md = "# PDF 질의응답 기록\n\n" + "\n\n".join(map(to_md, st.session_state.messages))
        st.download_button(
            "대화 기록 다운로드 (.md)",
            data=chat_md,
            file_name=download_name("pdf_chat"),
            mime="text/markdown",
            icon="📥",
        )

# ---- 문서 비교 (F-05) ----
with tab_compare:
    if len(docs) < 2:
        st.info("문서 비교는 텍스트가 있는 PDF를 2개 이상 업로드하면 사용할 수 있습니다.", icon="📚")
    else:
        st.caption("비교 대상: " + " · ".join(d.file_name for d in docs))
        compare_key = (tuple(sorted(d.file_hash for d in docs)), language, chunk_size, max_chunks)
        if st.button("문서 비교 실행", type="primary", icon="▶️", disabled=not can_call):
            try:
                summary_instruction = ANALYSIS_TYPES["요약"]
                summaries = {
                    d.file_name: run_analysis(d, "요약", summary_instruction, "비교용 요약")
                    for d in docs
                }
                with st.spinner("문서를 비교하는 중..."):
                    st.session_state.compare_results[compare_key] = fix_markdown(
                        compare_documents(summaries, language)
                    )
                st.success("비교가 완료되었습니다.", icon="✅")
            except LLMError as e:
                st.error(str(e), icon="🚨")

        result = st.session_state.compare_results.get(compare_key)
        if result:
            with st.container(border=True):
                st.markdown(result)
            st.download_button(
                "비교 결과 다운로드 (.md)",
                data=f"# PDF 문서 비교\n\n대상: {', '.join(d.file_name for d in docs)}\n\n{result}",
                file_name=download_name("pdf_compare"),
                mime="text/markdown",
                icon="📥",
            )
