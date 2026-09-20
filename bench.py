"""
Benchmark Script cho Lab 7 — Nền Tảng Dữ Liệu & Vector Store
Chủ đề: Chính sách Thương mại Điện tử (Shopee) — Biến thể L3B
Chiến lược cá nhân: Recursive Chunking (thử lần lượt các dấu phân cách theo thứ tự ưu tiên)
Đồng bộ 5 câu hỏi benchmark thống nhất của nhóm.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Thiết lập UTF-8 cho stdout trên môi trường Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agent import KnowledgeBaseAgent
from src.chunking import ChunkingStrategyComparator, RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# 5 câu hỏi đánh giá benchmark thống nhất của nhóm (từ bench.py gốc)
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Người Mua có bao nhiêu ngày gửi yêu cầu Trả hàng/Hoàn tiền sau khi giao thành công? Thực phẩm tươi sống/đông lạnh khác không?",
        "gold_answer": "15 ngày; riêng thực phẩm tươi sống/đông lạnh là 24 giờ",
        "source": "return-refund-policy mục 3.2",
        "expected_doc": "return-refund-policy",
        "filter": None,
    },
    {
        "id": 2,
        "query": "Sản phẩm mua tại Shopee được bảo hành miễn phí khi hội đủ điều kiện nào?",
        "gold_answer": "Lỗi kỹ thuật do NSX; còn hạn bảo hành; có hóa đơn điện tử hoặc mã đơn hàng; phiếu/tem bảo hành (điện gia dụng) còn nguyên vẹn",
        "source": "buyer-warranty-policy mục 1",
        "expected_doc": "buyer-warranty-policy",
        "filter": None,
    },
    {
        "id": 3,
        "query": "Người Bán phải đảm bảo hàng còn tối thiểu bao nhiêu hạn sử dụng khi giao? Thực phẩm dưới 30 ngày thì sao?",
        "gold_answer": "Ít nhất 30% hạn sử dụng và còn ít nhất 30 ngày; thực phẩm dưới 30 ngày phải ghi rõ hạn sử dụng và tự sắp xếp vận chuyển",
        "source": "seller-listing-regulations mục D.2.a–b",
        "expected_doc": "seller-listing-regulations",
        "filter": None,
    },
    {
        "id": 4,
        "query": "Thời hạn khiếu nại vận chuyển với đơn giao không thành công (hàng thất lạc / hư hại khi hoàn trả)?",
        "gold_answer": "Thất lạc: 07 ngày; hư hại/không nguyên vẹn: 03 ngày sau khi chuyển hoàn thành công",
        "source": "shipping-policy mục D.1.b.i",
        "expected_doc": "shipping-policy",
        "filter": None,
    },
    {
        "id": 5,
        "query": "Mức bồi thường gian lận tối đa/đơn và ngày áp dụng?",
        "gold_answer": "Lên đến 10.000.000 VND mỗi đơn vi phạm, cộng dồn, cấn trừ Số dư; áp dụng từ 28/12/2023",
        "source": "seller-fraud-penalty-policy mục 3.b",
        "expected_doc": "seller-fraud-penalty-policy",
        "filter": {"audience": "seller"},
    },
]


def parse_markdown_with_frontmatter(file_path: Path) -> tuple[dict, str]:
    """Tách frontmatter YAML và phần thân bài viết Markdown."""
    text = file_path.read_text(encoding="utf-8")
    metadata: dict = {
        "doc_id": file_path.stem,
        "source": str(file_path),
    }
    content = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_meta = parts[1].strip()
            content = parts[2].strip()
            for line in raw_meta.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip().strip('"').strip("'")

    return metadata, content


def get_embedder():
    """Lấy embedder backend phù hợp (ưu tiên mock hoặc cấu hình .env)."""
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    elif provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    elif provider == "gemini":
        try:
            return GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    return _mock_embed


def run_benchmark():
    data_dir = Path("data/ecommerce")
    if not data_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu {data_dir}")
        return 1

    md_files = sorted(list(data_dir.glob("*.md")))
    if not md_files:
        print(f"Lỗi: Không tìm thấy file .md nào trong {data_dir}")
        return 1

    print("=" * 80)
    print("      LAB 7: BENCHMARK RETRIEVAL TRÊN TẬP DỮ LIỆU CHÍNH SÁCH SHOPEE")
    print("=" * 80)
    print(f"Số lượng tài liệu nạp vào: {len(md_files)}")
    for f in md_files:
        print(f"  - {f.name}")

    # Khởi tạo Chunker & Store
    chunker = RecursiveChunker(chunk_size=500)
    embedder = get_embedder()
    backend_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    print(f"\nEmbedding Backend sử dụng: {backend_name}")

    store = EmbeddingStore(collection_name="ecommerce_benchmark_store", embedding_fn=embedder)

    documents_to_add: list[Document] = []
    chunk_counts: dict[str, int] = {}

    for file_path in md_files:
        metadata, body = parse_markdown_with_frontmatter(file_path)
        chunks = chunker.chunk(body)
        doc_id = metadata.get("doc_id", file_path.stem)
        chunk_counts[doc_id] = len(chunks)

        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["doc_id"] = doc_id
            doc = Document(
                id=f"{doc_id}#{idx}",
                content=chunk_text,
                metadata=chunk_metadata,
            )
            documents_to_add.append(doc)

    store.add_documents(documents_to_add)
    print(f"Tổng số chunk được tạo và lưu trữ: {store.get_collection_size()}")
    for did, count in chunk_counts.items():
        print(f"  * {did}: {count} chunks")

    # Giả lập mô hình LLM tóm tắt câu trả lời dựa trên ngữ cảnh được cấp
    def mock_llm_summarizer(prompt: str) -> str:
        # Lấy ngữ cảnh trích dẫn đầu tiên
        lines = prompt.splitlines()
        extracted = []
        capture = False
        for l in lines:
            if "Ngữ cảnh:" in l:
                capture = True
                continue
            if "Câu hỏi:" in l:
                break
            if capture and l.strip() and not l.startswith("["):
                extracted.append(l.strip())
        summary = " ".join(extracted)[:200]
        return f"[RAG Agent Trả Lời]: Dựa theo quy định tại ngữ cảnh [1], nội dung chính xác là: {summary}..."

    agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm_summarizer)

    print("\n" + "=" * 80)
    print("                 CHẠY 5 CÂU HỎI BENCHMARK CỦA NHÓM")
    print("=" * 80)

    top3_relevant_count = 0

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        q_text = item["query"]
        gold = item["gold_answer"]
        src_note = item["source"]
        filt = item["filter"]
        expected_doc = item["expected_doc"]

        print(f"\n--- Câu hỏi {qid}: {q_text} ---")
        print(f"Gold Answer: {gold} (Nguồn: {src_note})")
        if filt:
            print(f"Metadata filter áp dụng: {filt}")
            results = store.search_with_filter(q_text, top_k=3, metadata_filter=filt)
        else:
            results = store.search(q_text, top_k=3)

        if not results:
            print("  -> Không tìm thấy kết quả phù hợp.")
            continue

        top1 = results[0]
        top1_doc = top1["metadata"].get("doc_id", "unknown")
        top1_score = top1["score"]
        top1_preview = top1["content"][:140].replace("\n", " ")

        # Kiểm tra xem có chunk liên quan trong top-3 không
        has_relevant_in_top3 = any(r["metadata"].get("doc_id") == expected_doc for r in results)
        is_top1_relevant = (top1_doc == expected_doc)

        if has_relevant_in_top3:
            top3_relevant_count += 1

        print(f"Top-1 Chunk ID: {top1['id']} (File: {top1_doc}) | Score: {top1_score:.4f}")
        print(f"Nội dung Top-1: {top1_preview}...")
        print(f"Đánh giá độ liên quan: Top-1 khớp? {'CÓ' if is_top1_relevant else 'KHÔNG'} | Có trong Top-3? {'CÓ' if has_relevant_in_top3 else 'KHÔNG'}")

        agent_ans = agent.answer(q_text, top_k=3)
        print(f"Agent Trả Lời: {agent_ans}")

    print("\n" + "=" * 80)
    print(f"TỔNG KẾT BENCHMARK: {top3_relevant_count}/5 câu hỏi có chunk liên quan trong Top-3")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_benchmark())