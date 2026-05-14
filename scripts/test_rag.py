"""
test_rag.py — Quick CLI test of the RAG pipeline

Usage:
  python scripts/test_rag.py --pdf path/to/report.pdf
  python scripts/test_rag.py --ask "What are the key targets?"
  python scripts/test_rag.py --insights
"""

import sys
import argparse
sys.path.append(".")

from backend.rag import index_document, answer_question, generate_insights, is_document_indexed


def main():
    parser = argparse.ArgumentParser(description="Test the Niti Aayog RAG pipeline")
    parser.add_argument("--pdf", type=str, help="Path to PDF to index")
    parser.add_argument("--ask", type=str, help="Question to ask")
    parser.add_argument("--insights", action="store_true", help="Generate insights")
    args = parser.parse_args()

    if args.pdf:
        print(f"\n📄 Indexing: {args.pdf}")
        count = index_document(args.pdf)
        print(f"✅ Done! Indexed {count} chunks.\n")

    elif args.ask:
        if not is_document_indexed():
            print("❌ No document indexed. Run with --pdf first.")
            return
        print(f"\n❓ Question: {args.ask}")
        result = answer_question(args.ask)
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n📄 Sources ({len(result['sources'])} chunks retrieved):")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src[:150]}...")

    elif args.insights:
        if not is_document_indexed():
            print("❌ No document indexed. Run with --pdf first.")
            return
        print("\n💡 Generating insights...")
        insights = generate_insights()
        print("\nKey Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
