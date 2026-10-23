# RAG that runs in console

from src.pipeline import RagPipeline

def main():
    rag = RagPipeline()
    print("Ask about your documents. \n Type 'EXIT' to quit.")
    while True:
        q = input("\n>").strip()
        if q.lower() in ["exit", "quit"]:
            break
        if not q:
            continue
        ans = rag.ask(q)
        print(f"\n{ans.text}")
        for s in ans.sources:
            print(f"- {s['source']}, page {s['page']}")

if __name__ == "__main__":
    main()