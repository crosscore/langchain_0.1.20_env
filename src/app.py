import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 環境変数のロード
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ドキュメントの準備
documents = [
    Document(
        page_content="ハムスターは飼いやすく、小さな子供にも人気のあるペットです。",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="フェレットは遊び心があり、飼い主との絆を深めることができます。",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="ベタは美しい色合いを持ち、水槽を彩る魅力的な魚です。",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="カナリアは美しい声で歌うことができ、音楽的な楽しみを提供します。",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="モルモットは社交的で、グループで飼うと特に幸せになります。",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# VectorStoreの準備
vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

def main():
    # 類似度検索の例
    print("Similarity Search for 'ハムスター':")
    results = vectorstore.similarity_search("ハムスター")
    for result in results:
        print(f"- {result.page_content} (source: {result.metadata['source']})")
    print("\n")

    # Retrieverの準備
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    # Retrieverの実行例
    print("Retriever Batch Search Results:")
    retriever_results = retriever.batch(["ハムスター", "カナリア"])
    for query, result in zip(["ハムスター", "カナリア"], retriever_results):
        print(f"Query: {query}")
        for doc in result:
            print(f"- {doc.page_content} (source: {doc.metadata['source']})")
    print("\n")

    # LLMの準備
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens=100)

    # プロンプトテンプレートの準備
    message = """
    以下のContextのみを使用して、Questionに答えてください。

    Context:
    {context}

    Question:
    {question}
    """

    prompt = ChatPromptTemplate.from_messages([("human", message)])

    # RAGチェーンの準備
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    # 質問応答
    print("RAG Chain Response for 'ハムスターについて教えてください':")
    response = rag_chain.invoke("ハムスターについて教えてください")
    print(response.content)

if __name__ == "__main__":
    main()
