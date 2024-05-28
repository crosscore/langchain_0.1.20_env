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
        page_content="犬は忠誠心とフレンドリーさで知られる、素晴らしいパートナーです。",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="猫は独立したペットであり、自分だけの空間を楽しむことがよくあります。",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="金魚は比較的簡単な飼育で初心者にも人気のペットです。",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="オウムは人間の言葉を真似ることができる賢い鳥です。",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="ウサギは社会的な動物であり、飛び回るのに十分なスペースが必要です。",
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
    print("Similarity Search for '猫':")
    results = vectorstore.similarity_search("猫")
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
    retriever_results = retriever.batch(["猫", "魚"])
    for query, result in zip(["猫", "魚"], retriever_results):
        print(f"Query: {query}")
        for doc in result:
            print(f"- {doc.page_content} (source: {doc.metadata['source']})")
    print("\n")

    # LLMの準備
    llm = ChatOpenAI(model="gpt-3.5-turbo")

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
    print("RAG Chain Response for '猫について教えてください':")
    response = rag_chain.invoke("猫について教えてください")
    print(response.content)

if __name__ == "__main__":
    main()
