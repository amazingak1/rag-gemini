from config import TOP_K, PROMPT_TEMPLATE, GEMINI_MODEL, client

def run_rag(query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=query
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text, docs
