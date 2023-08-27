from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


# Load documents from directory into memory to be chunked and emmbeded
loader = DirectoryLoader("documents/", loader_cls=TextLoader)


# The text has to be split now, as this allows the embedding model
# to process the documents, it is incapable of processing large
# amounts of texts in one go, it has a token limit
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True,
)


splitted_text = loader.load_and_split(text_splitter=text_splitter)


# Load documents into vector database. The document has to be embedded first,
# which takes the document and passes it to a vector, to do this we need to
# use the Embeddings class provided by LangChain, and we also need to specify
# which embedding model provider we will use, and ensure our query (input)
# also is embedded using the same model, otherwise we won't obtain meaningful
# results from querying the data, since the query and the document are
# passed to a multidimensional vector, and then we try to calculate the
# similarity of the embedded query against the embedded document (similar
# to Euclidian distance but with a multidimensional vector of floating
# point numbers) we use the closest vectors to provide a response. Because we
# are building a chatbot using free resources we must use the package
# sentence_transformer and then use a model from hugging face:
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# This model will embed both the document and the query
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)


# The below is for when the DB hasn't been created since we already processed
# the API document and the vector db is saved locally we can instantiate a
# Chroma object directly and load the DB instead
db = Chroma.from_documents(
    splitted_text,
    embeddings,
    persist_directory="chromadb/",
    collection_name="langchain_test_collection",
    verbose=True,
)


# The bellow is just to load the already existing DB
# db = Chroma(
#     collection_name="api_functions_signatures_and_docstrings",
#     persist_directory="chromadb/",
#     embedding_function=embeddings,
# )


model_path = (
    "path/to/your/download/folder/wizardlm-1.0-uncensored-llama2-13b.ggmlv3.q5_K_S.bin"
)
llm = GPT4All(
    model=model_path,
    backend="llama",
    verbose=True,
    temp=0.1,
    n_ctx=1024,
)


qa = RetrievalQA.from_llm(
    llm=llm,
    retriever=db.as_retriever(verbose=True),
    return_source_documents=True,
    verbose=True,
)


result = qa(
    {
        "query": "What did the president say about justice Breyer?"
    }
)

print(result["result"])
print(result["source_documents"][0])
