from langchain_community.document_loaders import PyPDFLoader

# Loading PDF file
data = PyPDFLoader(file_path="../GRU.pdf")

docs = data.load()

# Printing first page of the PDF
print(docs[0])

print("\n\n")

# Printing last page of the PDF
print(docs[len(docs) - 1])