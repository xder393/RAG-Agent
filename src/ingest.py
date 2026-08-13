def load_text_file(file_path):
    """
    Load the content of a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the text file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
def load_pdf_file(file_path):
    """
    Load the content of a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content of the PDF file.
    """
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
def chunk_text(text, chunk_size=100, overlap=20):
    """
    Split the text into chunks of specified size with optional overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >=len(text):
            break
        start = end - overlap
    return chunks
if __name__ == "__main__":
    content=load_text_file("data/sample.txt")
    chunks=chunk_text(content, chunk_size=100, overlap=20)
    print(f"Total chunks created: {len(chunks)}")
    for i,chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")