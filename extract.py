import pdfplumber

pdf = pdfplumber.open('./dataset/113/03-113學測數a試題定稿.pdf')
for page_num, page in enumerate(pdf.pages):
    print(f"### Page: {page_num+1}")
    text = page.extract_text()
    print(text)
