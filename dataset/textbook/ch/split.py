from PyPDF2 import PdfWriter, PdfReader
import os
from os import listdir
from os.path import isfile, join
files = [f for f in listdir("./") if isfile(join("./", f))]
for filename in files:
    if  not filename.endswith(".pdf"):
        continue
    inputpdf = PdfReader(open(filename, "rb"))
    plain_name = filename.replace(".pdf" , "")
    for i in range(len(inputpdf.pages)):
        output = PdfWriter()
        output.add_page(inputpdf.pages[i])
        with open(f"s-{plain_name}-page_{i+1:03}.pdf", "wb") as outputStream:
            output.write(outputStream)