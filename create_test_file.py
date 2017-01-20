
import pandas as pd 
from bs4 import BeautifulSoup

f = open("../2281.txt")

arr = []
for line in f:
	line = line.strip().replace("\n","").replace("/n","").strip().replace(",",";")
	arr.extend(line.split("."))
	
f = open("classify.csv", "w+")
f.write("text,label\n")
for val in arr:
	f.write(val + ", \n")

f.close()
