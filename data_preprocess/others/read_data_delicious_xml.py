import pandas as pd
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import parse
import io
import re
#
# text = open("../data/delicious/delicious_crawl_min.xml", encoding="utf8").read()
# text = re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+", u"", text)
#
# data = pd.read_xml(text)
# print("data:", data)
#
# # data.to_csv("../data/delicious/delicious.csv", index=False)
#
import xml.etree.ElementTree as ET

from lxml import etree
#
parser = etree.XMLParser(recover=True)
# etree.fromstring("../data/delicious/delicious_crawl_min.xml", parser=parser)

tree = ET.parse("../data/delicious/delicious_crawl.xml", parser=parser)
# tree = ET.parse("../data/delicious/delicious_crawl_min.xml", parser=parser)
# tree = ET.parse(text)
root = tree.getroot()

cols_ui = ["user", "item"]
cols_it = ["item", "tag"]
rows_ui = []
rows_it = []

for node in root:
    s_user = node.attrib.get("u")
    s_item = node.attrib.get("href")
    for t in node.iter('t'):
        # t = re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+", u"", t.text)
        rows_it.append({"item": s_item, "tag": t.text})

    rows_ui.append({"user": s_user, "item": s_item})

df_ui = pd.DataFrame(rows_ui, columns=cols_ui)
df_it = pd.DataFrame(rows_it, columns=cols_it)

print("df_ui:", df_ui)
print("df_it:", df_it)

df_ui.to_csv("../data/delicious/delicious_ui.csv", index=False)
df_it.to_csv("../data/delicious/delicious_it.csv", index=False)
