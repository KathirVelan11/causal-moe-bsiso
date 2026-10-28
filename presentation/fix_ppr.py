"""pptxgenjs writes one <a:pPr> per run in multi-run paragraphs; keep only the
first (the one carrying the bullet) so PowerPoint reads the intended format.
Usage: python fix_ppr.py deck.pptx
"""
import re
import shutil
import sys
import tempfile
import zipfile

from lxml import etree

A = "http://schemas.openxmlformats.org/drawingml/2006/main"
path = sys.argv[1]
tmp = tempfile.mktemp(suffix=".pptx")
fixed = 0
with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if re.match(r"ppt/slides/slide\d+\.xml$", item.filename):
            root = etree.fromstring(data)
            for p in root.iter(f"{{{A}}}p"):
                pprs = p.findall(f"{{{A}}}pPr")
                for extra in pprs[1:]:
                    p.remove(extra)
                    fixed += 1
                if pprs and p.index(pprs[0]) != 0:
                    p.remove(pprs[0])
                    p.insert(0, pprs[0])
            data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        zout.writestr(item, data)
shutil.move(tmp, path)
print("removed", fixed, "duplicate pPr")
