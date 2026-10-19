"""Render the seven-page course PDF and matching editable LaTeX source from computed evidence."""
from pathlib import Path
import json
import re
import shutil
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter

ROOT=Path(__file__).resolve().parent


def latex_escape(text):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}
    return ''.join(mapping.get(c,c) for c in text)


def main():
    content=json.loads((ROOT/'results/report_content.json').read_text())
    for alias,filename in [('TNR','times.ttf'),('TNRB','timesbd.ttf'),('TNRI','timesi.ttf')]:
        font = Path('C:/Windows/Fonts')/filename
        if font.exists():
            pdfmetrics.registerFont(TTFont(alias,str(font)))
        else:
            fallback={'TNR':'Times-Roman','TNRB':'Times-Bold','TNRI':'Times-Italic'}[alias]
            pdfmetrics.registerFont(pdfmetrics.Font(alias,fallback,'WinAnsiEncoding'))
    normal=ParagraphStyle('Body',fontName='TNR',fontSize=10,leading=11.6,spaceAfter=7)
    title=ParagraphStyle('Title',parent=normal,fontName='TNRB',fontSize=17,leading=20,alignment=TA_CENTER,spaceAfter=7)
    heading=ParagraphStyle('Heading',parent=normal,fontName='TNRB',fontSize=12,leading=14,spaceAfter=9)
    center=ParagraphStyle('Center',parent=normal,alignment=TA_CENTER,fontSize=10)
    small=ParagraphStyle('Small',parent=normal,fontSize=8,leading=9,spaceAfter=5)
    cell=ParagraphStyle('Cell',parent=normal,fontSize=8.5,leading=10,spaceAfter=0)
    story=[]
    latex=[r'\documentclass{article}',r'\usepackage[preprint,nonatbib]{neurips_2021}',r'\usepackage{graphicx,booktabs,hyperref,array,float}',r'\makeatletter\renewcommand{\@noticestring}{MH6805 Machine Learning in Finance Group 12}\makeatother',r'\title{'+latex_escape(content['title'])+'}',r'\author{Wu Yangjie \And Zhao Jingjie \And Wang Zhaobo}',r'\begin{document}',r'\maketitle']
    for i,section in enumerate(content['sections']):
        if i:
            story.append(PageBreak());latex.append(r'\clearpage')
        else:
            story += [Paragraph(content['title'],title),Paragraph(content['authors'],center),Paragraph(content['course'],center),Spacer(1,9)]
        story.append(Paragraph(section['title'],heading))
        latex.append(r'\section*{'+latex_escape(section['title'])+'}')
        for para in section['paragraphs']:
            story.append(Paragraph(escape(para),normal))
            latex += [latex_escape(para), '']
        if 'table' in section:
            t=section['table'];rows=[[Paragraph(escape(str(x)),cell) for x in row] for row in [t['headers']]+t['rows']]
            if i==1:widths=[42,265,89]
            elif i==4:widths=[45,351]
            elif len(t['headers'])==5:widths=[65,89,89,71,82]
            else:widths=None
            table=Table(rows,colWidths=widths,hAlign='CENTER')
            table.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story += [table,Spacer(1,4),Paragraph(escape(section['caption']),small)]
            n=len(t['headers']);fmt=r'p{.1\linewidth}'+''.join('p{'+str(round(.72/(n-1),3))+r'\linewidth}' for _ in range(n-1))
            latex += [r'{\small\begin{tabular}{'+fmt+'}',r'\toprule', ' & '.join(latex_escape(str(x)) for x in t['headers'])+r' \\ \midrule']
            latex += [' & '.join(latex_escape(str(x)) for x in row)+r' \\' for row in t['rows']]
            latex += [r'\bottomrule\end{tabular}}',r'\par\smallskip{\footnotesize '+latex_escape(section['caption'])+r'}\par']
        for stem,height,caption in section.get('figures',[]):
            image=Image(str(ROOT/'figures'/f'{stem}.png'))
            ratio=min(396/image.imageWidth,height*72/image.imageHeight)
            image.drawWidth=image.imageWidth*ratio;image.drawHeight=image.imageHeight*ratio
            story += [Spacer(1,4),image,Spacer(1,3),Paragraph(escape(caption),small)]
            latex += [r'\begin{center}\includegraphics[width=\linewidth,height='+str(height)+r'in,keepaspectratio]{figures/'+stem+r'.pdf}\end{center}',r'{\footnotesize '+latex_escape(caption)+r'}\par']
        if section.get('references'):
            story.append(Paragraph('References',heading));latex.append(r'\section*{References}')
        for ref in section.get('references',[]):
            story.append(Paragraph(escape(ref),small));latex += [r'{\footnotesize '+latex_escape(ref)+r'\par}']
    latex.append(r'\end{document}')
    target=ROOT/'deliverables/Group12_Final_Report.pdf'
    doc=SimpleDocTemplate(str(target),pagesize=letter,leftMargin=108,rightMargin=108,topMargin=54,bottomMargin=54,title=content['title'],author='Wu Yangjie; Zhao Jingjie; Wang Zhaobo')
    def footer(canvas,doc):
        canvas.setFont('TNR',8);canvas.drawCentredString(306,30,f'MH6805 Group 12   |   {doc.page}')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    project=ROOT/'report_latex'
    (project/'main.tex').write_text('\n'.join(latex),encoding='utf-8')
    shutil.copytree(ROOT/'figures',project/'figures',dirs_exist_ok=True)
    print(target)


if __name__=='__main__':main()
