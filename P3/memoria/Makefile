# Uso esta plantilla para hacer el pdf
# https://github.com/Wandmalfarbe/pandoc-latex-template
# Hace falta tener instalado Pandoc y LaTex

OUT="pdf"
TEMP="./templates/eisvogel.latex"
source1 = memory.tex
source2 = presentation.tex 

all: $(OUT)/memory.pdf $(OUT)/presentation.pdf

presentation: $(OUT)/presentation.pdf

memory: $(OUT)/memory.pdf

$(OUT)/memory.pdf : $(source1)
	pdflatex -shell-escape $(source1) -output-format=pdf
	pdflatex -shell-escape $(source1) -output-format=pdf
	make clean

$(OUT)/presentation.pdf : $(source2)
	pdflatex -shell-escape $(source2) -output-format=pdf
	pdflatex -shell-escape $(source2) -output-format=pdf
	make clean

clean:
	rm -rf  $(TARGET) *.class *.html *.log *.aux *.out *.glo *.glg *.gls *.ist *.xdy *.1 *.toc *.snm *.nav *.vrb *.fls *.fdb_latexmk *.pyg
