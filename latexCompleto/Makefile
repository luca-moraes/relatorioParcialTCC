NAME = fei
SHELL = bash
PWD  = $(shell pwd)
TEMP := $(shell mktemp -d)
TDIR = $(TEMP)/$(NAME)
VERS = $(shell ltxfileinfo -v $(NAME).cls)
LOCAL = $(shell kpsewhich --var-value TEXMFLOCAL)
UTREE = $(shell kpsewhich --var-value TEXMFHOME)
$(NAME).cls README.txt $(NAME).pdf: $(NAME).dtx
	latexmk -pdf $(NAME).dtx -gg
clean:
	git clean -Xdf
format:
	cp fei.dtx tmp.cls
	latexindent -w tmp.cls
	mv tmp.cls fei.dtx
distclean: clean
	rm -f *.{pdf,cls} README.txt fei-template.tex fei-template-sublist.tex $(NAME)-$(VERS).zip
inst: $(NAME).cls
	mkdir -p $(UTREE)/{tex,source,doc}/latex/$(NAME)
	cp $(NAME).dtx $(UTREE)/source/latex/$(NAME)
	cp $(NAME).cls $(UTREE)/tex/latex/$(NAME)
	cp $(NAME).cls $(UTREE)/doc/latex/$(NAME)
install: $(NAME).cls
	sudo mkdir -p $(LOCAL)/{tex,source,doc}/latex/$(NAME)
	sudo cp $(NAME).dtx $(LOCAL)/source/latex/$(NAME)
	sudo cp $(NAME).cls $(LOCAL)/tex/latex/$(NAME)
	sudo cp $(NAME).cls $(LOCAL)/doc/latex/$(NAME)
zip: $(NAME).cls $(NAME).pdf fei-template.tex fei-template-sublist.tex README.txt
	mkdir $(TDIR)
	cp README.txt $(TDIR)/README
	cp $(NAME).{pdf,dtx} fei-template*.tex referencias.bib $(TDIR)
	cd $(TEMP); zip -Drq $(PWD)/$(NAME)-$(VERS).zip $(NAME)
templates fei-template.tex fei-template-sublist.tex: tests/test-full-template.tex tests/test-full-template-sublist.tex
	cp tests/test-full-template.tex fei-template.tex
	cp tests/test-full-template-sublist.tex fei-template-sublist.tex
tests tests/test-full-template.tex tests/test-full-template-sublist.tex: $(NAME).cls tests/pieces/*.tex
	awk 'FNR==1{print ""}{print}' tests/pieces/documentclass.tex \
		tests/pieces/inputenc-author-title.tex \
		tests/pieces/newsubfloatfigure.tex \
		tests/pieces/subtitulo.tex \
		tests/pieces/acronimos.tex \
		tests/pieces/simbolos.tex \
		tests/pieces/addbibresource.tex \
		tests/pieces/makeindex.tex \
		tests/pieces/makeglossaries.tex \
		tests/pieces/begin-document.tex \
		tests/pieces/titulo.tex \
		tests/pieces/folha-de-rosto.tex \
		tests/pieces/cat-aprov.tex \
		tests/pieces/dedicatoria.tex \
		tests/pieces/agradecimentos.tex \
		tests/pieces/epigrafe.tex \
		tests/pieces/resumo.tex \
		tests/pieces/abstract.tex \
		tests/pieces/tocloftasa.tex \
		tests/pieces/first-chapter-title.tex \
		tests/pieces/first-chapter-text.tex \
		tests/pieces/document-text.tex \
		tests/pieces/printbibliography.tex \
		tests/pieces/printindex.tex \
		tests/pieces/end-document.tex > tests/test-full-template.tex

	awk 'FNR==1{print ""}{print}' tests/pieces/documentclass-numeric.tex \
		tests/pieces/inputenc-author-title.tex \
		tests/pieces/newsubfloatfigure.tex \
		tests/pieces/subtitulo.tex \
		tests/pieces/acronimos.tex \
		tests/pieces/simbolos.tex \
		tests/pieces/addbibresource.tex \
		tests/pieces/makeindex.tex \
		tests/pieces/makeglossaries.tex \
		tests/pieces/begin-document.tex \
		tests/pieces/titulo.tex \
		tests/pieces/folha-de-rosto.tex \
		tests/pieces/cat-aprov.tex \
		tests/pieces/dedicatoria.tex \
		tests/pieces/agradecimentos.tex \
		tests/pieces/epigrafe.tex \
		tests/pieces/resumo.tex \
		tests/pieces/abstract.tex \
		tests/pieces/tocloftasa.tex \
		tests/pieces/first-chapter-title.tex \
		tests/pieces/first-chapter-text.tex \
		tests/pieces/document-text.tex \
		tests/pieces/printbibliography.tex \
		tests/pieces/printindex.tex \
		tests/pieces/end-document.tex > tests/test-full-template-numeric.tex

	awk 'FNR==1{print ""}{print}' tests/pieces/documentclass-backrefs.tex \
		tests/pieces/inputenc-author-title.tex \
		tests/pieces/newsubfloatfigure.tex \
		tests/pieces/subtitulo.tex \
		tests/pieces/acronimos.tex \
		tests/pieces/simbolos.tex \
		tests/pieces/addbibresource.tex \
		tests/pieces/makeindex.tex \
		tests/pieces/makeglossaries.tex \
		tests/pieces/begin-document.tex \
		tests/pieces/titulo.tex \
		tests/pieces/folha-de-rosto.tex \
		tests/pieces/cat-aprov.tex \
		tests/pieces/dedicatoria.tex \
		tests/pieces/agradecimentos.tex \
		tests/pieces/epigrafe.tex \
		tests/pieces/resumo.tex \
		tests/pieces/abstract.tex \
		tests/pieces/tocloftasa.tex \
		tests/pieces/first-chapter-title.tex \
		tests/pieces/first-chapter-text.tex \
		tests/pieces/document-text.tex \
		tests/pieces/printbibliography.tex \
		tests/pieces/printindex.tex \
		tests/pieces/end-document.tex > tests/test-full-template-backrefs.tex

	awk 'FNR==1{print ""}{print}' tests/pieces/documentclass-numeric-backrefs.tex \
		tests/pieces/inputenc-author-title.tex \
		tests/pieces/newsubfloatfigure.tex \
		tests/pieces/subtitulo.tex \
		tests/pieces/acronimos.tex \
		tests/pieces/simbolos.tex \
		tests/pieces/addbibresource.tex \
		tests/pieces/makeindex.tex \
		tests/pieces/makeglossaries.tex \
		tests/pieces/begin-document.tex \
		tests/pieces/titulo.tex \
		tests/pieces/folha-de-rosto.tex \
		tests/pieces/cat-aprov.tex \
		tests/pieces/dedicatoria.tex \
		tests/pieces/agradecimentos.tex \
		tests/pieces/epigrafe.tex \
		tests/pieces/resumo.tex \
		tests/pieces/abstract.tex \
		tests/pieces/tocloftasa.tex \
		tests/pieces/first-chapter-title.tex \
		tests/pieces/first-chapter-text.tex \
		tests/pieces/document-text.tex \
		tests/pieces/printbibliography.tex \
		tests/pieces/printindex.tex \
		tests/pieces/end-document.tex > tests/test-full-template-numeric-backrefs.tex

	awk 'FNR==1{print ""}{print}' tests/pieces/pdfa-xmpdata-filecontents.tex \
		tests/pieces/documentclass-nopdfa.tex \
		tests/pieces/inputenc-author-title.tex \
		tests/pieces/newsubfloatfigure.tex \
		tests/pieces/subtitulo.tex \
		tests/pieces/acronimos.tex \
		tests/pieces/simbolos.tex \
		tests/pieces/addbibresource.tex \
		tests/pieces/makeindex.tex \
		tests/pieces/makeglossaries.tex \
		tests/pieces/begin-document.tex \
		tests/pieces/titulo.tex \
		tests/pieces/folha-de-rosto.tex \
		tests/pieces/cat-aprov.tex \
		tests/pieces/dedicatoria.tex \
		tests/pieces/agradecimentos.tex \
		tests/pieces/epigrafe.tex \
		tests/pieces/resumo.tex \
		tests/pieces/abstract.tex \
		tests/pieces/tocloftasa.tex \
		tests/pieces/first-chapter-title.tex \
		tests/pieces/first-chapter-text.tex \
		tests/pieces/document-text.tex \
		tests/pieces/printbibliography.tex \
		tests/pieces/printindex.tex \
		tests/pieces/end-document.tex > tests/test-full-template-nopdfa.tex

	awk 'FNR==1{print ""}{print}' tests/pieces/pdfa-xmpdata-filecontents.tex \
		tests/pieces/documentclass-sublist.tex \
		tests/pieces/inputenc-author-title.tex \
		tests/pieces/newsubfloatfigure.tex \
		tests/pieces/subtitulo.tex \
		tests/pieces/acronimos-simbolos-sublist.tex \
		tests/pieces/addbibresource.tex \
		tests/pieces/makeindex.tex \
		tests/pieces/makeglossaries.tex \
		tests/pieces/begin-document.tex \
		tests/pieces/titulo.tex \
		tests/pieces/folha-de-rosto.tex \
		tests/pieces/cat-aprov.tex \
		tests/pieces/dedicatoria.tex \
		tests/pieces/agradecimentos.tex \
		tests/pieces/epigrafe.tex \
		tests/pieces/resumo.tex \
		tests/pieces/abstract.tex \
		tests/pieces/tocloftasa.tex \
		tests/pieces/first-chapter-title.tex \
		tests/pieces/first-chapter-text.tex \
		tests/pieces/document-text.tex \
		tests/pieces/printbibliography.tex \
		tests/pieces/printindex.tex \
		tests/pieces/end-document.tex > tests/test-full-template-sublist.tex

	awk 'FNR==1{print ""}{print}' tests/pieces/documentclass.tex \
		tests/pieces/inputenc-author-title.tex \
		tests/pieces/newsubfloatfigure.tex \
		tests/pieces/addbibresource.tex \
		tests/pieces/begin-document.tex \
		tests/pieces/titulo.tex \
		tests/pieces/folha-de-rosto.tex \
		tests/pieces/cat-aprov.tex \
		tests/pieces/resumo.tex \
		tests/pieces/abstract.tex \
		tests/pieces/tocloftasa.tex \
		tests/pieces/first-chapter-title.tex \
		tests/pieces/first-chapter-text.tex \
		tests/pieces/document-text.tex \
		tests/pieces/printbibliography.tex \
		tests/pieces/end-document.tex > tests/test-only-required.tex

	awk 'FNR==1{print ""}{print}' tests/pieces/documentclass.tex \
		tests/pieces/inputenc-author-title.tex \
		tests/pieces/begin-document.tex \
		tests/pieces/first-chapter-text.tex \
		tests/pieces/end-document.tex > tests/test-only-text.tex

	awk 'FNR==1{print ""}{print}' tests/pieces/documentclass.tex \
		tests/pieces/inputenc-author-title.tex \
		tests/pieces/newsubfloatfigure.tex \
		tests/pieces/addbibresource.tex \
		tests/pieces/begin-document.tex \
		tests/pieces/first-chapter-title.tex \
		tests/pieces/first-chapter-text.tex \
		tests/pieces/document-text.tex \
		tests/pieces/printbibliography.tex \
		tests/pieces/end-document.tex > tests/test-only-text-and-titles.tex

	cp $(NAME).cls referencias.bib tests	
	latexmk -pdf tests/test-full-template.tex
	latexmk -pdf tests/test-full-template-numeric.tex
	latexmk -pdf tests/test-full-template-backrefs.tex
	latexmk -pdf tests/test-full-template-numeric-backrefs.tex
	latexmk -pdf tests/test-full-template-nopdfa.tex
	latexmk -pdf tests/test-full-template-sublist.tex
	latexmk -pdf tests/test-only-required.tex
	latexmk -pdf tests/test-only-text.tex
	latexmk -pdf tests/test-only-text-and-titles.tex
	rm tests/referencias.bib tests/$(NAME).cls
