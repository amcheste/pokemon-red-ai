# Build targets for the EWRL 2026 paper.
#
# Requires a TeX distribution with pdflatex + bibtex.  On macOS:
#   brew install --cask mactex-no-gui     # full
#   brew install --cask basictex          # smaller
#   eval "$(/usr/libexec/path_helper)"
#
# Targets:
#   make pdf       — full build (pdflatex → bibtex → pdflatex × 2)
#   make quick     — single pdflatex pass (refs may be stale)
#   make clean     — remove all build artefacts (keeps PDF)
#   make distclean — remove everything including the PDF

PAPER     := main
PDFLATEX  := pdflatex -interaction=nonstopmode -halt-on-error -file-line-error
BIBTEX    := bibtex

.PHONY: pdf quick clean distclean help

help:
	@echo "Available targets:"
	@echo "  make pdf        Full build: pdflatex → bibtex → pdflatex × 2"
	@echo "  make quick      Single pdflatex pass (for fast iteration)"
	@echo "  make clean      Remove build artefacts (keep PDF)"
	@echo "  make distclean  Remove everything (including PDF)"

pdf: $(PAPER).pdf

# Full citation-resolving build — required after editing references.bib
# or adding new \cite{} commands.
$(PAPER).pdf: $(PAPER).tex references.bib $(wildcard sections/*.tex)
	$(PDFLATEX) $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(PDFLATEX) $(PAPER).tex
	$(PDFLATEX) $(PAPER).tex

# Fast single-pass build for iterating on prose without changing
# citations or section structure.
quick:
	$(PDFLATEX) $(PAPER).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz
	rm -f sections/*.aux

distclean: clean
	rm -f $(PAPER).pdf
