book:
	jupyter-book build ./
	open ./_build/html/index.html

pdf:
	jupyter-book build ./ --builder pdflatex
	open _build/latex/book.pdf
	
website: book
	ghp-import -n -p -f _build/html
	open https://tristanvanleeuwen.github.io/IP_and_Im_Lectures/intro.html

clean:
	jupyter-book clean ./
