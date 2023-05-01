doc:
	sphinx-apidoc -f -o ./rst/apidoc ./acryo
	sphinx-build -b html ./rst ./docs

watch-rst:
	watchfiles "sphinx-build -b html ./rst ./_docs_temp" rst
