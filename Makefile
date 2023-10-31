.DEFAULT_GOAL := help

help:
	@echo "    prepare              desc of the command prepare"
	@echo "    install              desc of the command install"


install:
	@echo "Installing..."
	python -m pip install -r requirements.txt

activate:
	@echo "Activating virtual environment"
	source env/bin/activate

export:
	@echo "Exporting dependencies to requirements file"
	python -m pip freeze > requirements.txt

backup: # To push to Github without running precommit
	git commit --no-verify -m "backup"
	git push origin
