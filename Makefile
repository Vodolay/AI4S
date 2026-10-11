# Convenience targets. The build steps themselves live in run_all.py and
# report/build_docx.sh so that CI and a developer run exactly the same thing.
.PHONY: help report report-check report-latex notebook test lint quick full clean-report

# The project's virtualenv when there is one, otherwise whatever interpreter is
# on the path. CI has no .venv and a developer does, and hardcoding either meant
# the two ran different commands - which is how formatting drifted unnoticed.
PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || command -v python3 || command -v python)
PAGES_MIN ?= 15
PAGES_MAX ?= 20

help:
	@echo "make report        build report/TECHNICAL_REPORT.pdf"
	@echo "make report-check  build it and require $(PAGES_MIN)-$(PAGES_MAX) pages"
	@echo "make quick         python run_all.py --quick"
	@echo "make full          python run_all.py"
	@echo "make notebook      write the Kaggle notebook and stage its dataset"
	@echo "make test / lint   pytest / ruff"

# The technical report is the Word build: it carries the table design a reviewer
# asked for, and the PDF is LibreOffice's rendering of that .docx, so the two
# cannot drift apart. The page range is checked on that PDF, here and in CI,
# because it is the one submitted. build_report.sh is the LaTeX path:
# report-latex renders the technical report through it for comparison but
# does not gate its length - its wider margins and 11 pt
# serif set the same text several pages longer, so its count says nothing about
# the submission.
report:
	report/build_docx.sh

report-check:
	report/build_docx.sh --check-pages $(PAGES_MIN) $(PAGES_MAX)

report-latex:
	report/build_report.sh --output report/TECHNICAL_REPORT_latex.pdf

notebook:
	$(PYTHON) -m neuromps.demo.notebook --dataset-dir outputs/demo/kaggle_dataset

quick:
	$(PYTHON) run_all.py --quick

full:
	$(PYTHON) run_all.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

clean-report:
	rm -f report/TECHNICAL_REPORT.pdf
