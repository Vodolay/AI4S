# Convenience targets. The build steps themselves live in run_all.py and
# report/build_report.sh so that CI and a developer run exactly the same thing.
.PHONY: help report report-check report-latex explainers notebook video test lint quick full clean-report

# The project's virtualenv when there is one, otherwise whatever interpreter is
# on the path. CI has no .venv and a developer does, and hardcoding either meant
# the two ran different commands - which is how formatting drifted unnoticed.
PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || command -v python3 || command -v python)
PAGES_MIN ?= 15
PAGES_MAX ?= 20

help:
	@echo "make report        build report/TECHNICAL_REPORT.pdf"
	@echo "make report-check  build it and require $(PAGES_MIN)-$(PAGES_MAX) pages"
	@echo "make video         build outputs/video/neuromps_demo.mp4"
	@echo "make quick         python run_all.py --quick"
	@echo "make full          python run_all.py"
	@echo "make explainers    build the plain-language companion PDFs"
	@echo "make notebook      write the Kaggle notebook and stage its dataset"
	@echo "make test / lint   pytest / ruff"

# The technical report is the Word build: it carries the table design a reviewer
# asked for, and the PDF is LibreOffice's rendering of that .docx, so the two
# cannot drift apart. build_report.sh is the LaTeX path, kept for the companions
# below and as a second opinion; it writes to the SAME TECHNICAL_REPORT.pdf, so
# do not run it expecting the Word one to survive.
report:
	report/build_docx.sh

report-check:
	report/build_docx.sh --check-pages $(PAGES_MIN) $(PAGES_MAX)

report-latex:
	report/build_report.sh --output report/TECHNICAL_REPORT_latex.pdf \
	  --check-pages $(PAGES_MIN) $(PAGES_MAX)

# The two plain-language companions. Gitignored, so this target does nothing on a
# clone that does not have them; it is here so the invocation does not have to be
# remembered.
notebook:
	$(PYTHON) -m neuromps.demo.notebook --dataset-dir outputs/demo/kaggle_dataset

explainers:
	@test -f report/MODELLING_FOR_BIOLOGISTS.md \
	  && report/build_report.sh --source report/MODELLING_FOR_BIOLOGISTS.md \
	       --title "Prediction models, explained" \
	       --subtitle "A companion to the NeuroMPS technical report, for readers without a machine-learning background" \
	  || echo "report/MODELLING_FOR_BIOLOGISTS.md not present; skipped"
	@test -f report/APP_WALKTHROUGH.md \
	  && report/build_report.sh --source report/APP_WALKTHROUGH.md \
	       --title "The demo app, a walkthrough" \
	       --subtitle "Running the NeuroMPS demo, what every control does, and what to look at" \
	  || echo "report/APP_WALKTHROUGH.md not present; skipped"

video:
	$(PYTHON) -m neuromps.video.build

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
