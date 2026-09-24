# Convenience targets. The build steps themselves live in run_all.py and
# report/build_report.sh so that CI and a developer run exactly the same thing.
.PHONY: help report report-check explainers video test lint quick full clean-report

PYTHON ?= .venv/bin/python
PAGES_MIN ?= 15
PAGES_MAX ?= 20

help:
	@echo "make report        build report/TECHNICAL_REPORT.pdf"
	@echo "make report-check  build it and require $(PAGES_MIN)-$(PAGES_MAX) pages"
	@echo "make video         build outputs/video/neuromps_demo.mp4"
	@echo "make quick         python run_all.py --quick"
	@echo "make full          python run_all.py"
	@echo "make explainers    build the two plain-language companion PDFs"
	@echo "make test / lint   pytest / ruff"

report:
	report/build_report.sh

report-check:
	report/build_report.sh --check-pages $(PAGES_MIN) $(PAGES_MAX)

# The two plain-language companions. Gitignored, so this target does nothing on a
# clone that does not have them; it is here so the invocation does not have to be
# remembered.
explainers:
	@test -f report/MODELLING_FOR_BIOLOGISTS.md \
	  && report/build_report.sh --source report/MODELLING_FOR_BIOLOGISTS.md \
	       --title "Prediction models, explained" \
	       --subtitle "A companion to the NeuroMPS technical report, for readers without a machine-learning background" \
	  || echo "report/MODELLING_FOR_BIOLOGISTS.md not present; skipped"
	@test -f report/DIGITAL_TWIN_FOR_BIOLOGISTS.md \
	  && report/build_report.sh --source report/DIGITAL_TWIN_FOR_BIOLOGISTS.md \
	       --title "The digital twin, explained" \
	       --subtitle "A companion to the NeuroMPS technical report, for readers without a computational-modelling background" \
	  || echo "report/DIGITAL_TWIN_FOR_BIOLOGISTS.md not present; skipped"
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
