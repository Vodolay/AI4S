# Convenience targets. The build steps themselves live in run_all.py and
# report/build_report.sh so that CI and a developer run exactly the same thing.
.PHONY: help report report-check video test lint quick full clean-report

PYTHON ?= .venv/bin/python
PAGES_MIN ?= 14
PAGES_MAX ?= 22

help:
	@echo "make report        build report/TECHNICAL_REPORT.pdf"
	@echo "make report-check  build it and require $(PAGES_MIN)-$(PAGES_MAX) pages"
	@echo "make video         build outputs/video/neuromps_demo.mp4"
	@echo "make quick         python run_all.py --quick"
	@echo "make full          python run_all.py"
	@echo "make test / lint   pytest / ruff"

report:
	report/build_report.sh

report-check:
	report/build_report.sh --check-pages $(PAGES_MIN) $(PAGES_MAX)

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
