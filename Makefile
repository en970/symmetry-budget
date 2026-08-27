.PHONY: grid status submit collect audit report validate clean

grid:      ## expand PROTOCOL.md into an explicit cell list
	python3 tools/grid.py

status:    ## grid progress (counts only — metrics locked until a phase closes)
	python3 tools/status.py

validate:  ## check every result file against the protocol contract
	python3 tools/validate_result.py

submit:    ## dispatch pending cells for the open phase to Kaggle
	python3 tools/submit.py

collect:   ## pull finished Kaggle runs into results/
	python3 tools/collect.py

report:    ## regenerate reports/ from results/ (never hand-edited)
	python3 tools/report.py

clean:
	rm -rf __pycache__ src/__pycache__ tools/__pycache__
