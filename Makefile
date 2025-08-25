PY=python
PKG=xtc

format:
	black . && isort .

lint:
	ruff .

test:
	pytest -q

train:
	$(PY) -m $(PKG).train --epochs 1 --batch_size 16

docker-build:
	docker build -t xtc:latest .

docker-run:
	docker run -p 8000:8000 xtc:latest
