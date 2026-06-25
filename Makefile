NAME=custom-rl-environment-for-llm-safety
PYVERSION=3.11

app-version:
	echo "COMMIT_ID=${COMMIT_ID}"

venv:
	uv venv /tmp/venv/${NAME} --python ${PYVERSION}

source-env:
	echo source /tmp/venv/${NAME}/bin/activate

deactivate:
	deactivate

install:
	uv pip install -r requirements.txt

build-docker-image:
	export DOCKER_BUILDKIT=1 && \
	docker image build \
	-t ${NAME}:${COMMIT_ID} . 

run:
	python src/main.py
