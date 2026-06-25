NAME=custom-rl-environment-for-llm-safety

app-version:
	echo "COMMIT_ID=${COMMIT_ID}"

venv:
	uv venv --python 3.12.0 /tmp/venv/${NAME}

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
