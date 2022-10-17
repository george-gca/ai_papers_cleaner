# Created by: George Corrêa de Araújo (george.gcac@gmail.com)

# ==================================================================
# environment variables
# ------------------------------------------------------------------

NLTK_DATA = $(HOME)/nltk_data
DOCKERFILE_CONTEXT = $(PWD)
DOCKERFILE = $(PWD)/Dockerfile
WORK_DIR = $(PWD)
RUN_STRING = bash start_here.sh

# ==================================================================
# Docker settings
# ------------------------------------------------------------------

CONTAINER_NAME = cleaner-$(USER)-$(shell echo $$STY | cut -d'.' -f2) # use gnu screen name when creating container
CONTAINER_FILE = cleaner-$(USER).tar
HOSTNAME = docker-$(shell hostname)
IMAGE_NAME = $(USER)/text_cleaner
WORK_PATH = /work

NLTK_MOUNT_STRING = --mount type=bind,source=$(NLTK_DATA),target=/home/$(USER)/nltk_data
PDB_MOUNT_STRING = --mount type=bind,source=/home/$(USER)/.pdbhistory,target=/home/$(USER)/.pdbhistory
RUN_CONFIG_STRING = --name $(CONTAINER_NAME) --hostname $(HOSTNAME) --rm -it --dns 8.8.8.8 \
	--userns=host --ipc=host --ulimit memlock=-1 -w $(WORK_PATH) $(IMAGE_NAME):latest
WORK_MOUNT_STRING = --mount type=bind,source=$(WORK_DIR),target=$(WORK_PATH)

# ==================================================================
# Make commands
# ------------------------------------------------------------------

# Build image
build:
	docker build \
		--build-arg GROUPID=$(shell id -g) \
		--build-arg GROUPNAME=$(shell id -gn) \
		--build-arg USERID=$(shell id -u) \
		--build-arg USERNAME=$(USER) \
		-f $(DOCKERFILE) \
		--pull --no-cache --force-rm \
		-t $(IMAGE_NAME) \
		$(DOCKERFILE_CONTEXT)


# Remove the image
clean:
	docker rmi $(IMAGE_NAME)


# Load image from file
load:
	docker load -i $(CONTAINER_FILE)


# Kill running container
kill:
	docker kill $(CONTAINER_NAME)


# Run RUN_STRING inside container
run:
	docker run \
		$(WORK_MOUNT_STRING) \
		$(NLTK_MOUNT_STRING) \
		$(PDB_MOUNT_STRING) \
		$(RUN_CONFIG_STRING) \
		$(RUN_STRING)


# Save image to file
save:
	docker save -o $(CONTAINER_FILE) $(IMAGE_NAME)


# Start container by opening shell
start:
	docker run \
		$(WORK_MOUNT_STRING) \
		$(NLTK_MOUNT_STRING) \
		$(PDB_MOUNT_STRING) \
		$(RUN_CONFIG_STRING)


# Test image by printing some info
test:
	nvidia-docker run \
		$(RUN_CONFIG_STRING) \
		python -V

