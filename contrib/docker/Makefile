# The orangu stack in one container image: orangu-coordinator in front of
# orangu-server, with the models from `orangu-server list` it serves.
#
#   make list                     # the models on this machine, and their labels
#   make CONFIG=coordinator.conf  # build every profile in a coordinator config
#   make run                      # start it on http://localhost:9000/v1
#
# See README.md, or the Container image chapter of the manual.

# The orangu-coordinator.conf the image runs: every profile in it goes in.
# Unset, the coordinator's own config is used (./orangu-coordinator.conf,
# then ~/.orangu/orangu-coordinator.conf).
CONFIG ?=
# The base image: rocky10, debian12 or archlinux (Dockerfile.<DISTRO>).
DISTRO ?= rocky10
# The image name.
IMAGE ?= orangu:$(DISTRO)
# podman, docker, or auto: podman if it answers, else docker.
ENGINE ?= auto
# e.g. linux/arm64; empty builds for this machine.
PLATFORM ?=
# The host port `make run` publishes the coordinator on.
PORT ?= 9000
# The orangu-server used to resolve models.
ORANGU_SERVER ?= orangu-server

# The recipes are POSIX sh. GNU make on Windows uses sh.exe from PATH, and
# falls back to cmd.exe — reporting SHELL as a bare `sh.exe` — when there is
# none, which is the case in PowerShell and cmd. There, use the sh that ships
# with Git for Windows, found from git.exe on PATH (Git\cmd\..\bin\sh.exe).
ifeq ($(OS)$(SHELL),Windows_NTsh.exe)
GIT_SH := $(shell for %%G in (git.exe) do @if exist "%%~dp$$PATH:G..\bin\sh.exe" echo %%~dp$$PATH:G..\bin\sh.exe)
ifeq ($(GIT_SH),)
$(error no POSIX sh found: install Git for Windows, or run make from Git Bash or WSL)
endif
SHELL := $(subst \,/,$(GIT_SH))
endif

ROOT := $(abspath $(CURDIR)/../..)
STAGE := $(CURDIR)/build
COMMIT := $(shell git -C "$(ROOT)" rev-parse --short HEAD 2>/dev/null)
DOCKERFILE := $(CURDIR)/Dockerfile.$(DISTRO)
DISTROS := $(patsubst Dockerfile.%,%,$(filter-out %.dockerignore,$(notdir $(wildcard $(CURDIR)/Dockerfile.*))))

# The engine, asked for once and only by the targets that need one, so
# `make list` works with neither installed. Podman first: it needs no daemon.
ifeq ($(ENGINE),auto)
ENGINE_CMD = $(eval ENGINE_CMD := $$(shell podman version >/dev/null 2>&1 && echo podman || { docker version >/dev/null 2>&1 && echo docker; }))$(ENGINE_CMD)
else
ENGINE_CMD = $(ENGINE)
endif

# Docker reads Dockerfile.<DISTRO>.dockerignore next to the Dockerfile on its
# own; Podman has to be told.
IGNOREFILE = $(if $(filter podman,$(ENGINE_CMD)),--ignorefile "$(DOCKERFILE).dockerignore")

.PHONY: image help list stage check run clean
.DEFAULT_GOAL := image

help:
	@sed -n '1,8p' Makefile | sed 's/^# \{0,1\}//'

list:
	$(ORANGU_SERVER) list

# Fails before anything is staged: copying gigabytes of models only to find
# there is no engine, or no such distro, is the slow way to find out.
check:
	@if [ ! -f "$(DOCKERFILE)" ]; then \
	    echo "error: no Dockerfile.$(DISTRO); DISTRO is one of: $(DISTROS)" >&2; exit 1; \
	fi
	@if [ "$(DISTRO)" = archlinux ] && [ -n "$(filter-out linux/amd64,$(PLATFORM))" ]; then \
	    echo "error: the archlinux base image is linux/amd64 only" >&2; exit 1; \
	fi
	@if [ -z "$(ENGINE_CMD)" ]; then \
	    echo "error: neither podman nor docker answers; install or start one (podman machine start, or Docker Desktop)" >&2; \
	    exit 1; \
	fi

# Copies the models into build/models and writes the coordinator config for
# them; see stage.sh.
stage:
	@CONFIG='$(CONFIG)' ORANGU_SERVER='$(ORANGU_SERVER)' sh "$(CURDIR)/stage.sh" "$(STAGE)"

image: check stage
	@echo "Image      $(IMAGE) ($(DISTRO), $(ENGINE_CMD))"
	$(ENGINE_CMD) build \
	    $(if $(PLATFORM),--platform $(PLATFORM)) \
	    -f "$(DOCKERFILE)" $(IGNOREFILE) \
	    --build-context stage="$(STAGE)" \
	    --build-arg ORANGU_BUILD_COMMIT=$(or $(COMMIT),unknown) \
	    -t $(IMAGE) \
	    "$(ROOT)"
	@rm -rf "$(STAGE)"
	@echo "Built $(IMAGE). Start it with: make run$(if $(filter-out rocky10,$(DISTRO)), DISTRO=$(DISTRO))"

run:
	$(ENGINE_CMD) run --rm -it -p $(PORT):9000 -v orangu-data:/home/orangu/.orangu $(IMAGE)

clean:
	rm -rf "$(STAGE)"
