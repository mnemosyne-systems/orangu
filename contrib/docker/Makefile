# The orangu stack in one container image: orangu-coordinator in front of
# orangu-server, serving one model from `orangu-server list`.
#
#   make list                  # the models on this machine, with their NR
#   make MODEL=1               # build orangu:latest with model NR 1
#   make run                   # start it on http://localhost:9000/v1
#
# See README.md for the variables.

# The model: an NR or MODEL from `orangu-server list`, resolved by the
# orangu-server on this machine. MODEL_FILES instead names the .gguf file(s)
# directly (space-separated, every shard of a split model).
MODEL ?=
MODEL_FILES ?=
# The image name.
IMAGE ?= orangu:latest
# docker or podman.
ENGINE ?= docker
# e.g. linux/arm64; empty builds for this machine.
PLATFORM ?=
# The host port `make run` publishes the coordinator on.
PORT ?= 9000
# The orangu-server used to resolve MODEL.
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

# Docker reads Dockerfile.dockerignore next to the Dockerfile on its own;
# Podman has to be told.
IGNOREFILE := $(if $(filter podman,$(ENGINE)),--ignorefile "$(CURDIR)/Dockerfile.dockerignore")

.PHONY: image help list stage run clean
.DEFAULT_GOAL := image

help:
	@sed -n '1,8p' Makefile | sed 's/^# \{0,1\}//'

list:
	$(ORANGU_SERVER) list

# Copies the model into build/models — a hard link where the filesystem
# allows one, so a multi-gigabyte model is not copied twice — and writes the
# coordinator config naming it. A multi-token-prediction draft head that
# `download` put in MTP/ beside the model goes into models/MTP/, where the
# server looks for it; without it the image serves correctly, only slower.
stage:
	@if [ -z "$(MODEL)$(MODEL_FILES)" ]; then \
	    echo "error: set MODEL to an NR or MODEL from 'orangu-server list' (see 'make list')" >&2; \
	    exit 1; \
	fi
	@rm -rf "$(STAGE)" && mkdir -p "$(STAGE)/models"
	@set -e; \
	if [ -n "$(MODEL_FILES)" ]; then \
	    printf '%s\n' $(foreach file,$(MODEL_FILES),'$(file)'); \
	else \
	    $(ORANGU_SERVER) show "$(MODEL)" --path; \
	fi > "$(STAGE)/files.txt"; \
	first=; source=; \
	while IFS= read -r file; do \
	    file=$$(printf '%s' "$$file" | tr -d '\r' | tr '\134' '/'); \
	    [ -n "$$file" ] || continue; \
	    name=$$(basename "$$file"); \
	    echo "Model      $$file"; \
	    ln -L "$$file" "$(STAGE)/models/$$name" 2>/dev/null \
	        || cp -L "$$file" "$(STAGE)/models/$$name"; \
	    [ -n "$$first" ] || { first=$$name; source=$$(dirname "$$file"); }; \
	done < "$(STAGE)/files.txt"; \
	rm -f "$(STAGE)/files.txt"; \
	if [ -z "$$first" ]; then echo "error: no model files to copy" >&2; exit 1; fi; \
	for heads in "$$source/MTP" "$$source/../MTP"; do \
	    ls "$$heads"/mtp-*.gguf >/dev/null 2>&1 || continue; \
	    mkdir -p "$(STAGE)/models/MTP"; \
	    for head in "$$heads"/mtp-*.gguf; do \
	        echo "MTP head   $$head"; \
	        ln -L "$$head" "$(STAGE)/models/MTP/" 2>/dev/null \
	            || cp -L "$$head" "$(STAGE)/models/MTP/"; \
	    done; \
	    break; \
	done; \
	sed "s|@MODEL@|/opt/orangu/models/$$first|" orangu-coordinator.conf.in \
	    > "$(STAGE)/orangu-coordinator.conf"

image: stage
	$(ENGINE) build \
	    $(if $(PLATFORM),--platform $(PLATFORM)) \
	    -f "$(CURDIR)/Dockerfile" $(IGNOREFILE) \
	    --build-context stage="$(STAGE)" \
	    --build-arg ORANGU_BUILD_COMMIT=$(or $(COMMIT),unknown) \
	    -t $(IMAGE) \
	    "$(ROOT)"
	@rm -rf "$(STAGE)"
	@echo "Built $(IMAGE). Start it with: make run"

run:
	$(ENGINE) run --rm -it -p $(PORT):9000 -v orangu-data:/home/orangu/.orangu $(IMAGE)

clean:
	rm -rf "$(STAGE)"
