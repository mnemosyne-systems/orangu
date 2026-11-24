# `make install` installs orangu for Emacs into ~/.emacs.d/elpa/orangu, in
# place of any version already there. `make uninstall` removes it, and
# `make clean` removes what a build leaves in this directory.

EMACS ?= emacs

ELPA   := $(HOME)/.emacs.d/elpa
TARGET := $(ELPA)/orangu
BUILD  := $(CURDIR)/.build

.PHONY: install uninstall clean

install: uninstall clean
	$(EMACS) -Q --batch -l package \
		--eval '(setq package-user-dir "$(BUILD)")' \
		-f package-initialize \
		--eval '(package-install-file "$(CURDIR)")'
	mkdir -p "$(ELPA)"
	mv "$(BUILD)"/orangu-[0-9]* "$(TARGET)"
	rm -rf "$(BUILD)"

uninstall:
	rm -rf "$(TARGET)" "$(ELPA)"/orangu-[0-9]*

clean:
	rm -rf "$(BUILD)" *.elc orangu-autoloads.el orangu-pkg.el
