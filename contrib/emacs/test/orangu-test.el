;;; orangu-test.el --- Tests of orangu for Emacs  -*- lexical-binding: t; -*-

;; Copyright (C) 2026 The orangu community
;;
;; This program is free software: you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
;; GNU General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with this program. If not, see <https://www.gnu.org/licenses/>.

;;; Commentary:

;; `make test' runs these.  They need no orangu: the runs start a shell
;; script that stands in for it, and are skipped where there is no sh.

;;; Code:

(require 'ert)
(require 'orangu)

(defconst orangu-test--directory
  (file-name-directory (or load-file-name buffer-file-name))
  "The folder of the tests.")

(defconst orangu-test--fake (expand-file-name "fake-orangu.sh" orangu-test--directory)
  "The script that stands in for orangu.")

(defmacro orangu-test--with-folder (name &rest body)
  "Run BODY with NAME bound to a new folder, removed afterwards."
  (declare (indent 1))
  `(let ((,name (make-temp-file "orangu-test-" t)))
     (unwind-protect (progn ,@body)
       (delete-directory ,name t))))

;;; Escapes

(ert-deftest orangu-test-filter-plain ()
  "Plain text passes through, without carriage returns."
  (should (equal (funcall (orangu-run-make-filter) "one\r\ntwo\n") "one\ntwo\n")))

(ert-deftest orangu-test-filter-colours ()
  "Colours and cursor moves are removed."
  (should (equal (funcall (orangu-run-make-filter) "\e[1;31mred\e[0m and \e[2Kplain")
                 "red and plain")))

(ert-deftest orangu-test-filter-cut ()
  "A sequence cut between two pieces is held back until it ends."
  (let ((filter (orangu-run-make-filter)))
    (should (equal (funcall filter "a\e") "a"))
    (should (equal (funcall filter "[3") ""))
    (should (equal (funcall filter "2mb") "b"))))

(ert-deftest orangu-test-filter-titles ()
  "Titles and links end at a bell or at ESC backslash."
  (let ((filter (orangu-run-make-filter)))
    (should (equal (funcall filter "\e]0;title\ax") "x"))
    (should (equal (funcall filter "\e]8;;https://example.org\e\\link\e]8;;\e\\") "link"))))

(ert-deftest orangu-test-filter-two-characters ()
  "A two-character sequence is dropped whole."
  (should (equal (funcall (orangu-run-make-filter) "a\e7b\e8c") "abc")))

;;; Command lines

(ert-deftest orangu-test-command-lines ()
  "A one-shot is -w and -p; spaces, quotes and new lines stay as they are."
  (should (equal (orangu-core-one-shot "/usr/bin/orangu" "/work/my project" "what does \"x\" do?\nand y")
                 '("/usr/bin/orangu" "-w" "/work/my project" "-p" "what does \"x\" do?\nand y")))
  (should (equal (orangu-core-interactive "/usr/bin/orangu" "/work/app")
                 '("/usr/bin/orangu" "-w" "/work/app")))
  (should (equal (orangu-core-plain "/usr/bin/orangu" "/work/app")
                 '("/usr/bin/orangu" "-w" "/work/app" "--plain"))))

(ert-deftest orangu-test-plain-input ()
  "Each line of a prompt but the last ends in \\, and a last \\ is kept."
  (should (equal (orangu-core-plain-input "one line\n\n") "one line"))
  (should (equal (orangu-core-plain-input "first\nsecond\nthird") "first\\\nsecond\\\nthird"))
  (should (equal (orangu-core-plain-input "ends in \\") "ends in \\ ")))

;;; Finding orangu and the workspace

(ert-deftest orangu-test-locate ()
  "`orangu-path' may name the binary or its folder, and must name one."
  (orangu-test--with-folder folder
    (let ((binary (expand-file-name (orangu-core-binary-name) folder)))
      (with-temp-file binary (insert "#!/bin/sh\n"))
      (set-file-modes binary #o755)
      (let ((orangu-path binary))
        (should (equal (orangu-core-locate) binary)))
      (let ((orangu-path folder))
        (should (equal (orangu-core-locate) binary)))
      (let ((orangu-path (expand-file-name "missing" folder)))
        (should-error (orangu-core-locate) :type 'user-error)))))

(ert-deftest orangu-test-workspace ()
  "The workspace is the repository a folder is in, without a trailing slash."
  (orangu-test--with-folder folder
    (let ((inner (expand-file-name "src/deep" folder)))
      (make-directory (expand-file-name ".git" folder))
      (make-directory inner t)
      (should (equal (file-truename (orangu-core-workspace inner))
                     (file-truename (directory-file-name folder))))
      (should (equal (orangu-core-workspace folder) (directory-file-name (expand-file-name folder)))))))

;;; Prompts

(cl-defun orangu-test--selection (&key (file "/work/app/src/parser.rs") (start 12) (end 14)
                                        (text "fn parse() {
    todo!()
}
"))
  "A selection: lines 12-14 of src/parser.rs in /work/app, unless FILE, START, END or TEXT say otherwise."
  (orangu-selection-create :file file :start start :end end :text text))

(ert-deftest orangu-test-about-selection ()
  "A question about a selection carries the file, the lines and the code."
  (should (equal (orangu-core-about-selection "/work/app" (orangu-test--selection) "  Why is this slow?  ")
                 (concat "Why is this slow?\n\nThis is lines 12-14 of src/parser.rs:\n\n"
                         "```rust\nfn parse() {\n    todo!()\n}\n```"))))

(ert-deftest orangu-test-about-one-line ()
  "An empty question asks what the code does, and one line says line."
  (should (equal (orangu-core-about-selection
                  "/work/app" (orangu-test--selection :end 12 :text "let x = 1;") " ")
                 (concat "Explain what this code does.\n\nThis is line 12 of src/parser.rs:\n\n"
                         "```rust\nlet x = 1;\n```"))))

(ert-deftest orangu-test-about-position ()
  "A position without text says where the cursor is."
  (should (equal (orangu-core-about-selection
                  "/work/app" (orangu-test--selection :end 12 :text nil) "What is here?")
                 "What is here?\n\nThe cursor is at line 12 of src/parser.rs.")))

(ert-deftest orangu-test-fence ()
  "The fence is longer than any backticks in the code, and has a language only when known."
  (should (string-suffix-p "\n`````markdown\n````\ncode\n````\n`````"
                           (orangu-core-about-selection
                            "/work/app"
                            (orangu-test--selection :file "/work/app/README.md" :text "````\ncode\n````")
                            "q")))
  (should (string-suffix-p "\n```\nx\n```"
                           (orangu-core-about-selection
                            "/work/app" (orangu-test--selection :file "/work/app/notes.txt" :text "x") "q"))))

(ert-deftest orangu-test-implement ()
  "An instruction always starts the same way, so it is never taken for a command."
  (let ((prefix "Implement this in the workspace, editing its files with your tools:\n\n"))
    (should (equal (orangu-core-implement "/work/app" nil "  edit parser.rs \n")
                   (concat prefix "edit parser.rs")))
    (should (equal (orangu-core-implement "/work/app" (orangu-test--selection :end 12 :text nil) "fix it")
                   (concat prefix "fix it\n\nThe cursor is at line 12 of src/parser.rs.")))))

(ert-deftest orangu-test-label ()
  "The label is the short path and the lines."
  (should (equal (orangu-core-selection-label "/work/app" (orangu-test--selection)) "src/parser.rs:12-14"))
  (should (equal (orangu-core-selection-label "/work/app" (orangu-test--selection :end 12)) "src/parser.rs:12")))

(ert-deftest orangu-test-display-path ()
  "Paths are relative to the workspace, or whole when outside it."
  (should (equal (orangu-core-display-path "/work/app" "/work/app/a/b.el") "a/b.el"))
  (should (equal (orangu-core-display-path "/work/app/" "/work/app/a/b.el") "a/b.el"))
  (should (equal (orangu-core-display-path "/work/app" "/work/other/b.el") (expand-file-name "/work/other/b.el")))
  (should (equal (orangu-core-display-path "/work/app" "/work/application/b.el")
                 (expand-file-name "/work/application/b.el"))))

(ert-deftest orangu-test-review-file ()
  "A file review is /auto_review of that file, and there is none outside the workspace."
  (should (equal (orangu-core-review-file "/work/app" "/work/app/src/parser.rs") "/auto_review src/parser.rs"))
  (should (equal (orangu-core-review-file "/work/app" "/work/app/my notes.md") "/auto_review my?notes.md"))
  (should-error (orangu-core-review-file "/work/app" "/tmp/scratch.rs") :type 'user-error))

(ert-deftest orangu-test-review-argument ()
  "Glob characters are matched literally and spaces by ?, so a path stays one argument."
  (should (equal (orangu-core-review-argument "src/main.rs") "src/main.rs"))
  ;; `]' and `}' alone do not make orangu read a pattern.
  (should (equal (orangu-core-review-argument "notes/a]b}.md") "notes/a]b}.md"))
  (should (equal (orangu-core-review-argument "app/[id]/page.tsx") "app/[[]id[]]/page.tsx"))
  (should (equal (orangu-core-review-argument "src/what?.rs") "src/what[?].rs"))
  (should (equal (orangu-core-review-argument "src/{a,b}.rs") "src/[{]a,b[}].rs"))
  (should (equal (orangu-core-review-argument "src/star*.rs") "src/star[*].rs"))
  (should (equal (orangu-core-review-argument "docs/my notes.md") "docs/my?notes.md"))
  (dolist (file '("a b/c d.rs" "tab\there.rs" "[id] x/{y}.ts"))
    (should (= (length (split-string (orangu-core-review-argument file) "[[:space:]]+")) 1))))

;;; What the panel says

(ert-deftest orangu-test-end-status ()
  "The status says how a run went."
  (should (equal (orangu-panel-end-status '(:outcome succeeded :elapsed 1.54)) "Done in 1.5s"))
  (should (equal (orangu-panel-end-status '(:outcome cancelled :elapsed 3)) "Stopped"))
  (should (equal (orangu-panel-end-status '(:outcome failed :error "no server")) "orangu failed: no server"))
  (should (equal (orangu-changes-summary '(("a" 1 0))) "1 file changed"))
  (should (equal (orangu-changes-summary '(("a" 1 0) ("b" nil nil))) "2 files changed")))

;;; Runs

(defun orangu-test--run (prompt &optional on-note)
  "Run the stand-in with PROMPT, and return what it handed over once it ended.
That is a plist of :result, :answer and :notes.  ON-NOTE, if given, is
called with the run and each note as well."
  (let* ((answer "") (notes nil) (result nil) (run nil)
         (workspace (directory-file-name temporary-file-directory)))
    (setq run (orangu-run-start
               (cons "sh" (orangu-core-one-shot orangu-test--fake workspace prompt))
               workspace
               :on-answer (lambda (text) (setq answer (concat answer text)))
               :on-note (lambda (note)
                          (push note notes)
                          (when on-note (funcall on-note run note)))
               :on-end (lambda (end) (setq result end))))
    (with-timeout (15 (orangu-run-stop run) (error "The run did not end"))
      (while (not result)
        (accept-process-output nil 0.05)))
    (list :result result :answer answer :notes (nreverse notes))))

(ert-deftest orangu-test-run-command ()
  "orangu gets -w and -p, runs in the workspace without colour, and stdin is closed."
  (skip-unless (executable-find "sh"))
  (let* ((run (orangu-test--run "what is \"this\"?\nsecond line"))
         (answer (plist-get run :answer))
         (workspace (directory-file-name temporary-file-directory)))
    (should (eq (plist-get (plist-get run :result) :outcome) 'succeeded))
    (should (string-match-p (regexp-quote (format "args: [-w] [%s] [-p] [what is \"this\"?\nsecond line]\n"
                                                  workspace))
                            answer))
    (should (string-match "^cwd: \\(.*\\)$" answer))
    (should (equal (file-truename (match-string 1 answer)) (file-truename workspace)))
    (should (string-match-p "^NO_COLOR: 1$" answer))
    (should (string-match-p "^stdin: closed$" answer))
    ;; Notes come line by line, without their escapes.
    (should (equal (plist-get run :notes) '("Using the stand-in server" "Done: 1 request")))))

(ert-deftest orangu-test-run-stream ()
  "The answer streams in pieces, without escapes cut between them."
  (skip-unless (executable-find "sh"))
  (let ((run (orangu-test--run "stream")))
    (should (eq (plist-get (plist-get run :result) :outcome) 'succeeded))
    (should (equal (plist-get run :answer) "Hello, world!\n"))))

(ert-deftest orangu-test-run-failure ()
  "A failure says what orangu said last."
  (skip-unless (executable-find "sh"))
  (let ((result (plist-get (orangu-test--run "fail") :result)))
    (should (eq (plist-get result :outcome) 'failed))
    (should (equal (plist-get result :error) "error: the stand-in server is not running"))))

(ert-deftest orangu-test-run-stop ()
  "Stopping a run ends orangu and what it started."
  (skip-unless (and (executable-find "sh") (not (eq system-type 'windows-nt))))
  (let* ((child nil)
         (started (float-time))
         (run (orangu-test--run "sleep"
                                (lambda (run note)
                                  (when (string-match "^child \\([0-9]+\\)$" note)
                                    (setq child (string-to-number (match-string 1 note)))
                                    (orangu-run-stop run))))))
    (should (eq (plist-get (plist-get run :result) :outcome) 'cancelled))
    (should (< (- (float-time) started) 10))
    (should child)
    ;; Killed with its parent; as the only process of a container, Emacs
    ;; inherits it and does not reap it.
    (with-timeout (5 nil)
      (while (let ((attributes (process-attributes child)))
               (and attributes (not (equal (alist-get 'state attributes) "Z"))))
        (accept-process-output nil 0.05)))
    (let ((attributes (process-attributes child)))
      (should (or (null attributes) (equal (alist-get 'state attributes) "Z"))))))

(ert-deftest orangu-test-run-missing ()
  "A binary that is not there either fails to start or fails, and leaves nothing behind."
  (let ((before (length (process-list)))
        (result nil))
    ;; Whether Emacs signals or starts a process that fails depends on how
    ;; it starts processes on the system.
    (condition-case nil
        (progn
          (orangu-run-start (list (expand-file-name "no-such-orangu-here" temporary-file-directory))
                            temporary-file-directory
                            :on-answer #'ignore :on-note #'ignore
                            :on-end (lambda (end) (setq result end)))
          (with-timeout (5 (error "The run did not end"))
            (while (not result)
              (accept-process-output nil 0.05)))
          (should (eq (plist-get result :outcome) 'failed)))
      (file-error nil))
    (should (= (length (process-list)) before))))

(provide 'orangu-test)
;;; orangu-test.el ends here
