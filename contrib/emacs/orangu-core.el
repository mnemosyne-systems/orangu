;;; orangu-core.el --- Finding orangu, its workspace and its prompts  -*- lexical-binding: t; -*-

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

;; What does not touch a buffer or a process: where the binary and the
;; workspace are, the command lines, and the prompts the commands send.

;;; Code:

(require 'cl-lib)
(require 'project)
(require 'subr-x)

(defgroup orangu nil
  "The orangu coding environment."
  :group 'tools
  :prefix "orangu-"
  :link '(url-link "https://github.com/mnemosyne-systems/orangu"))

(defcustom orangu-path nil
  "The `orangu' binary, or the folder it is in.
When nil, the first `orangu' on the variable `exec-path', then the one
in ~/.cargo/bin."
  :type '(choice (const :tag "Find it" nil) file))
;; A folder's local variables must not choose what is run.
(put 'orangu-path 'risky-local-variable t)

;;; The binary

(defun orangu-core-binary-name ()
  "The file name of the orangu binary on this system."
  (if (eq system-type 'windows-nt) "orangu.exe" "orangu"))

(defun orangu-core-locate ()
  "Return the path of the orangu binary, or signal a `user-error'."
  (let ((name (orangu-core-binary-name))
        (setting (string-trim (or orangu-path ""))))
    (if (string-empty-p setting)
        (or (cl-find-if #'orangu-core--binary-p
                        (mapcar (lambda (directory) (expand-file-name name directory))
                                (append (cl-remove-if #'null exec-path)
                                        ;; A desktop's PATH may be without it.
                                        '("~/.cargo/bin"))))
            (user-error "Could not find %s on the PATH; install orangu, or set `orangu-path'"
                        name))
      ;; A folder is taken to mean the binary inside it.
      (or (cl-find-if #'orangu-core--binary-p
                      (list (expand-file-name setting)
                            (expand-file-name name setting)))
          (user-error "`orangu-path' names no binary: %s" setting)))))

(defun orangu-core--binary-p (file)
  "Non-nil when FILE is an executable file."
  (and (file-regular-p file) (file-executable-p file)))

;;; The workspace

(defun orangu-core-workspace (&optional directory)
  "Return the folder orangu runs in for DIRECTORY, without a trailing slash.
It is the project DIRECTORY is in, or else the Git repository, or else
DIRECTORY itself.  DIRECTORY defaults to `default-directory'."
  (let ((directory (expand-file-name (or directory default-directory))))
    (when (file-remote-p directory)
      (user-error "Orangu runs on this machine, and %s is on another one"
                  (abbreviate-file-name directory)))
    (directory-file-name
     (expand-file-name
      (or (when-let ((project (project-current nil directory)))
            (project-root project))
          (locate-dominating-file directory ".git")
          directory)))))

(defun orangu-core-relative (workspace file)
  "Return FILE relative to WORKSPACE with forward slashes, or nil when outside it."
  (let ((relative (lambda (file root)
                    (let ((root (file-name-as-directory root)))
                      (when (and (string-prefix-p root file)
                                 (not (string= root file)))
                        (substring file (length root)))))))
    (or (funcall relative (expand-file-name file) (expand-file-name workspace))
        ;; Reached through a link, the file is inside but its name is not.
        (funcall relative (file-truename file) (file-truename workspace)))))

(defun orangu-core-display-path (workspace file)
  "Return FILE as the tools of orangu take it in WORKSPACE."
  (or (orangu-core-relative workspace file) (expand-file-name file)))

;;; The command lines

(defun orangu-core-one-shot (binary workspace prompt)
  "The command that has BINARY answer PROMPT in WORKSPACE."
  (list binary "-w" workspace "-p" prompt))

(defun orangu-core-interactive (binary workspace)
  "The command that starts the interface of BINARY in WORKSPACE."
  (list binary "-w" workspace))

(defun orangu-core-plain (binary workspace)
  "The command that starts a plain conversation with BINARY in WORKSPACE."
  (list binary "-w" workspace "--plain"))

;;; Selections

(cl-defstruct (orangu-selection (:constructor orangu-selection-create) (:copier nil))
  "Lines of a file.  START and END count from 1; TEXT is nil for a position."
  file start end text)

(defun orangu-core-selection-label (workspace selection)
  "The short form of SELECTION in WORKSPACE: src/a.rs:3-9."
  (let ((start (orangu-selection-start selection))
        (end (orangu-selection-end selection)))
    (format "%s:%s"
            (orangu-core-display-path workspace (orangu-selection-file selection))
            (if (= start end) start (format "%d-%d" start end)))))

(defconst orangu-core-languages
  '(("c" . "c") ("h" . "c") ("cc" . "cpp") ("cpp" . "cpp") ("cxx" . "cpp")
    ("hpp" . "cpp") ("cs" . "csharp") ("rs" . "rust") ("py" . "python")
    ("js" . "javascript") ("mjs" . "javascript") ("jsx" . "jsx")
    ("ts" . "typescript") ("tsx" . "tsx") ("java" . "java") ("kt" . "kotlin")
    ("go" . "go") ("rb" . "ruby") ("php" . "php") ("swift" . "swift")
    ("sql" . "sql") ("json" . "json") ("xml" . "xml") ("html" . "html")
    ("css" . "css") ("yml" . "yaml") ("yaml" . "yaml") ("md" . "markdown")
    ("sh" . "bash") ("ps1" . "powershell") ("toml" . "toml")
    ("el" . "elisp") ("lisp" . "lisp") ("scm" . "scheme"))
  "The language of a code block, by file extension.")

(defun orangu-core--fence (code)
  "Return a fence longer than any run of backticks in CODE."
  (let ((longest 2) (start 0))
    (while (string-match "`+" code start)
      (setq longest (max longest (- (match-end 0) (match-beginning 0)))
            start (match-end 0)))
    (make-string (1+ longest) ?`)))

(defun orangu-core--where (workspace selection)
  "Describe SELECTION in WORKSPACE to the model: the place, and the code."
  (let* ((file (orangu-selection-file selection))
         (path (orangu-core-display-path workspace file))
         (start (orangu-selection-start selection))
         (end (orangu-selection-end selection))
         (text (orangu-selection-text selection)))
    (if (null text)
        (format "The cursor is at line %d of %s." start path)
      (let* ((code (string-trim-right (string-replace "\r\n" "\n" text) "\n+"))
             (fence (orangu-core--fence code))
             (language (cdr (assoc (downcase (or (file-name-extension file) ""))
                                   orangu-core-languages))))
        (format "This is %s of %s:\n\n%s%s\n%s\n%s"
                (if (= start end)
                    (format "line %d" start)
                  (format "lines %d-%d" start end))
                path fence (or language "") code fence)))))

;;; Prompts

(defconst orangu-core-default-question "Explain what this code does."
  "What is asked about a selection when nothing else is.")

(defconst orangu-core-review-changes "/auto_review"
  "The review orangu has of everything the branch changes.")

(defun orangu-core-about-selection (workspace selection question)
  "Ask QUESTION about SELECTION in WORKSPACE."
  (let ((question (string-trim question)))
    (concat (if (string-empty-p question) orangu-core-default-question question)
            "\n\n"
            (orangu-core--where workspace selection))))

(defun orangu-core-implement (workspace selection instruction)
  "Ask for INSTRUCTION to be carried out in WORKSPACE, at SELECTION or anywhere.
The prompt starts the same whatever INSTRUCTION is, so orangu never takes
one such as \"edit parser.rs\" for its own command of that name."
  (concat "Implement this in the workspace, editing its files with your tools:\n\n"
          (string-trim instruction)
          (if selection
              (concat "\n\n" (orangu-core--where workspace selection))
            "")))

(defun orangu-core-review-file (workspace file)
  "The prompt that reviews FILE in WORKSPACE, or a `user-error' when outside it."
  (let ((relative (orangu-core-relative workspace file)))
    (unless relative
      (user-error "%s is outside the workspace %s, so orangu cannot review it"
                  (abbreviate-file-name file) (abbreviate-file-name workspace)))
    (concat orangu-core-review-changes " " (orangu-core-review-argument relative))))

(defun orangu-core-review-argument (relative)
  "RELATIVE, a workspace path, as one argument of /auto_review.
Arguments are split on whitespace, and one with *, ?, [ or { in it is a
pattern, so such a path is given as a pattern that matches only it.  A
space cannot be kept: it becomes ?, which matches it."
  (if (not (string-match-p "[*?[{[:space:]]" relative))
      relative
    (mapconcat (lambda (c)
                 (cond ((memq c '(?* ?? ?\[ ?\] ?{ ?})) (format "[%c]" c))
                       ((string-match-p "[[:space:]\n]" (char-to-string c)) "?")
                       (t (char-to-string c))))
               relative "")))

(defun orangu-core-plain-input (text)
  "TEXT as `orangu --plain' reads one prompt: each line but the last ends in \\."
  (let ((text (string-trim-right text)))
    (concat (string-replace "\n" "\\\n" text)
            ;; The last line goes on to no other; a space keeps its own \.
            (if (string-suffix-p "\\" text) " " ""))))

(provide 'orangu-core)
;;; orangu-core.el ends here
