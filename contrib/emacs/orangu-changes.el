;;; orangu-changes.el --- What a run of orangu changed  -*- lexical-binding: t; -*-

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

;; orangu writes the files of the workspace itself.  Here buffers are kept
;; in step with the files, and in a Git repository what a run changed is
;; worked out, to be shown as a diff and to be undone.
;;
;; A snapshot is a Git tree of the files as they are, uncommitted changes
;; and untracked files included.  It is written with an index and an object
;; folder of its own, in a temporary folder, so the repository is only read.

;;; Code:

(require 'cl-lib)
(require 'diff-mode)
(require 'subr-x)

(defcustom orangu-save-buffers 'ask
  "What to do with the workspace's unsaved buffers before orangu runs.
orangu reads the files on disk."
  :group 'orangu
  :type '(choice (const :tag "Ask about each" ask)
                 (const :tag "Save them" t)
                 (const :tag "Leave them" nil)))

(defconst orangu-changes-untracked-bytes (* 1024 1024)
  "An untracked file larger than this is not part of a snapshot.")

(defconst orangu-changes-untracked-files 5000
  "With more untracked files than this, none is part of a snapshot.")

;;; Buffers

(defun orangu-changes--buffers (workspace)
  "Return the buffers visiting files in WORKSPACE."
  (let ((root (file-name-as-directory workspace)))
    (cl-remove-if-not (lambda (buffer)
                        (when-let ((file (buffer-file-name buffer)))
                          (and (not (file-remote-p file))
                               (file-in-directory-p file root))))
                      (buffer-list))))

(defun orangu-changes-save-buffers (workspace)
  "Save the unsaved buffers of WORKSPACE, as `orangu-save-buffers' says."
  (when orangu-save-buffers
    (let ((buffers (orangu-changes--buffers workspace)))
      (save-some-buffers (eq orangu-save-buffers t)
                         (lambda () (memq (current-buffer) buffers))))))

(defun orangu-changes-sync-buffers (workspace)
  "Bring the buffers of WORKSPACE up to date with their files.
Return the ones left alone because they have unsaved changes."
  (let (left)
    (dolist (buffer (orangu-changes--buffers workspace) (nreverse left))
      (with-current-buffer buffer
        (when (and (file-exists-p buffer-file-name)
                   (not (verify-visited-file-modtime buffer)))
          (if (buffer-modified-p)
              (push buffer left)
            (with-demoted-errors "orangu: %S"
              (revert-buffer t t t))))))))

;;; Git

(defun orangu-changes--git (root environment input &rest arguments)
  "Run git with ARGUMENTS in ROOT and return its output, as bytes.
ENVIRONMENT is added to the process environment.  INPUT, a string of
bytes, is its stdin.  Signal an error with what git said if it fails."
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (let* ((default-directory (file-name-as-directory root))
           (process-environment (append environment '("GIT_OPTIONAL_LOCKS=0")
                                        process-environment))
           (coding-system-for-read 'binary)
           (coding-system-for-write 'binary)
           (errors (make-temp-file "orangu-git"))
           (arguments (append '("-c" "core.fsmonitor=false" "-c" "core.quotePath=false"
                                "-c" "core.safecrlf=false" "--no-pager")
                              arguments)))
      (unwind-protect
          (let ((status (if input
                            (apply #'call-process-region input nil "git" nil
                                   (list t errors) nil arguments)
                          (apply #'call-process "git" nil (list t errors) nil arguments))))
            (if (eq status 0)
                (buffer-string)
              (error "%s" (with-temp-buffer
                            (let ((coding-system-for-read 'utf-8))
                              (insert-file-contents errors))
                            (let ((said (string-trim (buffer-string))))
                              (if (string-empty-p said)
                                  (format "git ended with %s" status)
                                said))))))
        (delete-file errors)))))

(defun orangu-changes--line (bytes)
  "Return the one line of BYTES, decoded."
  (string-trim-right (decode-coding-string bytes 'utf-8) "[\r\n]+"))

;;; Snapshots

(cl-defstruct (orangu-changes (:constructor orangu-changes--create) (:copier nil))
  "What a run changed: the trees BEFORE and AFTER, and FILES with their counts.
FILES is a list of (PATH ADDED REMOVED); the counts are nil for a binary."
  workspace root directory environment before after files undone)

(defvar orangu-changes--latest (make-hash-table :test #'equal)
  "The changes of the last run that made any, by workspace.")

(defvar orangu-changes--directories nil
  "The folders snapshots are in.")

(defun orangu-changes-begin (workspace)
  "Take a snapshot of WORKSPACE, and return it for `orangu-changes-end'.
Return nil when WORKSPACE is not in a Git repository."
  (when-let ((root (and (executable-find "git")
                        (ignore-errors
                          (orangu-changes--line
                           (orangu-changes--git workspace nil nil
                                                "rev-parse" "--show-toplevel"))))))
    (let* ((directory (make-temp-file "orangu-changes" t))
           (path (lambda (name)
                   (expand-file-name
                    (orangu-changes--line
                     (orangu-changes--git root nil nil "rev-parse" "--git-path" name))
                    root)))
           (done nil))
      (push directory orangu-changes--directories)
      (unwind-protect
          (let ((index (expand-file-name "index" directory))
                (objects (expand-file-name "objects" directory))
                (real-index (funcall path "index"))
                (alternates (getenv "GIT_ALTERNATE_OBJECT_DIRECTORIES")))
            (make-directory objects)
            ;; The index has the stat of each file, so only what changed
            ;; since is read.
            (when (file-exists-p real-index)
              (copy-file real-index index nil t))
            (let ((changes
                   (orangu-changes--create
                    :workspace workspace :root root :directory directory
                    :environment
                    (list (concat "GIT_INDEX_FILE=" index)
                          (concat "GIT_OBJECT_DIRECTORY=" objects)
                          (concat "GIT_ALTERNATE_OBJECT_DIRECTORIES="
                                  (funcall path "objects")
                                  (if alternates (concat path-separator alternates) ""))))))
              (setf (orangu-changes-before changes) (orangu-changes--tree changes))
              (setq done t)
              changes))
        (unless done
          (orangu-changes--remove directory))))))

(defun orangu-changes-end (changes)
  "Work out what changed since CHANGES began, and return the files.
They become the workspace's latest changes when there are any; with none,
CHANGES is the caller's to discard.  This may be called again later."
  (setf (orangu-changes-after changes) (orangu-changes--tree changes)
        (orangu-changes-files changes) (orangu-changes--files changes))
  (let* ((workspace (orangu-changes-workspace changes))
         (latest (gethash workspace orangu-changes--latest)))
    (when (orangu-changes-files changes)
      (unless (or (null latest) (eq latest changes))
        (orangu-changes-discard latest))
      (puthash workspace changes orangu-changes--latest)))
  (orangu-changes-files changes))

(defun orangu-changes-latest (workspace)
  "Return the changes of the last run in WORKSPACE that made any, or nil."
  (gethash workspace orangu-changes--latest))

(defun orangu-changes-discard (changes)
  "Remove what CHANGES keeps on disk."
  (orangu-changes--remove (orangu-changes-directory changes)))

(defun orangu-changes--remove (directory)
  "Remove DIRECTORY, a snapshot's."
  (setq orangu-changes--directories (delete directory orangu-changes--directories))
  (ignore-errors (delete-directory directory t)))

(defun orangu-changes--discard-all ()
  "Remove what every snapshot keeps on disk."
  (mapc #'orangu-changes--remove orangu-changes--directories)
  (clrhash orangu-changes--latest))

(add-hook 'kill-emacs-hook #'orangu-changes--discard-all)

(defun orangu-changes--tree (changes)
  "Write the files as they are now as a tree of CHANGES, and return its name."
  (let ((root (orangu-changes-root changes))
        (environment (orangu-changes-environment changes)))
    (orangu-changes--git root environment nil "add" "-u")
    (when-let ((untracked (orangu-changes--untracked root environment)))
      (orangu-changes--git root environment
                           (mapconcat (lambda (path) (concat path "\0")) untracked "")
                           "update-index" "-z" "--add" "--stdin"))
    (orangu-changes--line (orangu-changes--git root environment nil "write-tree"))))

(defun orangu-changes--untracked (root environment)
  "Return the untracked files in ROOT that belong in a snapshot, as bytes.
ENVIRONMENT names the snapshot's index."
  (let ((paths (split-string (orangu-changes--git root environment nil "ls-files" "-z"
                                                  "--others" "--exclude-standard")
                             "\0" t)))
    (unless (length> paths orangu-changes-untracked-files)
      (cl-remove-if-not
       (lambda (path)
         ;; A repository inside this one is listed as a folder.
         (unless (string-suffix-p "/" path)
           (when-let ((attributes (file-attributes
                                   (expand-file-name (decode-coding-string path 'utf-8)
                                                     root))))
             (and (not (eq t (file-attribute-type attributes)))
                  (<= (file-attribute-size attributes) orangu-changes-untracked-bytes)))))
       paths))))

(defun orangu-changes--diff (changes &rest options)
  "Run diff-tree with OPTIONS between the two trees of CHANGES."
  (apply #'orangu-changes--git (orangu-changes-root changes)
         (orangu-changes-environment changes) nil
         "diff-tree" "-r" "--no-renames" "--no-color" "--no-ext-diff" "--no-textconv"
         (append options (list (orangu-changes-before changes)
                               (orangu-changes-after changes)))))

(defun orangu-changes--files (changes)
  "Return the files of CHANGES, as (PATH ADDED REMOVED)."
  (mapcar (lambda (line)
            (pcase-let ((`(,added ,removed . ,path)
                         (split-string (decode-coding-string line 'utf-8) "\t")))
              (list (string-join path "\t")
                    (unless (equal added "-") (string-to-number added))
                    (unless (equal removed "-") (string-to-number removed)))))
          (split-string (orangu-changes--diff changes "--numstat" "-z") "\0" t)))

(defun orangu-changes-patch (changes &rest options)
  "Return CHANGES as a patch, as bytes.  OPTIONS are added to the diff's."
  (apply #'orangu-changes--diff changes "-p" "--src-prefix=a/" "--dst-prefix=b/" options))

(defun orangu-changes-undo (changes)
  "Put the files of CHANGES back as they were, or signal an error.
Either every file is put back or none is."
  (orangu-changes--git (orangu-changes-root changes) nil
                       (orangu-changes-patch changes "--binary" "--full-index")
                       "apply" "-R" "--whitespace=nowarn" "-")
  (setf (orangu-changes-undone changes) t))

(defun orangu-changes-summary (files)
  "Return a few words on FILES, as `orangu-changes-files' has them."
  (format "%d file%s changed" (length files) (if (length= files 1) "" "s")))

;;; The diff

(defvar-local orangu-changes--shown nil
  "The changes this buffer shows.")

(defvar orangu-changes-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-c C-k") #'orangu-undo-changes)
    map)
  "Keymap of `orangu-changes-mode'.")

(define-derived-mode orangu-changes-mode diff-mode "orangu-changes"
  "What a run of orangu changed, as a diff.

\\{orangu-changes-mode-map}"
  (setq buffer-read-only t)
  (setq header-line-format
        (substitute-command-keys
         "\\<orangu-changes-mode-map>What orangu changed.  \\[diff-goto-source] goes to the code, \
\\[orangu-undo-changes] undoes all of it")))

(defun orangu-changes--buffer-name (workspace)
  "Return the name of the buffer with the changes in WORKSPACE."
  (format "*orangu changes: %s*" (file-name-nondirectory workspace)))

(defun orangu-changes-buffer (workspace)
  "Return a buffer with the latest changes in WORKSPACE, or a `user-error'."
  (let ((changes (or (orangu-changes-latest workspace)
                     (user-error "Orangu has changed nothing in %s"
                                 (abbreviate-file-name workspace)))))
    (with-current-buffer (get-buffer-create (orangu-changes--buffer-name workspace))
      (let ((inhibit-read-only t))
        (erase-buffer)
        (insert (decode-coding-string (orangu-changes-patch changes) 'utf-8)))
      (orangu-changes-mode)
      (setq default-directory (file-name-as-directory (orangu-changes-root changes))
            orangu-changes--shown changes)
      (set-buffer-modified-p nil)
      (goto-char (point-min))
      (current-buffer))))

;;;###autoload
(defun orangu-undo-changes ()
  "Undo what the last run of orangu changed, after asking.
When a file has changed since, nothing is undone."
  (interactive)
  (require 'orangu-core)
  (let* ((workspace (if orangu-changes--shown
                        (orangu-changes-workspace orangu-changes--shown)
                      (orangu-changes--current-workspace)))
         (changes (or (orangu-changes-latest workspace)
                      (user-error "Orangu has changed nothing in %s"
                                  (abbreviate-file-name workspace)))))
    (when (orangu-changes-undone changes)
      (user-error "These changes are undone already"))
    (when (and orangu-changes--shown (not (eq orangu-changes--shown changes)))
      (user-error "Orangu has made newer changes; show them first"))
    (orangu-changes-save-buffers workspace)
    (when (yes-or-no-p (format "Undo orangu's changes (%s)? "
                               (orangu-changes-summary (orangu-changes-files changes))))
      (orangu-changes-undo changes)
      (orangu-changes-sync-buffers workspace)
      (message "Undone: %s" (string-join (mapcar #'car (orangu-changes-files changes)) ", ")))))

;;;###autoload
(defun orangu-show-changes ()
  "Show what the last run of orangu changed, as a diff."
  (interactive)
  (pop-to-buffer (orangu-changes-buffer (orangu-changes--current-workspace))))

(declare-function orangu-core-workspace "orangu-core" (&optional directory))

(defvar-local orangu-workspace nil
  "The workspace of a buffer orangu made.")

(defun orangu-changes--current-workspace ()
  "Return the workspace of the current buffer."
  (require 'orangu-core)
  (or orangu-workspace (orangu-core-workspace)))

(provide 'orangu-changes)
;;; orangu-changes.el ends here
