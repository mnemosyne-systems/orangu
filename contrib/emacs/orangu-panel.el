;;; orangu-panel.el --- The orangu panel  -*- lexical-binding: t; -*-

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

;; A buffer with the prompts and the answers above, and a place to type the
;; next prompt below.  Each prompt is one `orangu -w WORKSPACE -p PROMPT'.

;;; Code:

(require 'button)
(require 'cl-lib)
(require 'orangu-changes)
(require 'orangu-core)
(require 'orangu-run)
(require 'ring)
(require 'subr-x)

(defcustom orangu-show-changes t
  "Whether what `orangu-implement' changed is shown as a diff at the end."
  :group 'orangu
  :type 'boolean)

(defface orangu-prompt '((t :inherit font-lock-keyword-face :weight bold))
  "A prompt, in the panel."
  :group 'orangu)

(defface orangu-info '((t :inherit shadow))
  "What the panel says itself."
  :group 'orangu)

(defvar-local orangu-panel--run nil "The run that is going, or the last one.")
(defvar-local orangu-panel--status "" "What orangu is doing, or how the last run went.")
(defvar-local orangu-panel--attached nil "The selection attached to the next prompt.")
(defvar-local orangu-panel--output nil "Where the transcript ends.")
(defvar-local orangu-panel--input nil "Where the next prompt begins.")
(defvar-local orangu-panel--kind nil "What the transcript ends with.")
(defvar-local orangu-panel--history nil "The prompts typed here.")
(defvar-local orangu-panel--history-index nil "The prompt of the history shown, if any.")

;;; The mode

(defvar orangu-panel-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "RET") #'orangu-panel-send)
    (define-key map (kbd "C-c C-c") #'orangu-panel-send)
    (define-key map (kbd "S-<return>") #'newline)
    (define-key map (kbd "C-j") #'newline)
    (define-key map (kbd "M-p") #'orangu-panel-previous)
    (define-key map (kbd "M-n") #'orangu-panel-next)
    (define-key map (kbd "C-c C-k") #'orangu-stop)
    (define-key map (kbd "C-c C-l") #'orangu-clear)
    (define-key map (kbd "C-c C-d") #'orangu-panel-detach)
    (define-key map (kbd "C-c C-v") #'orangu-show-changes)
    (define-key map (kbd "C-c C-u") #'orangu-undo-changes)
    map)
  "Keymap of `orangu-panel-mode'.")

(define-derived-mode orangu-panel-mode fundamental-mode "orangu"
  "The orangu panel: type a prompt at the end and press \\[orangu-panel-send].

\\{orangu-panel-mode-map}"
  (setq-local truncate-lines nil)
  (visual-line-mode 1)
  (setq orangu-panel--history (make-ring 100))
  (setq header-line-format '(:eval (orangu-panel--header)))
  (setq mode-line-process '(:eval (if (orangu-run-live-p orangu-panel--run) ":running" "")))
  (add-hook 'kill-buffer-hook #'orangu-panel--killed nil t)
  (orangu-panel--reset))

(defun orangu-panel-buffer (workspace)
  "Return the panel of WORKSPACE, made when there is none."
  (or (cl-find-if (lambda (buffer)
                    (and (eq (buffer-local-value 'major-mode buffer) 'orangu-panel-mode)
                         (equal (buffer-local-value 'orangu-workspace buffer) workspace)))
                  (buffer-list))
      (with-current-buffer
          (generate-new-buffer (format "*orangu: %s*" (file-name-nondirectory workspace)))
        (orangu-panel-mode)
        (setq orangu-workspace workspace
              default-directory (file-name-as-directory workspace))
        (current-buffer))))

(defun orangu-panel--header ()
  "Return the header line: the workspace and the status."
  (string-replace "%" "%%"
                  (concat " " (abbreviate-file-name (or orangu-workspace ""))
                          (unless (string-empty-p orangu-panel--status)
                            (concat " · " orangu-panel--status)))))

(defun orangu-panel--set-status (status)
  "Show STATUS."
  (setq orangu-panel--status status)
  (force-mode-line-update))

(defun orangu-panel--killed ()
  "Stop the run of a panel that goes away."
  (orangu-run-stop orangu-panel--run))

;;; The text

(defmacro orangu-panel--writing (&rest body)
  "Run BODY, which changes what the panel shows above the prompt."
  (declare (indent 0) (debug t))
  `(progn
     (let ((inhibit-read-only t)
           (buffer-undo-list t))
       (save-excursion ,@body))
     ;; What could be undone is not where it was.
     (setq buffer-undo-list nil)))

(defun orangu-panel--fixed (text &optional face)
  "Return TEXT in FACE as text that cannot be edited."
  (propertize text 'face face 'read-only t 'front-sticky '(read-only) 'rear-nonsticky t
              'field 'orangu-output))

(defun orangu-panel--reset ()
  "Empty the transcript, and keep what is typed."
  (let ((typed (if orangu-panel--input (orangu-panel--typed) ""))
        (inhibit-read-only t))
    (erase-buffer)
    (setq orangu-panel--output (copy-marker (point-min))
          orangu-panel--input (copy-marker (point-min))
          orangu-panel--kind nil)
    (insert typed)
    (orangu-panel--draw-prompt)
    (goto-char (point-max))))

(defun orangu-panel--draw-prompt ()
  "Draw what stands before the prompt: the attached selection, and a mark."
  (let* ((from-end (- (point) orangu-panel--input))
         (selection orangu-panel--attached)
         (blank (= orangu-panel--output (point-min))))
    (orangu-panel--writing
      (delete-region orangu-panel--output orangu-panel--input)
      (goto-char orangu-panel--output)
      (unless blank
        (insert (orangu-panel--fixed "\n")))
      (when selection
        (insert (orangu-panel--fixed
                 (concat "[" (orangu-core-selection-label orangu-workspace selection) "]")
                 'orangu-info)
                (orangu-panel--fixed " "))
        (insert-text-button "Remove" 'action (lambda (_) (orangu-panel-detach))
                            'follow-link t 'read-only t 'rear-nonsticky t
                            'field 'orangu-output)
        (insert (orangu-panel--fixed "\n")))
      (insert (orangu-panel--fixed "> " 'orangu-prompt))
      (set-marker orangu-panel--input (point)))
    (when (>= from-end 0)
      (goto-char (+ orangu-panel--input from-end))
      (dolist (window (get-buffer-window-list nil nil t))
        (when (>= (window-point window) orangu-panel--output)
          (set-window-point window (point)))))))

(defun orangu-panel--write (text &optional face)
  "Add TEXT in FACE to the transcript."
  (let ((blank (= orangu-panel--output (point-min))))
    (orangu-panel--writing
      (goto-char orangu-panel--output)
      (insert (orangu-panel--fixed text face))
      (set-marker orangu-panel--output (point)))
    (when blank
      (orangu-panel--draw-prompt))))

(defun orangu-panel--section (kind)
  "Begin KIND in the transcript, with an empty line before it."
  (unless (eq kind orangu-panel--kind)
    (let ((newlines 0))
      (save-excursion
        (goto-char orangu-panel--output)
        (unless (bobp)
          (setq newlines (- 2 (min 2 (- (point) (progn (skip-chars-backward "\n") (point))))))))
      (when (> newlines 0)
        (orangu-panel--write (make-string newlines ?\n))))
    (setq orangu-panel--kind kind)))

(defun orangu-panel--prompt (text)
  "Show the prompt TEXT in the transcript."
  (setq orangu-panel--kind nil)
  (orangu-panel--section 'prompt)
  (orangu-panel--write (concat "> " text) 'orangu-prompt))

(defun orangu-panel--answer (text)
  "Show TEXT, more of the answer."
  (orangu-panel--section 'answer)
  (orangu-panel--write text))

(defun orangu-panel--info (text)
  "Say TEXT in the transcript."
  (orangu-panel--section 'info)
  (orangu-panel--write text 'orangu-info))

(defun orangu-panel--changed (changes)
  "List the files of CHANGES in the transcript."
  (setq orangu-panel--kind nil)
  (orangu-panel--info (concat (orangu-changes-summary (orangu-changes-files changes)) " ("))
  (orangu-panel--button "show" (lambda (_) (orangu-show-changes)))
  (orangu-panel--info ", ")
  (orangu-panel--button "undo" (lambda (_) (orangu-undo-changes)))
  (orangu-panel--info ")")
  (pcase-dolist (`(,path ,added ,removed) (orangu-changes-files changes))
    (let ((file (expand-file-name path (orangu-changes-root changes))))
      (orangu-panel--info "\n  ")
      (orangu-panel--button path (lambda (_) (find-file-other-window file)))
      (orangu-panel--info (if added (format " +%d -%d" added removed) " binary")))))

(defun orangu-panel--button (label action)
  "Add a button LABEL that does ACTION to the transcript."
  (orangu-panel--writing
    (goto-char orangu-panel--output)
    (insert-text-button label 'action action 'follow-link t 'read-only t
                        'rear-nonsticky t 'field 'orangu-output)
    (set-marker orangu-panel--output (point))))

;;; The prompt

(defun orangu-panel--typed ()
  "Return what is typed."
  (buffer-substring-no-properties orangu-panel--input (point-max)))

(defun orangu-panel--replace (text)
  "Replace what is typed with TEXT."
  (delete-region orangu-panel--input (point-max))
  (goto-char (point-max))
  (insert text))

(defun orangu-panel-fill (text)
  "Put TEXT at the prompt of this panel, unless something is typed there."
  (when (string-blank-p (orangu-panel--typed))
    (orangu-panel--replace text)))

(defun orangu-panel-send ()
  "Ask orangu what is typed; with a selection attached, about that selection."
  (interactive nil orangu-panel-mode)
  (if (< (point) orangu-panel--input)
      (goto-char (point-max))
    (let* ((text (string-trim (orangu-panel--typed)))
           (selection orangu-panel--attached))
      (unless (and (string-empty-p text) (null selection))
        (orangu-panel-run
         (if selection
             (orangu-core-about-selection orangu-workspace selection text)
           text)
         (if selection
             (format "%s\n(%s)"
                     (if (string-empty-p text) orangu-core-default-question text)
                     (orangu-core-selection-label orangu-workspace selection))
           text))
        ;; Only now has it started: what was typed is still there after an error.
        (unless (string-empty-p text)
          (ring-insert orangu-panel--history text))
        (setq orangu-panel--history-index nil)
        (orangu-panel--replace "")
        (orangu-panel-detach)))))

(defun orangu-panel-previous (&optional step)
  "Show the previous prompt typed here, or the one STEP before."
  (interactive nil orangu-panel-mode)
  (let ((length (ring-length orangu-panel--history))
        (index (+ (or orangu-panel--history-index -1) (or step 1))))
    (cond ((zerop length) (user-error "No prompts yet"))
          ((>= index length) (user-error "That was the first prompt"))
          ((< index 0)
           (setq orangu-panel--history-index nil)
           (orangu-panel--replace ""))
          (t (setq orangu-panel--history-index index)
             (orangu-panel--replace (ring-ref orangu-panel--history index))))))

(defun orangu-panel-next ()
  "Show the next prompt typed here."
  (interactive nil orangu-panel-mode)
  (orangu-panel-previous -1))

(defun orangu-panel-attach (selection)
  "Attach SELECTION to the next prompt of this panel."
  (setq orangu-panel--attached selection)
  (orangu-panel--draw-prompt)
  (orangu-panel-fill orangu-core-default-question)
  (goto-char (point-max)))

(defun orangu-panel-detach ()
  "Remove the selection attached to the next prompt."
  (interactive nil orangu-panel-mode)
  (when orangu-panel--attached
    (setq orangu-panel--attached nil)
    (orangu-panel--draw-prompt)))

;;; Runs

(defun orangu-panel-run (prompt shown &optional diff)
  "Send PROMPT to orangu and show SHOWN as what was asked.
With DIFF, what it changed is shown at the end."
  (when (orangu-run-live-p orangu-panel--run)
    (user-error "Orangu is still answering; stop it first"))
  (let* ((workspace orangu-workspace)
         (panel (current-buffer))
         (binary (orangu-core-locate))
         (changes nil)
         (here (lambda (function &rest arguments)
                 (when (buffer-live-p panel)
                   (with-current-buffer panel
                     (apply function arguments))))))
    (orangu-changes-save-buffers workspace)
    (setq changes (with-demoted-errors "orangu cannot follow what changes: %S"
                    (orangu-changes-begin workspace)))
    (condition-case err
        (setq orangu-panel--run
              (orangu-run-start
               (orangu-core-one-shot binary workspace prompt) workspace
               :on-answer (lambda (text) (funcall here #'orangu-panel--answer text))
               :on-note (lambda (line) (funcall here #'orangu-panel--set-status line))
               :on-end (lambda (result)
                         (funcall here #'orangu-panel--end workspace changes result diff)
                         (unless (buffer-live-p panel)
                           (orangu-changes-sync-buffers workspace)
                           (when changes (orangu-changes-discard changes))))))
      (error
       (when changes (orangu-changes-discard changes))
       (let ((message (format "Could not start orangu: %s" (error-message-string err))))
         (orangu-panel--set-status message)
         (user-error "%s" message))))
    (orangu-panel--prompt shown)
    (orangu-panel--set-status "Starting orangu…")))

(defun orangu-panel--end (workspace changes result diff)
  "The run in WORKSPACE is over, with RESULT.
CHANGES is the snapshot from before it; with DIFF, what changed is shown."
  (let ((left (orangu-changes-sync-buffers workspace))
        (files (when changes
                 (with-demoted-errors "orangu cannot tell what changed: %S"
                   (orangu-changes-end changes)))))
    (if files
        (orangu-panel--changed changes)
      (when changes
        (orangu-changes-discard changes)))
    (when left
      (orangu-panel--section 'left)
      (orangu-panel--info
       (format "Changed on disk, and not read again as it has unsaved changes: %s"
               (mapconcat #'buffer-name left ", "))))
    (orangu-panel--set-status
     (concat (orangu-panel-end-status result)
             (if files (concat " · " (orangu-changes-summary files)) "")))
    (when (and files diff)
      (orangu-panel--show-diff (orangu-changes-buffer workspace)))))

(defun orangu-panel--show-diff (diff)
  "Show the buffer DIFF below the panel, or wherever there is room."
  (let ((window (get-buffer-window)))
    (with-selected-window (or window (selected-window))
      (display-buffer diff (if window
                               '((display-buffer-reuse-window
                                  display-buffer-below-selected))
                             t)))))

(defun orangu-panel-end-status (result)
  "Return the status for RESULT, as `orangu-run-start' gives it."
  (pcase (plist-get result :outcome)
    ('succeeded (format "Done in %.1fs" (plist-get result :elapsed)))
    ('cancelled "Stopped")
    (_ (format "orangu failed: %s" (plist-get result :error)))))

(defun orangu-panel--current ()
  "Return the panel the current buffer is about."
  (orangu-panel-buffer (or orangu-workspace (orangu-core-workspace))))

;;;###autoload
(defun orangu-stop ()
  "Stop orangu, and everything it started."
  (interactive)
  (with-current-buffer (orangu-panel--current)
    (if (orangu-run-live-p orangu-panel--run)
        (orangu-run-stop orangu-panel--run)
      (message "orangu is not running"))))

;;;###autoload
(defun orangu-clear ()
  "Empty the panel.  A run that is going goes on."
  (interactive)
  (with-current-buffer (orangu-panel--current)
    (orangu-panel--reset)
    (unless (orangu-run-live-p orangu-panel--run)
      (orangu-panel--set-status ""))))

(provide 'orangu-panel)
;;; orangu-panel.el ends here
