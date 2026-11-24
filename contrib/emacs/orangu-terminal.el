;;; orangu-terminal.el --- The orangu interface and conversation  -*- lexical-binding: t; -*-

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

;; The full interface of orangu in a terminal inside Emacs, and its plain
;; conversation in a comint buffer.

;;; Code:

(require 'cl-lib)
(require 'comint)
(require 'orangu-changes)
(require 'orangu-core)
(require 'orangu-panel)
(require 'subr-x)

(defcustom orangu-terminal 'auto
  "The terminal emulator `orangu-open' uses.
`auto' is the first that is installed of eat, vterm and term, which is
part of Emacs."
  :group 'orangu
  :type '(choice (const auto) (const eat) (const vterm) (const term)))

(declare-function eat-make "ext:eat" (name program &optional startfile &rest switches))
(declare-function vterm "ext:vterm" (&optional arg))
(declare-function term-ansi-make-term "term" (name program &optional startfile &rest switches))
(declare-function term-mode "term" ())
(declare-function term-char-mode "term" ())
(defvar vterm-shell)
(defvar markdown-hide-markup)
(defvar markdown-fontify-code-blocks-natively)
(declare-function gfm-mode "ext:markdown-mode" ())

(defcustom orangu-chat-markdown t
  "Non-nil to show the answers in `orangu-chat' as Markdown.
This needs `markdown-mode'; without it, the answers are plain text."
  :group 'orangu
  :type 'boolean)

(defun orangu-terminal--kind ()
  "Return the terminal emulator to use."
  (cond ((not (eq orangu-terminal 'auto)) orangu-terminal)
        ((or (featurep 'eat) (locate-library "eat")) 'eat)
        ((or (featurep 'vterm) (locate-library "vterm")) 'vterm)
        (t 'term)))

(defun orangu-terminal--buffers (prefix workspace)
  "Return the buffers of WORKSPACE whose name starts with PREFIX."
  (cl-remove-if-not (lambda (buffer)
                      (and (string-prefix-p prefix (buffer-name buffer))
                           (equal (buffer-local-value 'orangu-workspace buffer) workspace)))
                    (buffer-list)))

(defun orangu-terminal--live (prefix workspace)
  "Return the buffer of WORKSPACE named PREFIX and more whose orangu still runs."
  (cl-find-if (lambda (buffer) (process-live-p (get-buffer-process buffer)))
              (orangu-terminal--buffers prefix workspace)))

;;; The interface

;;;###autoload
(defun orangu-open ()
  "Open the full interface of orangu in a terminal, in the workspace.
When it is open already, show it."
  (interactive)
  (let* ((workspace (or orangu-workspace (orangu-core-workspace)))
         (live (orangu-terminal--live "*orangu terminal" workspace)))
    (if live
        (pop-to-buffer live)
      (when (eq system-type 'windows-nt)
        (user-error "Emacs has no terminal on Windows; use `orangu-chat'"))
      ;; What an orangu that has ended left there is of no use.
      (mapc #'kill-buffer (orangu-terminal--buffers "*orangu terminal" workspace))
      (let* ((command (orangu-core-interactive (orangu-core-locate) workspace))
             (default-directory (file-name-as-directory workspace))
             (name (generate-new-buffer-name
                    (format "*orangu terminal: %s*" (file-name-nondirectory workspace))))
             (buffer (pcase (orangu-terminal--kind)
                       ('eat (orangu-terminal--eat name command))
                       ('vterm (orangu-terminal--vterm name command))
                       (_ (orangu-terminal--term name command)))))
        (with-current-buffer buffer
          (setq orangu-workspace workspace))
        (pop-to-buffer buffer)))))

(defun orangu-terminal--term (name command)
  "Run COMMAND in a `term' buffer called NAME."
  (require 'term)
  (let ((buffer (apply #'term-ansi-make-term name (car command) nil (cdr command))))
    (with-current-buffer buffer
      (term-mode)
      (term-char-mode))
    buffer))

(defun orangu-terminal--eat (name command)
  "Run COMMAND in an `eat' buffer called NAME."
  (require 'eat)
  (apply #'eat-make (string-trim name "\\*" "\\*") (car command) nil (cdr command)))

(defun orangu-terminal--vterm (name command)
  "Run COMMAND in a `vterm' buffer called NAME."
  (require 'vterm)
  (let ((vterm-shell (mapconcat #'shell-quote-argument command " ")))
    (save-window-excursion
      (vterm name)
      (current-buffer))))

;;; The conversation

(defvar-local orangu-chat--changes nil "The snapshot from before the last prompt.")
(defvar-local orangu-chat--marks 0 "The continuation marks still to come.")
(defvar-local orangu-chat--render-timer nil "The timer that renders the answer.")

(defvar orangu-chat-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-j") #'newline)
    (define-key map (kbd "S-<return>") #'newline)
    (define-key map (kbd "C-c C-v") #'orangu-show-changes)
    map)
  "Keymap of `orangu-chat-mode'.")

(define-derived-mode orangu-chat-mode comint-mode "orangu-chat"
  "A conversation with orangu, which remembers what was said.

\\{orangu-chat-mode-map}"
  (setq-local comint-prompt-regexp "^> ")
  (setq-local comint-prompt-read-only t)
  (setq-local comint-process-echoes nil)
  (setq-local comint-input-sender #'orangu-chat--send)
  (add-hook 'comint-preoutput-filter-functions #'orangu-chat--filter nil t)
  (add-hook 'comint-output-filter-functions #'orangu-chat--answered nil t)
  (when (and orangu-chat-markdown (require 'markdown-mode nil t))
    (add-to-invisibility-spec 'markdown-markup)
    (add-hook 'comint-output-filter-functions #'orangu-chat--render-soon nil t))
  (visual-line-mode 1))

;;;###autoload
(defun orangu-chat ()
  "Open a conversation with orangu in the workspace.
When it is open already, show it."
  (interactive)
  (let* ((workspace (or orangu-workspace (orangu-core-workspace)))
         (live (orangu-terminal--live "*orangu chat" workspace)))
    (pop-to-buffer
     (or live
         (let* ((command (orangu-core-plain (orangu-core-locate) workspace))
                (default-directory (file-name-as-directory workspace))
                (process-environment (cons "NO_COLOR=1" process-environment))
                ;; A terminal would cut a long line short.
                (process-connection-type nil)
                (coding-system-for-read 'utf-8-unix)
                (coding-system-for-write 'utf-8-unix)
                (buffer (generate-new-buffer
                         (format "*orangu chat: %s*" (file-name-nondirectory workspace)))))
           (with-current-buffer buffer
             (orangu-chat-mode)
             (setq orangu-workspace workspace)
             (add-hook 'kill-buffer-hook #'orangu-chat--discard nil t)
             (apply #'make-comint-in-buffer "orangu-chat" buffer (car command) nil
                    (cdr command))
             (set-process-query-on-exit-flag (get-buffer-process buffer) nil))
           buffer)))))

(defun orangu-chat--discard ()
  "Remove the snapshot of this conversation, unless it is the latest changes."
  (when (and orangu-chat--changes
             (not (eq orangu-chat--changes (orangu-changes-latest orangu-workspace))))
    (orangu-changes-discard orangu-chat--changes)))

(defun orangu-chat--send (process text)
  "Send TEXT, a prompt, to PROCESS."
  (orangu-changes-save-buffers orangu-workspace)
  (orangu-chat--discard)
  (setq orangu-chat--changes
        (with-demoted-errors "orangu cannot follow what changes: %S"
          (orangu-changes-begin orangu-workspace)))
  (let ((input (orangu-core-plain-input text)))
    (setq orangu-chat--marks (cl-count ?\n input))
    (comint-simple-send process input)))

(defun orangu-chat--filter (text)
  "Return TEXT, output of orangu, without escapes and continuation marks."
  (let ((text (replace-regexp-in-string "\e\\[[0-9;?]*[@-~]\\|\r" "" text)))
    (while (and (> orangu-chat--marks 0) (string-prefix-p "… " text))
      (setq text (substring text 2)
            orangu-chat--marks (1- orangu-chat--marks)))
    text))

(defun orangu-chat--answered (_text)
  "When orangu asks for the next prompt, look at what it changed.
An answer may have the mark of a prompt in it, so this may come early;
it comes again at the end."
  (when (and (get-buffer-process (current-buffer))
             (save-excursion
               (goto-char (process-mark (get-buffer-process (current-buffer))))
               (and (looking-back "^> " (line-beginning-position))
                    (= (line-beginning-position) (- (point) 2)))))
    (when (memq #'orangu-chat--render-soon comint-output-filter-functions)
      (orangu-chat--render (current-buffer)))
    (orangu-changes-sync-buffers orangu-workspace)
    (when orangu-chat--changes
      (with-demoted-errors "orangu cannot tell what changed: %S"
        (when-let ((files (orangu-changes-end orangu-chat--changes)))
          (message "%s: %s" (orangu-changes-summary files)
                   (mapconcat #'car files ", ")))))))

;;; Markdown

(defconst orangu-chat--note-regexp "^\\[\\(?:tool\\] .*\\|total .*\\)\\]?$"
  "A line of notes from orangu among the answer: a tool it ran, the timings.")

(defun orangu-chat--render-soon (_text)
  "Render the answer that is coming in, once the output pauses."
  (unless orangu-chat--render-timer
    (setq orangu-chat--render-timer
          (run-with-timer 0.2 nil #'orangu-chat--render (current-buffer)))))

(defun orangu-chat--render (buffer)
  "Show the whole lines of the latest answer in BUFFER as Markdown.
The text stays as orangu wrote it; only how it looks changes."
  (when (buffer-live-p buffer)
    (with-current-buffer buffer
      (when orangu-chat--render-timer
        (cancel-timer orangu-chat--render-timer)
        (setq orangu-chat--render-timer nil))
      (let* ((process (get-buffer-process buffer))
             (start (or (marker-position comint-last-input-end) (point-min)))
             (end (if process
                      (save-excursion
                        (goto-char (process-mark process))
                        (line-beginning-position))
                    (point-max))))
        (when (< start end)
          (orangu-chat--render-region start end))))))

(defun orangu-chat--render-region (start end)
  "Show the text between START and END as Markdown."
  (let ((text (buffer-substring-no-properties start end))
        (props '(face invisible display))
        spans)
    (with-temp-buffer
      (insert text)
      (delay-mode-hooks (gfm-mode))
      (setq-local markdown-hide-markup t)
      (setq-local markdown-fontify-code-blocks-natively t)
      (font-lock-ensure)
      (let ((pos (point-min)))
        (while (< pos (point-max))
          (let ((next (or (next-single-property-change pos 'face nil (point-max)) (point-max))))
            (dolist (prop '(invisible display))
              (setq next (min next (or (next-single-property-change pos prop nil next) next))))
            (let ((values (mapcar (lambda (prop) (get-text-property pos prop)) props)))
              (when (delq nil (copy-sequence values))
                (push (list (1- pos) (1- next) values) spans)))
            (setq pos next)))))
    (with-silent-modifications
      (remove-list-of-text-properties start end '(font-lock-face invisible display))
      (dolist (span spans)
        (pcase-let ((`(,from ,to (,face ,invisible ,display)) span))
          (let ((from (+ start from)) (to (+ start to)))
            (when face (put-text-property from to 'font-lock-face face))
            (when invisible (put-text-property from to 'invisible invisible))
            (when display (put-text-property from to 'display display)))))
      (save-excursion
        (goto-char start)
        (while (re-search-forward orangu-chat--note-regexp end t)
          (put-text-property (match-beginning 0) (match-end 0) 'font-lock-face 'orangu-info))))))

(provide 'orangu-terminal)
;;; orangu-terminal.el ends here
