;;; orangu.el --- Ask, implement and review with orangu  -*- lexical-binding: t; -*-

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
;;
;; Author: The orangu community
;; URL: https://github.com/mnemosyne-systems/orangu
;; Version: 2.1.0
;; Package-Requires: ((emacs "28.1"))
;; Keywords: tools, convenience

;;; Commentary:

;; Brings orangu into Emacs.  This package does no inference and has no agent
;; of its own: it runs the `orangu' binary, which talks to its server the way
;; orangu.conf says.  Whatever orangu can do in the terminal, it can do here.
;;
;;   M-x orangu            the panel: prompts and answers
;;   M-x orangu-implement  have code written or changed, and see the diff
;;   M-x orangu-chat       a conversation that is kept
;;   M-x orangu-open       the full interface in a terminal
;;
;; To have every command under one key:
;;
;;   (global-set-key (kbd "C-c o") 'orangu-command-map)

;;; Code:

(require 'orangu-changes)
(require 'orangu-core)
(require 'orangu-panel)
(require 'orangu-terminal)

;;; Selections

(defun orangu--file ()
  "Return the file of the current buffer, or a `user-error'."
  (or (buffer-file-name (buffer-base-buffer))
      (user-error "Orangu works on files, and this buffer has none")))

(defun orangu--region ()
  "Return the region as a selection.
A region that ends at the start of a line does not take that line."
  (let* ((start (region-beginning))
         (end (region-end))
         (first (line-number-at-pos start t))
         (last (line-number-at-pos end t)))
    (orangu-selection-create
     :file (orangu--file) :start first
     :end (if (and (> last first)
                   (save-excursion (goto-char end) (bolp)))
              (1- last)
            last)
     :text (buffer-substring-no-properties start end))))

(defun orangu--line ()
  "Return the line point is on as a selection."
  (let ((line (line-number-at-pos nil t)))
    (orangu-selection-create
     :file (orangu--file) :start line :end line
     :text (buffer-substring-no-properties (line-beginning-position) (line-end-position)))))

(defun orangu--position ()
  "Return where point is as a selection without text, or nil outside a file."
  (when (buffer-file-name (buffer-base-buffer))
    (let ((line (line-number-at-pos nil t)))
      (orangu-selection-create :file (orangu--file) :start line :end line))))

(defun orangu--workspace ()
  "Return the workspace of the current buffer."
  (or orangu-workspace (orangu-core-workspace)))

;;; Commands

;;;###autoload
(defun orangu ()
  "Open the orangu panel of the workspace."
  (interactive)
  (pop-to-buffer (orangu-panel-buffer (orangu--workspace)))
  (goto-char (point-max)))

;;;###autoload
(defun orangu-ask-about-selection ()
  "Attach the region, or the line point is on, to the next prompt in the panel."
  (interactive)
  (let ((selection (if (use-region-p) (orangu--region) (orangu--line)))
        (panel (orangu-panel-buffer (orangu--workspace))))
    (deactivate-mark)
    (pop-to-buffer panel)
    (orangu-panel-attach selection)))

(defvar orangu-implement-history nil "Instructions given to `orangu-implement'.")

;;;###autoload
(defun orangu-implement (instruction &optional anywhere)
  "Have orangu carry out INSTRUCTION in the files of the workspace.
The region goes with it, or else where point is; with a prefix argument,
ANYWHERE, neither does.  The files orangu changes are read again, and what
it changed is shown as a diff, which `orangu-undo-changes' undoes."
  (interactive
   (list (read-string (if (and (use-region-p) (not current-prefix-arg))
                          "Implement, with the region: "
                        "Implement: ")
                      nil 'orangu-implement-history)
         current-prefix-arg))
  (when (string-blank-p instruction)
    (user-error "Say what to implement"))
  (let* ((workspace (orangu--workspace))
         (selection (cond (anywhere nil)
                          ((and (use-region-p) (buffer-file-name (buffer-base-buffer)))
                           (orangu--region))
                          (t (orangu--position))))
         (panel (orangu-panel-buffer workspace)))
    (deactivate-mark)
    (with-current-buffer panel
      (orangu-panel-run
       (orangu-core-implement workspace selection instruction)
       (concat "Implement: " (string-trim instruction)
               (when selection
                 (format "\n(%s)" (orangu-core-selection-label workspace selection))))
       orangu-show-changes))
    (display-buffer panel)))

;;;###autoload
(defun orangu-review-file ()
  "Have orangu review the file of the current buffer, as it is on disk."
  (interactive)
  (let* ((workspace (orangu--workspace))
         (prompt (orangu-core-review-file workspace (orangu--file)))
         (source (current-buffer))
         (panel (orangu-panel-buffer workspace)))
    (orangu-changes-save-buffers workspace)
    (with-current-buffer panel
      (orangu-panel-run
       prompt
       (if (buffer-modified-p source)
           (concat prompt
                   "\n(the file has unsaved changes; orangu reviews it as it was last saved)")
         prompt)))
    (display-buffer panel)))

;;;###autoload
(defun orangu-review-changes ()
  "Have orangu review everything the branch changes."
  (interactive)
  (let ((panel (orangu-panel-buffer (orangu--workspace))))
    (with-current-buffer panel
      (orangu-panel-run orangu-core-review-changes orangu-core-review-changes))
    (display-buffer panel)))

;;; Keys and menus

;;;###autoload (autoload 'orangu-command-map "orangu" nil nil 'keymap)
(defvar orangu-command-map
  (let ((map (make-sparse-keymap)))
    (define-key map "o" #'orangu)
    (define-key map "a" #'orangu-ask-about-selection)
    (define-key map "i" #'orangu-implement)
    (define-key map "f" #'orangu-review-file)
    (define-key map "r" #'orangu-review-changes)
    (define-key map "v" #'orangu-show-changes)
    (define-key map "u" #'orangu-undo-changes)
    (define-key map "c" #'orangu-chat)
    (define-key map "t" #'orangu-open)
    (define-key map "k" #'orangu-stop)
    map)
  "The commands of orangu, to be bound to a key.")
(fset 'orangu-command-map orangu-command-map)

(defvar orangu-menu
  (let ((map (make-sparse-keymap "orangu")))
    (define-key map [undo] '(menu-item "Undo Changes" orangu-undo-changes))
    (define-key map [show] '(menu-item "Show Changes" orangu-show-changes))
    (define-key map [line-2] menu-bar-separator)
    (define-key map [review] '(menu-item "Review Changes" orangu-review-changes))
    (define-key map [file] '(menu-item "Review This File" orangu-review-file
                                       :enable buffer-file-name))
    (define-key map [implement] '(menu-item "Implement..." orangu-implement))
    (define-key map [ask] '(menu-item "Ask About This Selection" orangu-ask-about-selection
                                      :enable buffer-file-name))
    (define-key map [line-1] menu-bar-separator)
    (define-key map [open] '(menu-item "Open orangu" orangu-open))
    (define-key map [chat] '(menu-item "Chat" orangu-chat))
    (define-key map [panel] '(menu-item "Panel" orangu))
    map)
  "The menu of orangu.")

(defun orangu-context-menu (menu _click)
  "Add the commands of orangu to MENU, for `context-menu-functions'."
  (define-key-after menu [orangu] `(menu-item "orangu" ,orangu-menu))
  menu)

;;;###autoload
(define-minor-mode orangu-mode
  "Put orangu in the Tools menu and in the context menu."
  :global t
  :group 'orangu
  (if orangu-mode
      (progn
        (define-key-after global-map [menu-bar tools orangu]
          `(menu-item "orangu" ,orangu-menu))
        (add-hook 'context-menu-functions #'orangu-context-menu 50))
    (define-key global-map [menu-bar tools orangu] nil)
    (remove-hook 'context-menu-functions #'orangu-context-menu)))

(provide 'orangu)
;;; orangu.el ends here
