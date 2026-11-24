;;; orangu-run.el --- Running orangu  -*- lexical-binding: t; -*-

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

;; One run of orangu, with its answer (stdout) and its notes (stderr) handed
;; over as they come, without terminal escapes.  Stopping a run stops
;; everything it started.

;;; Code:

(require 'cl-lib)
(require 'subr-x)

(defconst orangu-run-stop-grace 2
  "Seconds a stopped run gets to end before it is killed outright.")

(defconst orangu-run-drain 0.5
  "Seconds to wait for the last notes after orangu itself has ended.")

;;; Escapes

(defun orangu-run-make-filter ()
  "Return a function that removes terminal escapes from streamed text.
orangu is plain when it is not on a terminal, but a command it runs may
still use colour.  A sequence cut between two pieces is held back."
  (let ((state 'text))
    (lambda (piece)
      (if (and (eq state 'text) (not (string-match-p "[\e\r]" piece)))
          piece
        (let (kept)
          (dolist (c (append piece nil))
            (pcase state
              ('text (cond ((eq c ?\e) (setq state 'escape))
                           ((not (eq c ?\r)) (push c kept))))
              ('escape (setq state (pcase c (?\[ 'csi) (?\] 'osc) (_ 'text))))
              ('csi (when (<= ?@ c ?~) (setq state 'text)))
              ('osc (pcase c (?\a (setq state 'text)) (?\e (setq state 'osc-escape))))
              ('osc-escape (setq state (if (eq c ?\\) 'text 'osc)))))
          (concat (nreverse kept)))))))

;;; Runs

(cl-defstruct (orangu-run (:constructor orangu-run--create) (:copier nil))
  "A run of orangu.  OUTCOME is nil while it goes on."
  process notes-process pid started outcome stopping timer on-answer on-note on-end
  (pending "") last-note)

(cl-defun orangu-run-start (command directory &key on-answer on-note on-end)
  "Start COMMAND, a list, in DIRECTORY and return the run.
ON-ANSWER gets each piece of the answer and ON-NOTE each line of notes.
ON-END gets a plist once it is over: :outcome is `succeeded', `failed' or
`cancelled', :error says why, and :elapsed is in seconds.  A COMMAND that
cannot be started at all signals an error, or fails, as Emacs sees fit."
  (let* ((default-directory (file-name-as-directory directory))
         (process-environment (cons "NO_COLOR=1" process-environment))
         (run (orangu-run--create :started (float-time) :on-answer on-answer
                                  :on-note on-note :on-end on-end))
         (answer-filter (orangu-run-make-filter))
         (note-filter (orangu-run-make-filter))
         (notes (make-pipe-process
                 :name "orangu notes" :buffer nil :noquery t :coding 'utf-8-unix
                 :filter (lambda (_ piece)
                           (orangu-run--take-notes run (funcall note-filter piece)))
                 :sentinel (lambda (&rest _) (orangu-run--settle run)))))
    (setf (orangu-run-notes-process run) notes)
    (condition-case err
        (let ((process (make-process
                        :name "orangu" :buffer nil :noquery t :command command
                        :connection-type 'pipe :coding 'utf-8-unix :stderr notes
                        :filter (lambda (_ piece)
                                  (let ((text (funcall answer-filter piece)))
                                    (unless (or (string-empty-p text) (orangu-run-outcome run))
                                      (funcall (orangu-run-on-answer run) text))))
                        :sentinel (lambda (process _)
                                    (unless (process-live-p process)
                                      (orangu-run--ended run))))))
          (setf (orangu-run-process run) process
                (orangu-run-pid run) (process-id process))
          ;; Nothing is ever typed into a one-shot.
          (ignore-errors (process-send-eof process))
          run)
      (error (delete-process notes)
             (signal (car err) (cdr err))))))

(defun orangu-run-live-p (run)
  "Non-nil while RUN goes on."
  (and run (null (orangu-run-outcome run))))

(defun orangu-run-stop (run)
  "Stop RUN and everything it started."
  (when (and (orangu-run-live-p run) (not (orangu-run-stopping run)))
    (setf (orangu-run-stopping run) t)
    (orangu-run--signal run 'SIGTERM)
    ;; orangu may go while something it started stays, so this happens
    ;; whether or not orangu has ended by then.
    (run-at-time orangu-run-stop-grace nil #'orangu-run--signal run 'SIGKILL)))

(defun orangu-run--signal (run signal)
  "Send SIGNAL to RUN and everything it started."
  (let ((pid (orangu-run-pid run)))
    (if (eq system-type 'windows-nt)
        (when (process-live-p (orangu-run-process run))
          (call-process "taskkill" nil 0 nil "/pid" (number-to-string pid) "/T" "/F"))
      ;; Emacs starts a process as the leader of its own group.  A group
      ;; that is already gone is no error.
      (ignore-errors (signal-process (- pid) signal)))))

(defun orangu-run--take-notes (run text)
  "Hand the whole lines of RUN's notes over, now that TEXT has come."
  (let* ((lines (split-string (concat (orangu-run-pending run) text) "\n"))
         (rest (car (last lines))))
    (setf (orangu-run-pending run) rest)
    (dolist (line (butlast lines))
      (orangu-run--note run line))))

(defun orangu-run--note (run line)
  "Hand LINE of RUN's notes over, unless it is blank."
  (let ((line (string-trim-right line)))
    (unless (or (string-empty-p line) (orangu-run-outcome run))
      (setf (orangu-run-last-note run) line)
      (funcall (orangu-run-on-note run) line))))

(defun orangu-run--ended (run)
  "The process of RUN has ended; its last notes may still be on the way."
  (unless (or (orangu-run-outcome run) (orangu-run-timer run))
    ;; Something orangu left running may hold its stderr open.
    (setf (orangu-run-timer run)
          (run-at-time orangu-run-drain nil #'orangu-run--finish run))
    (orangu-run--settle run)))

(defun orangu-run--settle (run)
  "Finish RUN when both its process and its notes have ended."
  (when (and (orangu-run-process run)
             (not (process-live-p (orangu-run-process run)))
             (not (process-live-p (orangu-run-notes-process run))))
    (orangu-run--finish run)))

(defun orangu-run--finish (run)
  "End RUN, once."
  (unless (orangu-run-outcome run)
    (when (orangu-run-timer run)
      (cancel-timer (orangu-run-timer run)))
    (orangu-run--note run (orangu-run-pending run))
    (let* ((process (orangu-run-process run))
           (code (process-exit-status process))
           (outcome (cond ((orangu-run-stopping run) 'cancelled)
                          ((and (eq (process-status process) 'exit) (zerop code)) 'succeeded)
                          (t 'failed))))
      (setf (orangu-run-outcome run) outcome)
      (delete-process (orangu-run-notes-process run))
      (funcall (orangu-run-on-end run)
               (list :outcome outcome
                     :elapsed (- (float-time) (orangu-run-started run))
                     :error (when (eq outcome 'failed)
                              (or (orangu-run-last-note run)
                                  (format "orangu exited with %s" code))))))))

(provide 'orangu-run)
;;; orangu-run.el ends here
