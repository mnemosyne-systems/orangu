#!/bin/sh
# Stands in for orangu in the tests of orangu-run.el. It is called as
# orangu is, `-w WORKSPACE -p PROMPT`, and the prompt says what to do.

case "$4" in
  fail)
    printf 'Using the stand-in server\n' >&2
    printf 'error: the stand-in server is not running\n' >&2
    exit 3 ;;
  stream)
    # An escape sequence cut between two pieces of the answer.
    printf 'Hel'
    sleep 0.2
    printf 'lo, \033[1'
    sleep 0.2
    printf ';31mworld\033[0m!\r\n' ;;
  sleep)
    # Something orangu started, which holds the output open.
    sleep 30 &
    printf 'child %s\n' "$!" >&2
    wait ;;
  *)
    printf 'args:'
    for argument in "$@"; do printf ' [%s]' "$argument"; done
    printf '\ncwd: %s\nNO_COLOR: %s\n' "$PWD" "$NO_COLOR"
    if read -r _; then echo 'stdin: open'; else echo 'stdin: closed'; fi
    printf 'Using the \033[2mstand-in\033[0m server\n' >&2
    printf 'Done: 1 request\n' >&2 ;;
esac
