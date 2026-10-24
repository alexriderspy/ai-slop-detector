#!/bin/bash
name="$1"; shift
out="media/terminal/$name.log"
{ echo "sreemanti@MacBook-Pro ai-slop-detector % $*"; "$@" 2>&1; } | tee "$out"
