#! /bin/bash 
# set-env.sh — update VAR in an .env file, with quoting control
# Usage:
#   ./set-env.sh VAR NEW_VALUE [FILE] [--quote auto|double|single|raw]
#   ./set-env.sh VAR NEW_VALUE --quote double               # FILE defaults to .env
#   ./set-env.sh VAR NEW_VALUE FILE --quote=double

set -eu

usage() {
  echo "Usage: $0 VAR NEW_VALUE [FILE] [--quote auto|double|single|raw]" >&2
  exit 2
}

[ "$#" -ge 2 ] || usage

VAR=${1-}
VAL=${2-}
shift 2

FILE=".env"
QUOTE_MODE="auto"

# Parse remaining args: FILE and/or --quote…
while [ "$#" -gt 0 ]; do
  case "$1" in
    --quote)
      [ "$#" -ge 2 ] || { echo "--quote requires an argument" >&2; exit 2; }
      QUOTE_MODE=$2
      shift 2
      ;;
    --quote=*)
      QUOTE_MODE=${1#--quote=}
      shift
      ;;
    *)
      FILE=$1
      shift
      ;;
  esac
done

[ -f "$FILE" ] || { echo "Error: $FILE not found" >&2; exit 1; }

# backup
cp "$FILE" "$FILE.bak"

tmp="$(mktemp)"
awk -v K="$VAR" -v V="$VAL" -v MODE="$QUOTE_MODE" '
function esc_dq(s,  t){ t=s; gsub(/\\/,"\\\\",t); gsub(/"/,"\\\"",t); return t }
BEGIN {
  re = "^[[:space:]]*(export[[:space:]]+)?";
  re = re K "[[:space:]]*=";
}
# keep full-line comments
/^[[:space:]]*#/ { print; next }

{
  if ($0 ~ re) {
    q = ""
    # decide quoting
    if (MODE == "double")      q="\""
    else if (MODE == "single") q="'\''"
    else if (MODE == "raw")    q=""
    else { # auto: detect existing quoting char after '=' and spaces
      i = index($0, "=")
      j = i + 1
      while (j <= length($0) && substr($0,j,1) ~ /[ \t]/) j++
      c = substr($0, j, 1)
      if (c == "\"" || c == "'\''") q = c
      # if existing was single but new value contains a single quote, switch to double
      if (q == "'\''" && V ~ /'\''/) q="\""
    }
    if (q == "\"")        print K "=\"" esc_dq(V) "\""
    else if (q == "'\''") print K "='\''" V "'\''"
    else                  print K "=" V
    next
  }
  print
}
' "$FILE" > "$tmp" && mv "$tmp" "$FILE"
