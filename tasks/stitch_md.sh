#!/bin/bash

# Markdown File Stitching Script
# This script provides different ways to attach one markdown file to another

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo -e "${BLUE}Usage: $0 [OPTIONS] <target_file> <source_file>${NC}"
    echo ""
    echo "Options:"
    echo "  -a, --append     Simply append source to target (default)"
    echo "  -s, --separator  Add a separator line between files"
    echo "  -t, --title      Add a title header for the appended content"
    echo "  -n, --new-page   Add a page break before appending"
    echo "  -b, --backup     Create backup of target file before modifying"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 README.md docs/subreadme.md"
    echo "  $0 -s -b README.md docs/subreadme.md"
    echo "  $0 -t \"Additional Documentation\" README.md docs/subreadme.md"
}

# Default options
APPEND_MODE="simple"
CREATE_BACKUP=false
SEPARATOR=""
TITLE=""
PAGE_BREAK=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--append)
            APPEND_MODE="simple"
            shift
            ;;
        -s|--separator)
            APPEND_MODE="separator"
            shift
            ;;
        -t|--title)
            APPEND_MODE="title"
            TITLE="$2"
            shift 2
            ;;
        -n|--new-page)
            PAGE_BREAK=true
            shift
            ;;
        -b|--backup)
            CREATE_BACKUP=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Check if we have the required arguments
if [[ $# -lt 2 ]]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    usage
    exit 1
fi

TARGET_FILE="$1"
SOURCE_FILE="$2"

# Validate files
if [[ ! -f "$TARGET_FILE" ]]; then
    echo -e "${RED}Error: Target file '$TARGET_FILE' does not exist${NC}"
    exit 1
fi

if [[ ! -f "$SOURCE_FILE" ]]; then
    echo -e "${RED}Error: Source file '$SOURCE_FILE' does not exist${NC}"
    exit 1
fi

# Create backup if requested
if [[ "$CREATE_BACKUP" == true ]]; then
    BACKUP_FILE="${TARGET_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
    cp "$TARGET_FILE" "$BACKUP_FILE"
    echo -e "${GREEN}Backup created: $BACKUP_FILE${NC}"
fi

# Function to append with simple concatenation
append_simple() {
    echo "" >> "$TARGET_FILE"
    cat "$SOURCE_FILE" >> "$TARGET_FILE"
}

# Function to append with separator
append_with_separator() {
    echo "" >> "$TARGET_FILE"
    echo "---" >> "$TARGET_FILE"
    echo "" >> "$TARGET_FILE"
    cat "$SOURCE_FILE" >> "$TARGET_FILE"
}

# Function to append with title
append_with_title() {
    echo "" >> "$TARGET_FILE"
    if [[ -n "$TITLE" ]]; then
        echo "## $TITLE" >> "$TARGET_FILE"
    else
        # Extract filename without extension as default title
        BASENAME=$(basename "$SOURCE_FILE" .md)
        echo "## ${BASENAME^}" >> "$TARGET_FILE"
    fi
    echo "" >> "$TARGET_FILE"
    cat "$SOURCE_FILE" >> "$TARGET_FILE"
}

# Function to append with page break
append_with_page_break() {
    echo "" >> "$TARGET_FILE"
    echo '<div style="page-break-before: always;"></div>' >> "$TARGET_FILE"
    echo "" >> "$TARGET_FILE"
    cat "$SOURCE_FILE" >> "$TARGET_FILE"
}

# Main execution
echo -e "${BLUE}Attaching '$SOURCE_FILE' to '$TARGET_FILE'...${NC}"

# Add page break if requested
if [[ "$PAGE_BREAK" == true ]]; then
    append_with_page_break
fi

# Apply the selected append mode
case $APPEND_MODE in
    "simple")
        append_simple
        ;;
    "separator")
        append_with_separator
        ;;
    "title")
        append_with_title
        ;;
esac

echo -e "${GREEN}Successfully attached '$SOURCE_FILE' to '$TARGET_FILE'${NC}"

# Show a preview of what was added
echo -e "${YELLOW}Preview of appended content:${NC}"
echo "----------------------------------------"
tail -n 10 "$TARGET_FILE"
echo "----------------------------------------"