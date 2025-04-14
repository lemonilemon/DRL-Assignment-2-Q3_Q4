#!/bin/bash
DROPBOX_LINK="https://www.dropbox.com/scl/fi/4swsfn1xce9ixbc6ois76/2048.bin?rlkey=17dec6efou3qm4ssea1mntukz&st=jgr5f9f9&dl=0"
OUTPUT_FILE="game2048/2048.bin"

curl -L "$DROPBOX_LINK" -o "$OUTPUT_FILE"
