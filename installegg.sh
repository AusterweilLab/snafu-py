#!/bin/sh

# copies rw package to python egg directory so that you can `import rw` from sub-directories
# do this when you update the `rw` package

DEST='/Library/Python/2.7/site-packages/rw'
sudo rm -rf $DEST
sudo cp -r rw $DEST
