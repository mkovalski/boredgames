#!/bin/bash
cd evaluate
ffmpeg -r 15  -s 1920x1080 -i image_%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
cd ../
