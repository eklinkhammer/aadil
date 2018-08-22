#!/usr/bin/env python3

import sys

counts = {}
for line in sys.stdin:
    line = line.rstrip()
    if line in counts:
        counts[line] += 1
    else:
        counts[line] = 1

for k,v in counts.items():
    print (str(k) + " " + str(v))
        
