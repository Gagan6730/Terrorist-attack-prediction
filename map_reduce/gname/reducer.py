#!/usr/bin/python
import sys
country_total = 0
oldKey = None
for line in sys.stdin:
    data = line.strip().split("\t")
    if len(data) != 2:
        continue
    thisKey, thisCountry = data
    if oldKey and oldKey != thisKey:
        print(oldKey,"\t",country_total)
        oldKey = thisKey
        country_total = 0
    oldKey = thisKey
    country_total += float(thisCountry)
if oldKey != None:
    print(oldKey, "\t", country_total)

