#!/usr/bin/python
import sys

for line in sys.stdin:
    data=line.strip().split(",")
    if len(data)==13:
        imonth,iday,country_txt,region_txt,provstate,city,success,suicide,attacktype1_txt,natlty1_txt,weaptype1_txt,gname,targtype1_txt=data
        print("{0}\t{1}".format(gname,1))



# hadoop jar /usr/local/Cellar/hadoop/3.1.1/libexec/share/hadoop/tools/lib/hadoop-streaming-3.1.1.jar -mapper mapper.py -reducer reducer.py -file mapper.py -file reducer.py -input /newdir -output /jobout