grep Execution 100000000_16thr.log | awk '{print $4}' | sed 's/.$//' | awk 'ORS=NR%3?"\t":"\n"'
