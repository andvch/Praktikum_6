reset
n=30				#number of intervals
max=0.005			#max value
min=0.				#min value
width=(max-min)/n	#interval width

hist(x,width)=width*floor(x/width)+width/2.0

set term png
set output "histogram.png"
set xrange [min:max]
set yrange [0:]

set xtics min,(max-min)/5,max
set boxwidth width*0.9
set style fill solid 0.5
set tics out nomirror
set xlabel "1-F"
set ylabel "N-observ"
set title "n = 26; e = 0.01"

plot "report" u (hist($5,width)):(1.0) smooth freq w boxes lc rgb"blue" notitle
