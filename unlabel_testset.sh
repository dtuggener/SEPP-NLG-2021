#!/bin/bash

indir=$1
outdir=$2
datadir=$3  # "test" or "surprise_test"

for lang in en de fr it;
	do
	  mkdir $outdir/$lang/
	  mkdir $outdir/$lang/$datadir
	done

echo $indir $outdir

for fname in $(ls $indir/*/"$datadir"/*tsv)
	do
	  echo $fname
	  outfile=$(echo $fname | sed "s|$indir|$outdir|g")
	  echo $outfile
	  awk '{print $1 "\t0\t0"}' $fname > $outfile
	done
