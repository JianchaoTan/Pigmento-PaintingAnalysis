#!/bin/bash
set -x
set -e

# runs all images in images/ directory.
# puts per-image results in output/ directory.
# puts aggregate results in webpage/ directory.

mkdir -p webpage
mkdir -p output

html="webpage/index.html"

rows="/tmp/rows.html"
rm -f $rows
touch $rows

for p in 5 4 6; do
	for image in images/*; do
		./run_one_image.sh $image $p

		base="`basename ${image%.*}`"
		input="${base}.png"
		dir="${base}-${p}"

		pattern="$dir/*-final_recursivelevel--fixed_KS-reconstructed.png"
		files=( $pattern )
		output="${files[0]}" 

		rmse="`grep RMSE $dir/log.txt | sed 's/RGB\ RMSE://' | tr -d ' ' | tr '\n' ' '`"

		cat <<EOF >>$rows
<tr> \
  <td><pre><a href="$dir/index.html">$dir</a></pre></td> \
  <td><img width="300" src="$dir/$input"></td> \
  <td><img height="50" style="border:10px solid gray" src="$dir/primary_pigments_color-${p}.png"></td> \
  <td><pre>$rmse</pre></td> \
</tr>
EOF

		mkdir webpage/$dir
		cp $dir/$input \
			$dir/index.html \
			$dir/log.txt \
			$dir/primary_Pigments_color-${p}.png \
			$dir/primary_pigments_*_curve-*.png \
			$dir/*-final_recursivelevel--fixed_KS-reconstructed.png \
			$dir/*-final_recursivelevel--mixing_weights_map-*.png \
			webpage/$dir
		mv $dir output/
	done
done



cat <<EOF >$html
<html>
<body>
<table width="100%">
EOF

cat $rows | sort >>$html

cat <<EOF >>$html
</table>
</body>
</html>
EOF
