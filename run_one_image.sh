#!/bin/bash
set -x
set -e

if [[ -z `python --version 2>&1 | grep 2.7` ]]; then
	source activate py2_env
fi
if [[ -z `python --version 2>&1 | grep 2.7` ]]; then
	echo "WARNING: not python 2.7, probably won't work!"
fi

input="$1"
pigments="$2"
if [[ -z $pigments ]]; then
	pigments=6
fi

dir="`basename ${input%.*}`-$pigments"
image="`basename ${input%.*}.png`"
log="log.txt"
html="index.html"

prefix="primary_pigments_color_vertex-${pigments}-KM_weights-W_w_10.0-W_sparse_0.1-W_spatial_1.0-choice_0-blf-W_neighbors_0.0-Recursive_Yes"

if [[ -d $dir ]]; then
	exit 1
fi

mkdir -p $dir
cp wheatfield-crop/Existing* $dir
convert $input -resize 600x600 $dir/$image
rm -f $dir/$log
rm -f $dir/$html

python \
	step1_ANLS_with_autograd.py \
	$image \
	Existing_KS_parameter_KS.txt \
	2 \
	None \
	sampled_pixels-400 \
	0 \
	$pigments \
	10.0 \
	0.0 \
	0.0 \
	0.001 \
	0.001 \
	1e-6 \
	/$dir \
	None \
	0 \
	1 \
	1000 \
	400 2>&1 | tee -a $dir/$log

cd $dir

python \
	../Solve_KM_mixing_model_fixed_KS_with_autograd.py \
	$image \
	primary_pigments_KS-${pigments}.txt  \
	None \
	$prefix \
	10.0 \
	0.1 \
	0 \
	1.0 \
	0.0 \
	blf \
	Yes 2>&1 | tee -a $log

cat <<EOF >>$html
<html>
<body>
<pre>

$prefix

EOF

grep -A5 "RMSE" $log >>$html

cat <<EOF >>$html

</pre>
<h1>Pigments</h1>
<img src="primary_pigments_color-${pigments}.png"><br>
<table width="100%">
<tr><th>K</th>
  <th>S</th>
  <th>K/S</th>
  <th>R</th></tr>
EOF

for i in $(seq $pigments); do
	p=$((i-1))
	cat <<EOF >>$html
<tr><td><img width="100%" src="primary_pigments_K_curve-${p}.png"></td>
  <td><img width="100%" src="primary_pigments_S_curve-${p}.png"></td>
  <td><img width="100%" src="primary_pigments_KS_curve-${p}.png"></td>
  <td><img width="100%" src="primary_pigments_R_curve-${p}.png"></td></tr>
EOF
done

cat <<EOF >>$html
</table>
<h1>Reconstruction</h1>
<table width="100%">
<tr><th>Input</th>
  <th>Reconstructed</th></tr>
<tr><td><img width="100%" src="${image}"></td>
  <td><img width="100%" src="${prefix}-final_recursivelevel--fixed_KS-reconstructed.png"</td></tr>
</table>
<h1>Mixing Weights</h1>
<img src="primary_pigments_color-${pigments}.png"><br>
<table width="100%">
EOF

for i in $(seq 1 2 $pigments); do
	p=$((i-1))
	q=$((p+1))
	cat <<EOF >>$html
<tr><td><img width="100%" src="${prefix}-final_recursivelevel--mixing_weights_map-0${p}.png"></td>
  <td><img width="100%" src="${prefix}-final_recursivelevel--mixing_weights_map-0${q}.png"></td></tr>
EOF
done

cat <<EOF >>$html
</table>
</body>
</html>
EOF
