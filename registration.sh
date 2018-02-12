data='/data2/HBNcore/CMI_HBN_Data/MRI/RU/data/'
ants_out='/data2/HBNcore/CMI_HBN_Data/MRI/RU/CPAC/output/pipeline_RU_CPAC/'
ants_work='/data2/HBNcore/CMI_HBN_Data/MRI/RU/CPAC/working/'
outpath='/data2/Projects/Jake/Registration_PEER'

cd $outpath

rm copy_raw.txt
rm copy_ants.txt

cd $data

##########################################################################
# Copy raw peer data

#for sub in $(ls);do
for sub in 'sub-5002891';do

	n=`find "$data/$sub/func" -iname '*peer*' -iname '*nii.gz*' |wc -l`

	if [ $n -gt "1" ]; then

	for nifti_file in $(ls $data'/'$sub'/'func/*.nii.gz);do

		if [[ $nifti_file == *"peer_run-1"* ]];then
			echo "cp $nifti_file $outpath/$sub/raw_peer_1.nii.gz" >> $outpath/copy_raw.txt


		elif [[ $nifti_file == *"peer_run-2"* ]];then
			echo "cp $nifti_file $outpath/$sub/raw_peer_2.nii.gz" >> $outpath/copy_raw.txt


		elif [[ $nifti_file == *"peer_run-3"* ]];then
			echo "cp $nifti_file $outpath/$sub/raw_peer_3.nii.gz" >> $outpath/copy_raw.txt

		fi

	done

	fi

		mkdir $outpath'/'$sub

done

cat $outpath/copy_raw.txt | parallel -j 5

##########################################################################
# Copy C-PAC outputs for ANTs registration

cd $ants_out

#for sub in $(ls);do 
for sub in 'sub-5002891_ses-1';do

	deskull=$ants_out/$sub/anatomical_brain/*
	skull=$ants_out/$sub/anatomical_reorient/*
	fmean1=$ants_out/$sub/mean_functional/_scan_peer_run-1/*
	fmean2=$ants_out/$sub/mean_functional/_scan_peer_run-2/*
	fmean3=$ants_out/$sub/mean_functional/_scan_peer_run-3/*

	outsub=$(echo $sub | cut -f1 -d"_")

	echo "cp $deskull $outpath/no_skull.nii.gz" >> $outpath/$outsub/copy_out.txt
	echo "cp $skull $outpath/skull.nii.gz" >> $outpath/$outsub/copy_out.txt
	echo "cp $fmean1 $outpath/fmean1.nii.gz" >> $outpath/$outsub/copy_out.txt
	echo "cp $fmean2 $outpath/fmean2.nii.gz" >> $outpath/$outsub/copy_out.txt
	echo "cp $fmean3 $outpath/fmean3.nii.gz" >> $outpath/$outsub/copy_out.txt

done

cat $outpath/copy_out.txt | parallel -j 5

cd $ants_work

for sub in 'resting_preproc_sub-5002891_ses-1';do

	outsub=$(echo $sub | cut -f3 -d"_")
	outsub=${outsub#*preproc_}

	segments=$ants_work/$sub/seg_preproc_0/WM/WM_mask/segment_seg_2_maths.nii.gz

	echo "cp $segments $outpath/segments.nii.gz" >> $outpath/$outsub/copy_working.txt

done

cat $outpath/copy_working.txt | parallel -j 5

##########################################################################
# ANTs registration - anatomical to template

cd $outpath

template='/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz'

for subject in $(ls);do

	echo "" >> anat_to_temp.txt

done

antsRegistration --collapse-output-transforms 0 --dimensionality 3 --initial-moving-transform [$template,$outpath/$subject/'no_skull.nii.gz',0] --interpolation Linear --output [transform, transform_Warped.nii.gz] --transform Rigid[0.1] --metric MI[$template, $outpath/$subject/'no_skull.nii.gz', 1, 32, Regular, 0.25] --convergence [1000x500x250x100, 1e-08, 10] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 8x4x2x1 --use-histogram-matching 1 --transform Affine[0.1] --metric MI[$template, $outpath/$subject/'no_skull.nii.gz', 1, 32, Regular, 0.25] --convergence [1000x500x250x100, 1e-08, 10] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 8x4x2x1 --use-histogram-matching 1 --transform SyN[.1, 3.0, 0.0] --metric CC[$template, $outpath/$subject/'skull.nii.gz', 1, 4] --convergence [100x100x70x20, 1e-09, 15] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 6x4x2x1 --use-histogram-matching 1 --winsorize-image-intensities [.01, .99]

















