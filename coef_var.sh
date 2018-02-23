data_path='/data2/Projects/Jake/Resampled'
outpath='/data2/Projects/Jake/coef_var'

rm $outpath/remove_vol.txt
rm $outpath/add_all.txt
rm $outpath/add_all2.txt

for sub in 'sub-5986705' 'sub-5375858' 'sub-5292617' 'sub-5397290' 'sub-5844932' 'sub-5787700' 'sub-5797959' 'sub-5378545' 'sub-5085726' 'sub-5984037' 'sub-5076391' 'sub-5263388' 'sub-5171285' 'sub-5917648' 'sub-5814325' 'sub-5169146' 'sub-5484500' 'sub-5481682' 'sub-5232535' 'sub-5905922' 'sub-5975698' 'sub-5986705' 'sub-5343770';do

	echo "fslroi $data_path/$sub/peer1_resampled.nii.gz $outpath/${sub}_peer1.nii.gz 0 -1 0 -1 0 -1 67 1" >> $outpath/remove_vol.txt

	echo "fslroi $data_path/$sub/peer2_resampled.nii.gz $outpath/${sub}_peer2.nii.gz 0 -1 0 -1 0 -1 67 1" >> $outpath/remove_vol.txt

	echo "fslroi $data_path/$sub/peer3_resampled.nii.gz $outpath/${sub}_peer3.nii.gz 0 -1 0 -1 0 -1 67 1" >> $outpath/remove_vol.txt

done

#cat $outpath/remove_vol.txt | parallel -j 10

for sub in 'sub-5986705' 'sub-5375858' 'sub-5292617' 'sub-5397290' 'sub-5844932' 'sub-5787700' 'sub-5797959' 'sub-5378545' 'sub-5085726' 'sub-5984037' 'sub-5076391' 'sub-5263388' 'sub-5171285' 'sub-5917648' 'sub-5814325' 'sub-5169146' 'sub-5484500' 'sub-5481682' 'sub-5232535' 'sub-5905922' 'sub-5975698' 'sub-5986705' 'sub-5343770';do

	echo "fslmerge -t $outpath/added/${sub}_combined.nii.gz $outpath/${sub}_peer1.nii.gz $outpath/${sub}_peer2.nii.gz $outpath/${sub}_peer3.nii.gz" >> $outpath/add_all.txt

done

#cat $outpath/add_all.txt | parallel -j 10

echo "fslmerge -t $outpath/total.nii.gz" >> $outpath/add_all2.txt

for sub in 'sub-5986705' 'sub-5375858' 'sub-5292617' 'sub-5397290' 'sub-5844932' 'sub-5787700' 'sub-5797959' 'sub-5378545' 'sub-5085726' 'sub-5984037' 'sub-5076391' 'sub-5263388' 'sub-5171285' 'sub-5917648' 'sub-5814325' 'sub-5169146' 'sub-5484500' 'sub-5481682' 'sub-5232535' 'sub-5905922' 'sub-5975698' 'sub-5986705' 'sub-5343770';do

	echo -n " $outpath/added/${sub}_combined.nii.gz" >> $outpath/add_all2.txt

done

#cat $outpath/add_all2.txt | parallel -j 10











