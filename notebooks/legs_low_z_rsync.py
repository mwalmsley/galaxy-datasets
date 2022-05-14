import pandas as pd


if __name__ == '__main__':

    # note - the low z (z<0.1) catalog!
    df = pd.read_parquet('/home/walml/repos/decals-rings/results/rings_megacatalog_z_below_0p1.parquet', columns=['dr8_id', 'iauname', 'redshift'])

    df['id_str'] = df['dr8_id']
    df['brickid'] = df['id_str'].apply(lambda x: x.split('_')[0])
    df['objid'] = df['id_str'].apply(lambda x: x.split('_')[1])
    # relative to rsync root
    df['file_loc'] = df['brickid'] + '/' + df['id_str'] + '.jpg'

    print(df['file_loc'][0])

    all_files = list(df['file_loc'])
    print(all_files[:3])

    all_files_path = '/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt'

    with open(all_files_path, 'w') as f:
        f.write('\n'.join(all_files))

    print(f'rsync --dry-run --files-from {all_files_path} walml@galahad.ast.man.ac.uk:/share/nas2/walml/galaxy_zoo/decals/dr8/jpg  /home/walml/repos/pytorch-galaxy-datasets/derived_data/legs_z_below_0p1_to_upload')