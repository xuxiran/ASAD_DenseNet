import urllib.request
import os
import tarfile

download = 1



if download == 1:
    if not os.path.exists('../2_data'):
        os.makedirs('../2_data')
    for i in range(1,17):
        url = 'https://zenodo.org/record/3377911/files/S{}.mat?download=1'.format(i)
        filename = '../2_data/S{}.mat'.format(i)
        print('Downloading {}...'.format(filename))
        urllib.request.urlretrieve(url, filename)


print('Done.')
