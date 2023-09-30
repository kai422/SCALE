import argparse
import os
import zipfile

import gdown

benchmarks_dict = {
    'imagenet-1k': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 
    ],
}

dir_dict = {
    'images_classic/': [ 'texture'],
    'images_largescale/': [
        'imagenet_1k',
        'species_sub',
        'ssb_hard',
        'ninco',
        'inaturalist',
        'openimage_o',
    ],
}

download_id_dict = {
    'imagenet_res50_v1.5': '15PdDMNRfnJ7f2oxW6lI-Ge4QJJH3Z0Fy',
    'benchmark_imglist': '1XKzBdWCqg3vPoj-D32YixJyJJ0hL63gP',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
}


def require_download(filename, path):
    for item in os.listdir(path):
        if item.startswith(filename) or filename.startswith(
                item) or path.endswith(filename):
            return False

    else:
        print(filename + ' needs download:')
        return True


def download_dataset(dataset, args):
    for key in dir_dict.keys():
        if dataset in dir_dict[key]:
            store_path = os.path.join(args.save_dir[0], key, dataset)
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            break
    else:
        print('Invalid dataset detected {}'.format(dataset))
        return

    if require_download(dataset, store_path):
        print(store_path)
        if not store_path.endswith('/'):
            store_path = store_path + '/'
        gdown.download(id=download_id_dict[dataset], output=store_path)

        file_path = os.path.join(store_path, dataset + '.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(store_path)
        os.remove(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download datasets and checkpoints')
    parser.add_argument('--contents',
                        nargs='+',
                        default=['datasets', 'checkpoints'])
    parser.add_argument('--datasets', nargs='+', default=['default'])
    parser.add_argument('--checkpoints', nargs='+', default=['all'])
    parser.add_argument('--save_dir',
                        nargs='+',
                        default=['./data', './results'])
    parser.add_argument('--dataset_mode', default='default')
    args = parser.parse_args()

    args.datasets = ['imagenet-1k']
    args.checkpoints = ['imagenet_res50_v1.5']

    for content in args.contents:
        if content == 'datasets':

            store_path = args.save_dir[0]
            if not store_path.endswith('/'):
                store_path = store_path + '/'
            if not os.path.exists(os.path.join(store_path,
                                               'benchmark_imglist')):
                gdown.download(id=download_id_dict['benchmark_imglist'],
                               output=store_path)
                file_path = os.path.join(args.save_dir[0],
                                         'benchmark_imglist.zip')
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    zip_file.extractall(store_path)
                os.remove(file_path)

            if args.dataset_mode == 'default' or \
                    args.dataset_mode == 'benchmark':
                for benchmark in args.datasets:
                    for dataset in benchmarks_dict[benchmark]:
                        download_dataset(dataset, args)

            if args.dataset_mode == 'dataset':
                for dataset in args.datasets:
                    download_dataset(dataset, args)

        elif content == 'checkpoints':
            if 'v1.5' in args.checkpoints[0]:
                store_path = args.save_dir[1]
            else:
                store_path = os.path.join(args.save_dir[1], 'checkpoints/')
            if not os.path.exists(store_path):
                os.makedirs(store_path)

            if not store_path.endswith('/'):
                store_path = store_path + '/'

            for checkpoint in args.checkpoints:
                if require_download(checkpoint, store_path):
                    gdown.download(id=download_id_dict[checkpoint],
                                   output=store_path)
                    file_path = os.path.join(store_path, checkpoint + '.zip')
                    with zipfile.ZipFile(file_path, 'r') as zip_file:
                        zip_file.extractall(store_path)
                    os.remove(file_path)