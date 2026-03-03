"""Code for getting the data loaders.

Extended to support hyperspectral datasets from `mambavision/Data` via:
- pad_with_zeros
- create_image_cubes
- apply_pca (optional)
- HyperspectralDataset
- get_hsi_loaders
"""

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
def _accumulate(iterable, fn=lambda x, y: x + y):
    """Accumulate function for compatibility with newer PyTorch versions."""
    it = iter(iterable)
    try:
        acc = next(it)
    except StopIteration:
        return
    yield acc
    for element in it:
        acc = fn(acc, element)
        yield acc
from timm.data import IterableImageDataset, ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
import os
import scipy.io as sio


def get_loaders(args, mode='eval', dataset=None):
    """Get data loaders for required dataset."""
    if dataset is None:
        dataset = args.dataset
    if dataset == 'imagenet':
        return get_imagenet_loader(args, mode)
    if dataset == 'hsi':
        return get_hsi_loaders(args)
    else:
        if mode == 'search':
            return get_loaders_search(args)
        elif mode == 'eval':
            return get_loaders_eval(dataset, args)


class Subset_imagenet(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset , indices) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = None

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.indices)


def get_loaders_eval(dataset, args):
    """Get train and valid loaders for cifar10/tiny imagenet."""

    if dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'cifar100':
        num_classes = 100
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(
            root=args.data, train=False, download=True, transform=valid_transform)

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)

        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=16)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False,
        sampler=valid_sampler, pin_memory=True, num_workers=16)

    return train_queue, valid_queue, num_classes


def get_loaders_search(args):
    """Get train and valid loaders for cifar10/tiny imagenet."""

    if args.dataset == 'cifar10':
        num_classes = 10
        train_transform, _ = _data_transforms_cifar10(args)
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_transform, _ = _data_transforms_cifar10(args)
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    print('Found %d samples' % (num_train))
    sub_num_train = int(np.floor(args.train_portion * num_train))
    sub_num_valid = num_train - sub_num_train

    sub_train_data, sub_valid_data = my_random_split(
        train_data, [sub_num_train, sub_num_valid], seed=0)
    print('Train: Split into %d samples' % (len(sub_train_data)))
    print('Valid: Split into %d samples' % (len(sub_valid_data)))

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            sub_train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            sub_valid_data)

    train_queue = torch.utils.data.DataLoader(
        sub_train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=16, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        sub_valid_data, batch_size=args.batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=16, drop_last=True)

    return train_queue, valid_queue, num_classes

################################################################################
# ImageNet
################################################################################
def get_imagenet_loader(args, mode='eval', testdir = ""):
    """Get train/val for imagenet."""
    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    print("verify testing path")
    if len(testdir) < 2:
        testdir = os.path.join("../ImageNetV2/", 'test')
        # print("\n\n\n loading imagenet v2 \n\n\n")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    downscale = 1
    val_transform = transforms.Compose([
        transforms.Resize(args.resize//downscale),
        transforms.CenterCrop(args.resolution//downscale),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.resolution//downscale),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if mode == 'eval':
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=train_transform)
            valid_data = imagenet_lmdb_dataset(
                validdir, transform=val_transform)
        else:
            train_data = dset.ImageFolder(traindir, transform=train_transform)
            valid_data = dset.ImageFolder(validdir, transform=val_transform)

        train_sampler, valid_sampler = None, None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data)

            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_data)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            pin_memory=True, num_workers=16, sampler=train_sampler, drop_last=True)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=(valid_sampler is None),
            pin_memory=True, num_workers=16, sampler=valid_sampler)
    elif mode == 'search':
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=val_transform)
        else:
            train_data = dset.ImageFolder(traindir, val_transform)

        num_train = len(train_data)
        print('Found %d samples' % (num_train))
        sub_num_train = int(np.floor(args.train_portion * num_train))
        sub_num_valid = num_train - sub_num_train

        sub_train_data, sub_valid_data = my_random_split(
            train_data, [sub_num_train, sub_num_valid], seed=0)
        print('Train: Split into %d samples' % (len(sub_train_data)))
        print('Valid: Split into %d samples' % (len(sub_valid_data)))

        train_sampler, valid_sampler = None, None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                sub_train_data)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                sub_valid_data)

        train_queue = torch.utils.data.DataLoader(
            sub_train_data, batch_size=args.batch_size,
            sampler=train_sampler, shuffle=(train_sampler is None),
            pin_memory=True, num_workers=16, drop_last=True)

        valid_queue = torch.utils.data.DataLoader(
            sub_valid_data, batch_size=args.batch_size,
            sampler=valid_sampler, shuffle=(valid_sampler is None),
            pin_memory=True, num_workers=16, drop_last=False)


    elif mode == 'timm':
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=None)
            valid_data = imagenet_lmdb_dataset(
                traindir, transform=val_transform)
        else:
            train_data =  ImageDataset(traindir)
            valid_data = dset.ImageFolder(traindir, transform=val_transform)

        train_interpolation = 'bicubic'
        train_queue = create_loader(
            train_data,
            input_size=args.resize // downscale,
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=0.2,
            re_mode="pixel",
            re_count=1,
            re_split=False,
            scale=[0.08, 1.0],
            ratio=[0.75, 1.3333333333333333],
            hflip=0.5,
            vflip=0.0,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5",
            num_aug_splits=0,
            interpolation=train_interpolation,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=16,
            distributed=args.distributed,
            collate_fn=None,
            pin_memory=False,
            use_multi_epochs_loader=False
        )

        num_train = len(valid_data)
        print('Found %d samples' % (num_train))
        sub_num_train = int(np.floor(args.train_portion * num_train))
        sub_num_valid = num_train - sub_num_train

        _, sub_valid_data = my_random_split(
            valid_data, [sub_num_train, sub_num_valid], seed=0)

        print('Valid: Split into %d samples' % (len(sub_valid_data)))

        train_sampler, valid_sampler = None, None
        if args.distributed:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                sub_valid_data)

        valid_queue = torch.utils.data.DataLoader(
            sub_valid_data, batch_size=args.batch_size,
            shuffle=(valid_sampler is None),
            sampler=valid_sampler, pin_memory=True, num_workers=16, drop_last=False)

    elif mode == 'timm2':
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=None)
            valid_data = imagenet_lmdb_dataset(
                traindir, transform=val_transform)
        else:
            train_data =  ImageDataset(traindir)

        valid_data = ImageDataset(testdir)

        train_interpolation = "bicubic"
        train_queue = create_loader(
            train_data,
            input_size=args.resize // downscale,
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=0.2,
            re_mode="pixel",
            re_count=1,
            re_split=False,
            scale=[0.08, 1.0],
            ratio=[0.75, 1.3333333333333333],
            hflip=0.5,
            vflip=0.0,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5",
            num_aug_splits=0,
            # interpolation=train_interpolation,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=16,
            distributed=args.distributed,
            collate_fn=None,
            pin_memory=False,
            use_multi_epochs_loader=False
        )
        valid_queue = create_loader(
            valid_data,
            input_size=args.resize // downscale,
            batch_size=args.batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation=train_interpolation,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=16,
            distributed=args.distributed,
            crop_pct=0.875,
            color_jitter=0.4,
            pin_memory=False,
        )

    elif mode == 'timm3':
        # with test set from ImageNetV2 test split
        if 'lmdb' in args.data:
            train_data = imagenet_lmdb_dataset(
                traindir, transform=None)
            valid_data = imagenet_lmdb_dataset(
                traindir, transform=val_transform)
        else:
            train_data = ImageDataset(traindir)

        valid_data = ImageDataset(testdir)
        # valid_data = ImageDataset(traindir)

        train_interpolation = 'bicubic'
        train_queue = create_loader(
            train_data,
            input_size=args.resize // downscale,
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=0.2,
            re_mode="pixel",
            re_count=1,
            re_split=False,
            scale=[0.08, 1.0],
            ratio=[0.75, 1.3333333333333333],
            hflip=0.5,
            vflip=0.0,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5",
            num_aug_splits=0,
            interpolation=train_interpolation,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            # num_workers=16,
            num_workers=8,
            distributed=args.distributed,
            collate_fn=None,
            pin_memory=False,
            use_multi_epochs_loader=False
        )

        valid_queue = create_loader(
            valid_data,
            input_size=args.resize // downscale,
            batch_size=args.batch_size * 4,
            is_training=True,
            use_prefetcher=True,
            interpolation=train_interpolation,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            # num_workers=16,
            num_workers=8,
            distributed=args.distributed,
            crop_pct=0.875,
            color_jitter=0.0,
            pin_memory=False,
        )

    return train_queue, valid_queue, 1000

################################################################################


def my_random_split(dataset, lengths, seed=0):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!")
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(sum(lengths), generator=g)
    return [Subset_imagenet(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
################################################################################


def my_random_split_perc(dataset, percent_train, seed=0):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        percent_train (float): portion of the dataset to be used for training
    """

    num_train = len(dataset)
    print('Found %d samples' % (num_train))
    sub_num_train = int(np.floor(percent_train * num_train))
    sub_num_valid = num_train - sub_num_train
    dataset_train, dataset_validation = my_random_split(dataset, [sub_num_train, sub_num_valid], seed=seed)
    print('Train: Split into %d samples' % (len(dataset)))


    return [dataset_train, dataset_validation]


################################################################################
# ImageNet - LMDB
################################################################################

import io
import os
try:
    import lmdb
except:
    pass
import torch
from torchvision import datasets
from PIL import Image


def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def imagenet_lmdb_dataset(
        root, transform=None, target_transform=None,
        loader=lmdb_loader):
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')
    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        print('Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        print('Saving pt to {}'.format(pt_path))
        print('Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for path, class_index in data_set.imgs:
                with open(path, 'rb') as f:
                    data = f.read()
                txn.put(path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)
    return data_set

################################################################################
# Hyperspectral Utilities & Loaders
################################################################################

def pad_with_zeros(X, margin=2):
    """Pad hyperspectral cube spatially with zeros.

    Args:
        X: np.ndarray [H, W, C]
        margin: int, padding on each side
    Returns:
        np.ndarray [H+2m, W+2m, C]
    """
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]), dtype=X.dtype)
    newX[margin:margin + X.shape[0], margin:margin + X.shape[1], :] = X
    return newX


def create_image_cubes(X, y, window_size=15, remove_zero_labels=True):
    """Extract centered patches for every pixel with optional background removal.

    Args:
        X: np.ndarray [H, W, C]
        y: np.ndarray [H, W]
        window_size: int (odd)
        remove_zero_labels: bool
    Returns:
        patches: np.ndarray [N, window, window, C]
        labels: np.ndarray [N]
    """
    margin = int((window_size - 1) / 2)
    padded = pad_with_zeros(X, margin=margin)
    patches, labels = [], []
    H, W = X.shape[0], X.shape[1]
    for r in range(H):
        rr = r + margin
        for c in range(W):
            cc = c + margin
            label = y[r, c]
            if not remove_zero_labels or label > 0:
                patch = padded[rr - margin:rr + margin + 1, cc - margin:cc + margin + 1, :]
                patches.append(patch)
                labels.append(label)
    patches = np.asarray(patches, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if remove_zero_labels:
        labels = labels - 1
    return patches, labels


def apply_pca(X, num_components=15):
    """Apply PCA across spectral bands to reduce to num_components.

    Args:
        X: np.ndarray [H, W, C]
    Returns:
        np.ndarray [H, W, num_components]
    """
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        raise ImportError("scikit-learn is required for PCA. Install with: pip install scikit-learn") from e
    H, W, C = X.shape
    X2 = X.reshape(-1, C)
    pca = PCA(n_components=num_components, whiten=True)
    Xp = pca.fit_transform(X2)
    return Xp.reshape(H, W, num_components)


class HyperspectralDataset(torch.utils.data.Dataset):
    """Dataset returning hyperspectral patches as tensors [C, H, W]."""
    def __init__(self, patches, labels, per_sample_norm=True):
        self.patches = patches  # np.ndarray [N, H, W, C] or [N, C, H, W]
        self.labels = labels
        self.per_sample_norm = per_sample_norm

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        x = self.patches[idx]
        if x.ndim == 3:  # H, W, C
            x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).float()
        if self.per_sample_norm:
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True) + 1e-8
            x = (x - mean) * (1.0 / std)
        y = int(self.labels[idx])
        return x, y


def _discover_hsi_in_dir(data_root, preferred_dataset=None):
    """Attempt to discover a pair of data/label .mat files and keys in a folder."""
    candidates = [
        # (data_path, label_path, data_key, label_key)
        ("Indian_pines_corrected.mat", "Indian_pines_gt.mat", "indian_pines_corrected", "indian_pines_gt"),
        ("WHU_Hi_LongKou.mat", "WHU_Hi_LongKou_gt.mat", "WHU_Hi_LongKou", "WHU_Hi_LongKou_gt"),
        ("PaviaU.mat", "PaviaU_gt.mat", "paviaU", "paviaU_gt"),
        ("PaviaC.mat", "PaviaC_gt.mat", "paviaC", "paviaC_gt"),
        ("Salinas_corrected.mat", "Salinas_gt.mat", "salinas_corrected", "salinas_gt"),
        ("WHU_Hi_HongHu.mat", "WHU_Hi_HongHu_gt.mat", "WHU_Hi_HongHu", "WHU_Hi_HongHu_gt"),
        ("QUH-Qingyun.mat", "QUH-Qingyun_GT.mat", "Chengqu", "ChengquGT"),
        ("data.mat", "labels.mat", "X", "y"),
    ]
    
    # If preferred dataset is specified, prioritize it
    if preferred_dataset:
        preferred_candidates = [c for c in candidates if preferred_dataset.lower() in c[0].lower()]
        if preferred_candidates:
            candidates = preferred_candidates + [c for c in candidates if c not in preferred_candidates]
    
    for dname, lname, dkey, lkey in candidates:
        dp = os.path.join(data_root, dname)
        lp = os.path.join(data_root, lname)
        if os.path.isfile(dp) and os.path.isfile(lp):
            return dp, lp, dkey, lkey
    raise FileNotFoundError(f"No supported HSI .mat pair found in {data_root}")


def get_hsi_loaders(args):
    """Create train/val loaders for hyperspectral data from mambavision/Data.

    Uses PCA to 15 bands, patch extraction, per-sample normalization.
    """
    data_root = args.data_dir or getattr(args, 'data', None) or os.path.join(os.path.dirname(__file__), '..', 'Data')
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        # default to repo Data folder
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_root = os.path.join(repo_root, 'Data')
    preferred_dataset = getattr(args, 'hsi_dataset', None)
    dp, lp, dkey, lkey = _discover_hsi_in_dir(data_root, preferred_dataset)
    
    # Extract dataset name from file path
    dataset_name = os.path.basename(dp).replace('.mat', '')

    X = sio.loadmat(dp)[dkey]
    y = sio.loadmat(lp)[lkey]
    # Ensure 2D labels
    if y.ndim > 2:
        y = y.squeeze()

    # Optional PCA to args.hsi_bands or default 15
    target_bands = getattr(args, 'hsi_bands', 15)
    if X.shape[2] != target_bands:
        X = apply_pca(X, num_components=target_bands)

    window = getattr(args, 'hsi_patch', 15)
    patches, labels = create_image_cubes(X, y, window_size=window, remove_zero_labels=True)

    # Train/val split
    try:
        from sklearn.model_selection import train_test_split
    except Exception as e:
        raise ImportError("scikit-learn is required for HSI split. Install with: pip install scikit-learn") from e
    test_ratio = getattr(args, 'hsi_test_ratio', 0.90)
    Xtr, Xval, ytr, yval = train_test_split(patches, labels, test_size=test_ratio, random_state=42, stratify=labels)

    # To [N, C, H, W]
    Xtr = np.transpose(Xtr, (0, 3, 1, 2))
    Xval = np.transpose(Xval, (0, 3, 1, 2))

    train_set = HyperspectralDataset(Xtr, ytr, per_sample_norm=True)
    val_set = HyperspectralDataset(Xval, yval, per_sample_norm=True)

    train_queue = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=min(8, os.cpu_count() or 4),
        drop_last=True,
    )

    valid_queue = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=min(8, os.cpu_count() or 4),
        drop_last=False,
    )

    num_classes = int(labels.max() + 1)
    return train_queue, valid_queue, num_classes, dataset_name


if __name__ == '__main__':
    # Simple HSI smoke test using mambavision/Data
    import argparse
    parser = argparse.ArgumentParser('HSI loader smoke test')
    parser.add_argument('--data_dir', type=str, default='./mambavision/Data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hsi_bands', type=int, default=15)
    parser.add_argument('--hsi_patch', type=int, default=15)
    parser.add_argument('--hsi_test_ratio', type=float, default=0.9)
    args = parser.parse_args()

    # Attach dataset flag expected by get_loaders
    args.dataset = 'hsi'

    train_loader, val_loader, num_classes = get_hsi_loaders(args)
    print('HSI smoke test:')
    print(' - classes:', num_classes)
    print(' - train batches:', len(train_loader))
    print(' - val batches:', len(val_loader))
    xb, yb = next(iter(train_loader))
    print(' - batch x shape:', tuple(xb.shape))  # [B, C, H, W]
    print(' - batch y shape:', tuple(yb.shape))
