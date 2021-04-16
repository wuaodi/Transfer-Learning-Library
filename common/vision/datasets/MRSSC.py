from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class MRSSC(ImageList):
    """MRSSC Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'V'``: VNI, \
            ``'I'``: IALT and ``'T'``: IALT_test.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            VNI/
                images/
                    city/
                        *.jpg
                        ...
            IALT/
            IALT_test/
            image_list/
                VNI.txt
                IALT.txt
                IALT_test.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d9bca681c71249f19da2/?dl=1"),
        ("amazon", "amazon.tgz", "https://cloud.tsinghua.edu.cn/f/edc8d1bba1c740dc821c/?dl=1"),
        ("dslr", "dslr.tgz", "https://cloud.tsinghua.edu.cn/f/ca6df562b7e64850ad7f/?dl=1"),
        ("webcam", "webcam.tgz", "https://cloud.tsinghua.edu.cn/f/82b24ed2e08f4a3c8888/?dl=1"),
    ]
    image_list = {
        "V": "image_list/VNI.txt",
        "I": "image_list/IALT.txt",
        "T": "image_list/IALT_test.txt"
    }
    CLASSES = ['city', 'coast', 'desert', 'farmland', 'lake', 'mountain', 'river']

    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        # if download:
        #     list(map(lambda args: download_data(root, *args), self.download_list))
        # else:
        #     list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(MRSSC, self).__init__(root, MRSSC.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())