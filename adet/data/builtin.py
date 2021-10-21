import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.register_coco import register_coco_instances

from .datasets.text import register_text_instances
from .datasets.youtubevis import register_youtubevis_video

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    'pic_person_train':
    ('pic/image/train', 'pic/annotations/train_person.json'),
    'pic_person_val': ('pic/image/val', 'pic/annotations/val_person.json'),
}

metadata_pic = {'thing_classes': ['person']}

_PREDEFINED_SPLITS_TEXT = {
    'totaltext_train': ('totaltext/train_images', 'totaltext/train.json'),
    'totaltext_val': ('totaltext/test_images', 'totaltext/test.json'),
    'ctw1500_word_train':
    ('CTW1500/ctwtrain_text_image',
     'CTW1500/annotations/train_ctw1500_maxlen100_v2.json'),
    'ctw1500_word_test': ('CTW1500/ctwtest_text_image',
                          'CTW1500/annotations/test_ctw1500_maxlen100.json'),
    'syntext1_train': ('syntext1/images', 'syntext1/annotations/train.json'),
    'syntext2_train': ('syntext2/images', 'syntext2/annotations/train.json'),
    'mltbezier_word_train':
    ('mlt2017/images', 'mlt2017/annotations/train.json'),
    'rects_train':
    ('ReCTS/ReCTS_train_images', 'ReCTS/annotations/rects_train.json'),
    'rects_val':
    ('ReCTS/ReCTS_val_images', 'ReCTS/annotations/rects_val.json'),
    'rects_test': ('ReCTS/ReCTS_test_images',
                   'ReCTS/annotations/rects_test.json'),
    'art_train': ('ArT/rename_artimg_train',
                  'ArT/annotations/abcnet_art_train.json'),
    'lsvt_train': ('LSVT/rename_lsvtimg_train',
                   'LSVT/annotations/abcnet_lsvt_train.json'),
    'chnsyn_train': ('ChnSyn/syn_130k_images',
                     'ChnSyn/annotations/chn_syntext.json'),
}

metadata_text = {'thing_classes': ['text']}

_PERDEFINED_SPLITS_YOUTUBEVIS_VIDEO = {
    'youtubevis_train':
    ('youtubevis/train/', 'youtubevis/annotations/train.json'),
    'youtubevis_valid':
    ('youtubevis/valid/', 'youtubevis/annotations/valid.json'),
    'youtubevis_test':
    ('youtubevis/test/', 'youtubevis/annotations/test.json'),
}

metadata_youtubevis_video = {
    'thing_classes': [
        'person', 'giant_panda', 'lizard', 'parrot', 'skateboard', 'sedan',
        'ape', 'dog', 'snake', 'monkey', 'hand', 'rabbit', 'duck', 'cat',
        'cow', 'fish', 'train', 'horse', 'turtle', 'bear', 'motorbike',
        'giraffe', 'leopard', 'fox', 'deer', 'owl', 'surfboard', 'airplane',
        'truck', 'zebra', 'tiger', 'elephant', 'snowboard', 'boat', 'shark',
        'mouse', 'frog', 'eagle', 'earless_seal', 'tennis_racket'
    ]
}


def register_all_coco(root='datasets'):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file)
            if '://' not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file)
            if '://' not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_youtubevis(root='datasets'):
    for key, (image_root,
              json_file) in _PERDEFINED_SPLITS_YOUTUBEVIS_VIDEO.items():
        register_youtubevis_video(
            key,
            metadata_youtubevis_video,
            os.path.join(root, json_file)
            if '://' not in json_file else json_file,
            os.path.join(root, image_root),
        )


register_all_coco()
register_all_youtubevis()
