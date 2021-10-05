import os
import codecs
import zipfile
import os.path as osp

from ..bbox import rbox2poly_single


def result2IC15(results, dst_path, dataset):
    CLASSES = dataset.CLASSES
    img_names = dataset.img_names
    assert len(results) == len(
        img_names), 'length of results must equal with length of img_names'
    if not osp.exists(dst_path):
        os.mkdir(dst_path)
    for idx, img_name in enumerate(img_names):
        out_file = osp.join(dst_path, 'res_' + img_name.split('.')[0] + '.txt')
        bboxes = results[idx][0]
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if(bboxes.size != 0):
                for bbox in bboxes:
                        score = bbox[5]
                        bbox = rbox2poly_single(bbox[:5])
                        if score > 0.5 :
                            f.write('{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}\n'.format(
                                bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6], bbox[7])
                            )
    
    return True
