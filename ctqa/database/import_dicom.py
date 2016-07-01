# -*- coding: utf-8 -*-

import numpy as np
import dicom
import os
import logging
#from ctqa.database.database import Series

logger = logging.getLogger('CTQA')


#handler = logging.StreamHandler()
#formatter = logging.Formatter(
#        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#handler.setFormatter(formatter)
#logger.addHandler(handler)
#logger.setLevel(logging.DEBUG)


VALID_SOP_CLASS = "CT Image Storage"
VALID_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'AXIAL']

def find_all_files(pathList):
    for p in pathList:
        path = os.path.abspath(p)
        if os.path.isdir(path):
            for dirname, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    yield os.path.normpath(os.path.join(dirname, filename))
        elif os.path.isfile(path):
            yield os.path.normpath(path)

def find_ct_series(pathlist):
    series = {}
    image_uids = []
    for fil in find_all_files(pathlist):
        try:
            dc = dicom.read_file(fil, stop_before_pixels=True)
        except Exception as e:
            logger.info('Error when reading file {}: {}'.format(fil, e))
        else:
            if (0x8, 0x16) in dc and (0x8, 0x8) in dc:
                if str(dc[0x8, 0x16].value) == VALID_SOP_CLASS:
                    if all([str(dc[0x8, 0x8].value[i]) == val for i, val in enumerate(VALID_IMAGE_TYPE)]):
                        series_uid = str(dc[0x20, 0xe].value)
                        image_uid = str(dc[0x8, 0x18].value)
                        if image_uid in image_uids:
                            continue
                        else:
                            image_uids.append(image_uid)
                        if series_uid in series:
                            series[series_uid].append(fil)
                        else:
                            series[series_uid] = [fil,]
                        logger.debug('Imported file {}'.format(fil))
                    else:
                        logger.debug('CT image {} is not reconstructed from raw data'.format(fil))
                else:
                    logger.debug('Image {} not a valid CT DICOM image'.format(fil))
            else:
                logger.debug('Image {} not a valid CT DICOM image'.format(fil))
                
    for value in series.values():
        yield [dicom.read_file(fil) for fil in value]
        


if __name__ == '__main__':
    for t in find_ct_series(['C://test//ctqatest']):
        print(len(t))
        
    