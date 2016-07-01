# -*- coding: utf-8 -*-


from ctqa.database import Database, find_ct_series
from ctqa.localizer import is_image_in_CP404, localize
from ctqa.analyse import analyse_all
import numpy as np
from matplotlib import pylab as plt




#    s_uids = db.get_series()
#    x=[]
#    y=[]
#    for serie in s_uids:
#        meta = db.get_series_metadata(serie)
#        shape = meta['shape']
#        pixel_spacing = meta['spacing']
#        i_pos = db.get_series_array('', serie, 'image_positions')
#        for i in range(shape[0]-1):
#
##            import pdb;pdb.set_trace()
#            arr = db.get_series_array('', serie, 'ctarray', start=i)
#            suc, corners, pos = is_image_in_CP404(arr, pixel_spacing)
#            if suc:
#                x.append(i_pos[i, 2])
#                y.append(pos)
#        plt.plot(x, y, 'o')
#        #linreg
#        reg = np.linalg.lstsq(np.vstack([np.array(x), np.ones(len(x))]).T, y)[0]
#        print(reg[1])
#        plt.plot(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100)*reg[0]+reg[1], '-b' )
#    plt.show()



if __name__ == '__main__':
    db = Database('C://test//dbctqa.h5')
#    for images in find_ct_series(['C://test//DICOM']):
#    for images in find_ct_series(['C://test//ctqatest']):
#    for images in find_ct_series(['C://test//cp404']):
#        db.add_series_from_dicom(images, overwrite=True)

    localize(db)
    analyse_all(db)
    l = db.list_analysis_types()
    s = db.get_analysis_data(l[0])['series_uid'][0].decode('utf-8')
    ana = db.get_analysis(l[0], s)
    import pdb;pdb.set_trace()



#    s_uids = db.get_series()
#    meta = db.get_series_metadata(s_uids[0])
#    arr, uids = db.get_series_array('', s_uids[0], 'ctarray', 'image_uids')
#    import pdb;pdb.set_trace()
