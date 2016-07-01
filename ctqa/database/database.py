# -*- coding: utf-8 -*-

import numpy as np
import tables as tb
import os
import logging

from .import_dicom import find_ct_series
logger = logging.getLogger('CTQA')


handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

METADATA_TEMPLATE = {
# variable: [numpy_type, description, dicom_tag_if_any, required]
    'scanner': [np.dtype('a128'), 'Scanner model name', (0x8, 0x1090), False],
    'software': [np.dtype('a128'), 'Software version', (0x18, 0x1020), False],
    'station': [np.dtype('a128'), 'station name', (0x8, 0x1010), False],
    'acu_date': [np.dtype('a8'), 'Acqusition date', (0x8, 0x22), False],
    'acu_time': [np.dtype('a6'), 'Acqusition time', (0x8, 0x32), False],
    'series_uid': [np.dtype('a64'), 'Series UID', (0x20, 0xE), True],
    'scan_fov': [np.dtype(np.double), 'Scan field of view [mm]', (0x18, 0x1100), False],
    'rec_fov': [np.dtype(np.double), 'Reconstructed field of view [mm]', (0x18, 0x90), False],
    'filter': [np.dtype('a128'), 'Filter type', (0x18, 0x1160), False],
    'kernel': [np.dtype('a128'), 'Reconstruction kernel',(0x18, 0x1210), False],
    'focus': [np.dtype(np.double), 'Focus spots', (0x18, 0x1190), False],
    'reconstruction_center': [np.dtype((np.double, 3)), 'Reconstruction center [mm]', (0x18, 0x9318), False],
    'data_center': [np.dtype((np.double, 3)), 'Data collection center [mm]', (0x18, 0x9313), False],
    'single_width': [np.dtype(np.double), 'Single collimation width [mm]', (0x18, 0x9306), False],
    'total_width': [np.dtype(np.double), 'Total collimation width [mm]', (0x18, 0x9307), False],
    'series_date': [np.dtype('a8'), 'Series date', (0x8, 0x20), False],
    'series_time': [np.dtype('a6'), 'Series time', (0x8, 0x31), False],
    'reconKernel': [np.dtype('a64'), 'Reconstruction kernel', (0x18, 0x1210), False],
    'kV': [np.dtype(np.double), 'Simulation tube potential [kV]', (0x18, 0x60), False],
    'is_spiral': [np.dtype(np.bool), 'Helical aqusition', None, False],
    'pitch': [np.dtype(np.double), 'Pitch', (0x18, 0x9311), False],
    'spacing': [np.dtype((np.double, 3)), 'Image matrix spacing [mm]', None, True],
    'slice_thickness': [np.dtype(np.double), 'Slice thickness [mm]', (0x18, 0x50), False],
    'shape': [np.dtype((np.int, 3)), 'Image matrix dimensions', None, True],
    'image_orientation': [np.dtype((np.double, 6)), 'Image patient orientation cosines', (0x20, 0x37), True],
    'image_position': [np.dtype((np.double, 3)), 'Image position (position of first voxel in volume) [mm]', (0x20, 0x32), True],
    'patient_position': [np.dtype('a3'), 'Patient position', (0x18, 0x5100), True],
    'for_uid': [np.dtype('a64'), 'Frame of reference UID', (0x20, 0x52), True],
    'study_uid': [np.dtype('a64'), 'Study UID', (0x20, 0xd), True],
    'rows': [np.dtype(np.int), 'Rows', (0x28, 0x10), True],
    'columns': [np.dtype(np.int), 'Columns', (0x28, 0x11), True],
    'is_localized': [np.dtype(np.bool), 'Series is localized', None, False],
    'is_analyzed': [np.dtype(np.bool), 'Series is analyzed', None, False],
    'phantom_offset': [np.dtype(np.double), 'Phantom offset', None, False],
    'cp404_rods_position': [np.dtype((np.double, 4 * 3)), 'Phantom offset', None, False],
    'phantom': [np.dtype('a128'), 'Phantom model', None, False],
    }


ARRAY_TEMPLATE = {
    'ctarray': [np.int16],
    'exposure': [np.float32],
    'image_positions':[np.float32],
    'image_uids': [np.dtype('a64')]
    }


def metadata_table_dtype():
    d = {'names': [],
         'formats': []}
    for key, value in METADATA_TEMPLATE.items():
        d['names'].append(key)
        d['formats'].append(value[0])
        d['names'].append(key + '_isset')
        d['formats'].append(np.dtype(np.bool))
    return np.dtype(d)


def sort_dicom_series(dclist):
    im_or = dclist[0][0x20, 0x37].value
    nx = np.array(im_or[:3])
    ny = np.array(im_or[3:])
    nz = np.cross(nx, ny)
    s_ind = np.argmax(nz)
    dclist.sort(key = lambda x: x[0x20, 0x32].value[s_ind])

class Analysis(object):
    def __init__(self, name, metadata, *extra_data_list, dtype=None):
        self.name = name


        self.arrays = {'circle_rois': None, # np.zeros(n_elements, dtype=[('x', np.float),('y', np.float),('z', np.float),('radius', np.float),('label', 'S32')])
                       'box_rois': None, # np.zeros(n_elements, dtype=[('x', np.float),('y', np.float),('z', np.float), ('height', np.float),('width', np.float),('label', 'S32')])
                       'table': None, #np.zeros((rows, columns), dtype='S32')
                       'plots': [], # [np.zeros(n_elements, dtype=dtype=[('x', np.float),('y', np.float),('label', 'S32'),[('xstd', np.float),('ystd', np.float)] ]), ...]
                       'plot_info': None, # np.array([xaxislabel, yaxixlabel, title], dtype='S32')
                       'referenced_images': None, # np.array([image_uids...], dtype='S64')
                       }
        self.data = {}
        self.dtype = {'names': [], 'formats': []}
        if dtype is None:
            for key in ['series_uid', 'acu_date', 'scanner', 'station', 'slice_thickness', 'kernel', 'kV']:
                if key in metadata:
                    self.data[key] = metadata[key]
                self.dtype['names'].append(key)
                self.dtype['formats'].append(METADATA_TEMPLATE[key][0])
        else:
            for key, value in metadata.items():
                self.data[key] = value
                self.dtype['names'].append(key)
                self.dtype['formats'].append(dtype[key])


        for data_name, value, datatype in extra_data_list:
            self.add_data(data_name, value, datatype)



    def add_data(self, data_name, value, dtype):
        self.data[data_name] = value
        self.dtype['formats'].append(dtype)
        self.dtype['names'].append(data_name)

    def tables(self):
        for name, array in self.arrays.items():
            if (array is not None) and (name is not 'plots'):
                yield name, array
    def plots(self):
        for ind, plot in enumerate(self.arrays['plots']):
            yield str(ind+1), plot

    @property
    def series_uid(self):
        return self.data['series_uid']

class Database(object):
    def __init__(self, path):
        self.db_path = os.path.abspath(path)
        self.db_instance = None
        self.filters = tb.Filters(complevel=9, complib='blosc', fletcher32=True, shuffle=True)
        self.init_database()

    def init_database(self):
        logger.debug('Initializing database {}'.format(self.db_path))
        self.open()
        self._get_node('/', 'metadata', create=True, obj=metadata_table_dtype())
        self.close()

    def open(self):
        if self.db_instance is not None:
            if self.db_instance.isopen:
                return
        self.db_instance = tb.open_file(self.db_path, mode='a',
                                        filters=self.filters)

    def close(self):
        if self.db_instance is not None:
            if self.db_instance.isopen:
                self.db_instance.close()
        self.db_instance = None

    def _get_node(self, where, name, create=False, obj=None, overwrite=False, earray=False, shape=None, atom=None):
        self.open()
        try:
            node = self.db_instance.get_node(where, name=name)
        except tb.NoSuchNodeError:
            if not create:
                raise ValueError("Node {0} do not exist in {1}. Was not allowed to create a new node".format(name, where))

            if obj is None and not earray:
                node = self.db_instance.create_group(where, name,
                                                     createparents=True)
            elif isinstance(obj, np.recarray) or isinstance(obj, np.dtype):
                node = self.db_instance.create_table(where, name,
                                                     description=obj,
                                                     createparents=True)
            elif isinstance(obj, np.ndarray):
                if obj.dtype.names is not None:
                    node = self.db_instance.create_table(where, name,
                                                         description=obj,
                                                         createparents=True)
                else:

                    node = self.db_instance.create_carray(where, name,
                                                              obj=obj,
                                                              createparents=True,
                                                              filters=self.filters)
            elif obj is None and earray:

                if shape is None:
                    raise ValueError('When creating earrays, the shape parameter must be set.')
                if atom is None and obj is None:
                    raise ValueError('When creating earrays, the atom parameter must be set if not obj is provided.')
                node = self.db_instance.create_earray(where, name,
                                                      obj=obj,
                                                      createparents=True,
                                                      filters=self.filters,
                                                      shape=shape,
                                                      atom=atom)

            else:
                raise ValueError("Node {0} do not exist in {1}. Unable to create new node, did not understand obj type".format(name, where))
            logger.debug('Created node {}/{}'.format(where, name))
        else:
            if overwrite and create:
                self.db_instance.remove_node(where, name)
                logger.debug('Overwriting node {}/{}'.format(where, name))
                return self._get_node(where, name, create, obj, overwrite)

        return node

    def _remove_node(self, where, name):
        self.open()
        if name == '':
            return
        try:
            self.db_instance.remove_node(where, name=name, recursive=True)
        except tb.NoSuchNodeError:
            logger.debug('No node in {0}/{1} to delete'.format(where, name))
        else:
            logger.debug('Deleted node {0}/{1}'.format(where, name))
        return



    def add_series_from_dicom(self, dclist, overwrite=False, phantom=''):
        logger.debug('Parsing dicom series')
        if len(dclist) == 0:
            raise ValueError('Input must be a sequence with lenght > 0')

        #sorting dicom series on position
        sort_dicom_series(dclist)

        dc0 = dclist[0]
        #dict of meta information on series
        meta = {}
        for key, value in METADATA_TEMPLATE.items():
            tag = value[2]
            if tag is not None:
                try:
                    meta[key] = dc0[tag].value
                except KeyError:
                    # if not all required tags in metainformation we retunr
                    meta[key] = None
                    if value[3]:
                        logger.debug("Insufficient information, series not imported")
#                        import pdb;pdb.set_trace()
                        return

        meta['shape'] = np.array([len(dclist), meta['columns'], meta['rows']], dtype=np.int)
        if len(dclist) > 1:
            zspacing = np.sum((np.array(dclist[1][0x20, 0x32].value) - np.array(dclist[0][0x20, 0x32].value))**2)**.5
        elif 'slice_thickness' in meta:
            zspacing = meta['slice_thickness']
        else:
            zspacing = 0
        meta['spacing'] = np.array([zspacing,] + list(dc0[0x28, 0x30].value))

        if meta['pitch'] is None:
            meta['is_spiral'] = False
        else:
            meta['is_spiral'] = True

        meta['is_localized'] = False
        meta['is_analyzed'] = False
        meta['phantom'] = str(phantom)

        self.open()

        meta_table = self._get_node('/', 'metadata')

        #testing if series exists
        indices = meta_table.get_where_list('series_uid == b"{}"'.format(meta['series_uid']))
        if len(indices) > 0:
            if overwrite:
                for ind in indices:
                    self._remove_node('/ctarrays/' + meta['study_uid'], meta['series_uid'])
                    try:
                        meta_table.remove_row(ind)
                    except NotImplementedError: #fix for deleting last row in table
                        self._remove_node('/', 'metadata')
                        meta_table = self._get_node('/', 'metadata', create=True, obj=metadata_table_dtype())
                        break

                meta_table.flush()
            else:
                self.close()
                logger.debug('Series {} is already in database'.format(meta['series_uid']))
                return

        logger.debug('Adding series {} to database'.format(meta['series_uid']))
        ## adding metadata
        meta_table_row = meta_table.row
        for key, value in meta.items():
            if value is not None:
                meta_table_row[key] = value
                meta_table_row[key + '_isset'] = True
        meta_table_row.append()
        meta_table.flush()


        #adding arrays
        where = '/' + '/'.join(['ctarrays', meta['study_uid'], meta['series_uid']])
        ctarray_node = self._get_node(where, 'ctarray', create=True, overwrite=overwrite, earray=True, shape=(0, meta['columns'], meta['rows']), atom=tb.Atom.from_dtype(np.dtype(np.int16)))
        exposure = np.zeros(len(dclist))
        image_positions = np.empty((len(dclist), 3))
        image_uids = np.zeros(len(dclist), dtype=np.dtype('a64'))
        for i, dc in enumerate(dclist):
            arr = dc.pixel_array * dc[0x28, 0x1053].value + dc[0x28, 0x1052].value
            ctarray_node.append(arr.reshape((1,) + arr.shape).astype(np.int16))
            try:
                exposure[i] = dc[0x18, 0x1150].value * dc[0x18, 0x1151].value / 1000
            except KeyError:
                import pdb; pdb.set_trace()
#                pass
            image_positions[i, :] = dc[0x20, 0x32].value
            image_uids[i] = dc[0x8, 0x18].value

        self._get_node(where, 'exposure', create=True, obj=exposure)
        self._get_node(where, 'image_positions', create=True, obj=image_positions)
        self._get_node(where, 'image_uids', create=True, obj=image_uids)
        self.close()
        logger.debug('Done')
        return meta

    def update_series_metadata(self, series_uid, fields):
        self.open()
        meta_table = self._get_node('/', 'metadata')
        condition = 'series_uid == b"{}"'.format(series_uid)
        for row in meta_table.where(condition):
            for key, value in fields.items():
                row[key] = value
                row[key + '_isset'] = True
            row.update()
            logger.debug('Updated metadata for series {}'.format(series_uid))
        meta_table.flush()
        self.close()
        return

    def get_metadata(self, column, condition):
        """
        find a column in metadatatable based on condition, condition is a search string, i.e:
        condition = 'acu_date == 26102016 & study_uid == b"1.0.456742.1344..."'
        Note the string comparison handling!
        Returns a list of rows in column corresponding to condition
        """
        self.open()
        meta_table = self._get_node('/', 'metadata')
        if len(condition) > 0:
            data = [row[column] for row in meta_table.where(condition)]
        else:
            data = [row[column] for row in meta_table]

        if meta_table.coldtypes[column].kind == 'S':
            data = [d.decode('utf-8') for d in data]
        self.close()
        return list(data)

    def get_series(self, condition=''):
        """
        find series based on condition, condition is a search string, i.e:
        condition = 'acu_date == 26102016 & study_uid == b"1.0.456742.1344..."'
        Note the string comparison handling!
        Returns a list of series uids corresponding to condition
        """
        self.open()
        meta_table = self._get_node('/', 'metadata')
        if len(condition) > 0:
            data = list([row['series_uid'].decode('utf-8') for row in meta_table.where(condition)])
        else:
            data = list([row['series_uid'].decode('utf-8') for row in meta_table])
        self.close()
        return data

    def get_series_metadata(self, series_uid):
        """
        returns a dict of metadata corresponding to the requested series
        """
        self.open()
        meta_table = self._get_node('/', 'metadata')
        condition = 'series_uid == b"{}"'.format(series_uid)
        for row in meta_table.where(condition):
            metadata = {}
            for key, value in METADATA_TEMPLATE.items():
                if row[key + '_isset']:
                    if meta_table.coldtypes[key].kind == 'S':
                        metadata[key] = row[key].decode('utf-8')
                    else:
                        metadata[key] = row[key]
            self.close()
            return metadata
        else:
            self.close()
            raise ValueError('No series with series UID {} in database'.format(series_uid))

    def get_series_array(self, study_uid, series_uid, *args, start=None, stop=None, step=None):
        """
        get numpy array for series series_uid, use start, stop, step to select a range in z direction
        """
        self.open()
        if study_uid == '':
            meta_table = self._get_node('/', 'metadata')
            for row in meta_table.where('series_uid == b"{}"'.format(series_uid)):
                study_uid = row['study_uid'].decode('utf-8')
                break
            else:
                self.close()
                raise ValueError('Could not find series with series UID {}'.format(series_uid))

        if len(args) == 0:
            raise ValueError('Must supply an array name to read from database')
        elif len(args) == 1:
            arrays = self._get_node('/ctarrays/{}/{}'.format(study_uid, series_uid), args[0]).read(start, stop, step)
        else:
            arrays = tuple((self._get_node('/ctarrays/{}/{}'.format(study_uid, series_uid), array_name).read(start, stop, step) for array_name in args))
#        array_node = self._get_node('/ctarrays/{}/{}'.format(study_uid, series_uid), array_name)
#        array = array_node.read(start, stop, step)
        self.close()
        return arrays

    def add_analysis(self, analysis):
        if not analysis.success:
            logger.debug('Analysis not valid')
            return
        self.open()
        ana_table = self._get_node('/analysis/{}'.format(analysis.name), 'data', create=True, obj=np.dtype(analysis.dtype))


        #testing if series exists
        indices = ana_table.get_where_list('series_uid == b"{}"'.format(analysis.series_uid))
        if len(indices) > 0:
            for ind in indices:
                self._remove_node('/analysis/{}'.format(analysis.name), analysis.series_uid)
                try:
                    ana_table.remove_row(ind)
                except NotImplementedError: #fix for deleting last row in table
                    self._remove_node('/analysis/{}'.format(analysis.name), 'data')
                    ana_table = self._get_node('/analysis/{}'.format(analysis.name), 'data', create=True, obj=np.dtype(analysis.dtype))
                    break

            ana_table.flush()

        row = ana_table.row
        for key, value in analysis.data.items():
            row[key] = value
        row.append()
        ana_table.flush()

        for name, array in analysis.tables():
            self._get_node('/analysis/{}/{}'.format(analysis.name, analysis.series_uid), name, create=True, obj=array)
        for name, array in analysis.plots():
            self._get_node('/analysis/{}/{}/plots'.format(analysis.name, analysis.series_uid), name, create=True, obj=array)
        logger.debug('Analysis saved')
        self.close()


    def get_analysis(self, name, series_uid):
        self.open()
        ana_table = self._get_node('/analysis/{}'.format(name), 'data')

        # settingdata
        data = {}
        data_dtype = {}
        for row in ana_table.where('series_uid == b"{}"'.format(series_uid)):
            for col_name, col_dtype in zip(ana_table.colnames, ana_table.coldtypes):
                if col_dtype is str:
                    data[col_name] = row[col_name].decode('utf-8')
                else:
                    data[col_name] = row[col_name]

                data_dtype[col_name] = col_dtype
            break
        else:
            self.close()
            raise ValueError('No analysis for {} found'.format(series_uid))
        ana = Analysis(name, data, dtype=data_dtype)


        #setting tables
        ana_node = self._get_node('/analysis/{}'.format(name), series_uid)
        for node in ana_node:
            if node._v_name != 'plots':
                ana.arrays[node._v_name] = node.read()
            else:
                for plot in ana_node.plots:
                    ana.arrays['plots'].append(plot.read())
        self.close()
        return ana

    def get_analysis_data(self, name, condition=None, field=None):
        self.open()
        try:
            ana_table = self._get_node('/analysis/{}'.format(name), 'data')
        except ValueError:
            self.close()
            raise ValueError('No analysis named {}'.format(name))
        if condition is None:
            data = ana_table.read(field=field)
        else:
            data = ana_table.read_where(condition, field=field)
        self.close()
        return data



    def list_analysis_types(self):
        self.open()
        analysis_node = self._get_node('/', 'analysis', create=True)
        analysis_types = list([n._v_name for n in analysis_node])
        self.close()
        return analysis_types


if __name__ == '__main__':
    db = Database('C://test//dbctqa.h5')
#    for images in find_ct_series(['C://test//DICOM']):
#    for images in find_ct_series(['C://test//ctqatest']):
#        db.add_series_from_dicom(images, overwrite=True)
#    print(db.studies())

    s_uids = db.get_series()
    meta = db.get_series_metadata(s_uids[0])
#    c_node = db._get_node('/ctarrays/{}/{}'.format(meta['study_uid'], meta['series_uid']), 'ctarray')
    arr, uids = db.get_series_array('', s_uids[0], 'ctarray', 'image_uids')
    print(arr.shape)
    import pdb;pdb.set_trace()

    s_uids = db.get_series()
    import random
    print(db.get_series_metadata(s_uids[int(random.uniform(0, len(s_uids)))]))



