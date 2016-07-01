# -*- coding: utf-8 -*-

import numpy as np
from ctqa.localizer import image_world_transform
from ctqa.utils import circle_indices
from ctqa.database import Analysis
from .cp404attenuationcoef import ATTCOEF


def cp404localizer(metadata):
    rods_position = metadata['cp404_rods_position'].reshape((4, 3))
    image_orientation = metadata['image_orientation']
    image_position = metadata['image_position']
    spacing = metadata['spacing']
    # rods coordinates in image world
    rods = image_world_transform(image_orientation, image_position, spacing,
                                 rods_position, inverse=True)
    #geometric center of all four rods
    center = rods.mean(axis=0)
    # mean rotation of rods about center
    rotation_angle = np.mean(np.arctan2(rods[:, 0]-center[0],
                                        rods[:, 1]-center[1]) -
                             np.deg2rad([-135, -45, 45, 135])
                             )
    # geometric mean distance between neighboring rods
    rod_spacing = np.mean(np.sum((rods - np.roll(rods, 3))**2, axis=1)**.5)
    return (np.rint(center).astype(np.int), np.rint(rods).astype(np.int),
            rotation_angle, rod_spacing)


def linarity_analysis_effective_energy(measured, material_names):
    hum = np.vstack((np.array(measured), np.ones(len(measured)))).T
    atts_index = np.array(ATTCOEF['kev']) <= 150.
    atts_N = atts_index.sum()
    atts = np.empty((hum.shape[0], atts_N))
    u_water = np.array(ATTCOEF['Water'])[atts_index]
    u_air = np.array(ATTCOEF['Air'])[atts_index]
    for i, name in enumerate(material_names):
        atts[i, :] = (1000. / (u_water - u_air)) * (np.array(ATTCOEF[name])[atts_index] - u_water)
    # linear least square solver for
    resid = np.linalg.lstsq(hum, atts)[1]
    r2 = 1. - resid / (atts.shape[0] * atts.var(axis=0))
    r_max_index = np.argmax(r2)

    nom = atts[:,r_max_index]

#    import pdb;pdb.set_trace()
    return ATTCOEF['kev'][r_max_index], nom, r2[r_max_index]

def linarity_analysis(ct_array, ct_pos, ct_uids, metadata):
    spacing = metadata['spacing']
    offset = metadata['phantom_offset']

    center, rods, rotation_angle, rod_spacing = cp404localizer(metadata)

    angles = np.deg2rad(np.array([0., 60., 90., 120., 180., 240., 270., 300.])) + rotation_angle

    center_array_index = np.argmin(np.abs(ct_pos[:, 2] + offset))
    array = np.squeeze(ct_array[center_array_index, :, :])

    roi_radius = int(round(4. / spacing[1]))
    roi_spacing = rod_spacing * 117. / 50. / 2.

    analysis = Analysis('linarity', metadata)

    HUmeasurement = []
    for angle in angles:
        pos = center[:2] + roi_spacing * np.array([np.sin(angle), np.cos(angle)])
        mask = circle_indices(array.shape, roi_radius, pos)
        if not mask.any():
            return analysis
        HUmeasurement.append((np.mean(array[mask]), np.std(array[mask]), pos))

    #testing if water vial is filled in phantom
    materials = ['Air', 'PMP', 'LDPE', 'Polystyrene', 'Water', 'Acrylic', 'Delrin', 'Teflon']
    water_index = np.argmin(np.abs(angles - np.deg2rad(270)))
    if HUmeasurement[water_index][0] < -300:
        del HUmeasurement[water_index]
        del materials[4]
        analysis.add_data('Water', np.nan, np.float)

    #sorting materials
    HUmeasurement.sort(key=lambda x: x[0])


    analysis.arrays['circle_rois'] = np.zeros(len(materials), dtype=[('x', np.float),('y', np.float),('z', np.float),('radius', np.float),('label', 'S32')])
    analysis.arrays['table'] = np.zeros((5, len(materials)+1), dtype=np.dtype('S32'))
    analysis.arrays['table'][:4, 0] = ['Material', 'Measured [HU]', 'St.dev [HU]', 'Nominal [HU]']
    meas_values = []
    for ind, material, value in zip(range(len(materials)), materials, HUmeasurement):
        analysis.arrays['circle_rois'][ind] = (value[2][0],
                                               value[2][1],
                                               center[2],
                                               roi_radius,
                                               material,
                                               )
        analysis.arrays['table'][0, ind+1] = material
        analysis.arrays['table'][1, ind+1] = str(round(value[0]))
        analysis.arrays['table'][2, ind+1] = str(round(value[1], 1))
        analysis.add_data(material, value[0], np.float)
        meas_values.append(value[0])
    # Getting effective energy
    effective_energy, nom_values, rkvad = linarity_analysis_effective_energy(meas_values, materials)
    analysis.arrays['table'][3, 1:] = [str(round(n)) for n in nom_values]

    analysis.arrays['table'][4,:4] = ['Effective energy [kev]:',
                                       str(round(effective_energy)),
                                       'r squared:',
                                       str(round(rkvad, 4))]

    analysis.arrays['plot_info'] = np.array(['Nominal ({}kev) [HU]'.format(effective_energy),
                                             'Measured [HU]',
                                             'CT numbers'], dtype=np.dtype('S32'))
    analysis.arrays['plots'].append(np.array([(x, y, l) for x, y, l in zip(nom_values, meas_values, materials)],
                                             dtype=[('x', np.float),('y', np.float),('label', 'S32')]))

    analysis.success = True
    analysis.arrays['referenced_images'] = np.array([ct_uids[center_array_index],], dtype='S64')
    analysis.add_data('Rsqr', rkvad, np.dtype(np.float))
#    from matplotlib import pylab as plt
#    array[np.rint(analysis.arrays['circle_rois']['x']).astype(np.int), np.rint(analysis.arrays['circle_rois']['y']).astype(np.int)]= array.max()*1.2
#    plt.imshow(array)
#    plt.show()
    return analysis

def slice_thickness_analysis(ct_array, ct_pos, ct_uids, metadata):
    spacing = metadata['spacing']
    offset = metadata['phantom_offset']

    center, rods, rotation_angle, rod_spacing = cp404localizer(metadata)

    center_array_index = np.argmin(np.abs(ct_pos[:, 2] + offset))
    array = np.squeeze(ct_array[center_array_index, :, :])

    center = np.rint(center).astype(np.int)
    d15 = int(np.rint(12.5 / spacing[1]))
    d30 = int(np.rint(30.0 / spacing[1]))
#    d5 = np.rint(5.0 / dxdy)
    r5 = int(np.rint(5.0 / spacing[1]))

    results = Analysis('slice_thickness', metadata)

    results.arrays['referenced_images'] = np.array([ct_uids[center_array_index]], dtype='S64')

    # graphics items
    results.arrays['box_rois'] = np.zeros(4, dtype=[('x', np.float),('y', np.float),('z', np.float),('height', np.float),('width', np.float),('label', 'S32')])

    results.arrays['box_rois'][0] = (center[1] - d30, center[0] - d30 - d15, center[2], 2 * d30, d15, 'Region 1')
    results.arrays['box_rois'][1] = (center[1] + d30, center[0] - d30, center[2], d15,  2 * d30, 'Region 2')
    results.arrays['box_rois'][2] = (center[1] - d30, center[0] + d30, center[2], 2 * d30, d15, 'Region 3')
    results.arrays['box_rois'][3] = (center[1] - d30 - d15, center[0] - d30, center[2], d15,  2 * d30, 'Region 4')

    results.arrays['circle_rois'] = np.array([(center[0], center[1], center[2], r5, 'Background')], dtype=[('x', np.float),('y', np.float),('z', np.float),('radius', np.float),('label', 'S32')])


    # measurements
    q1 = array[center[0] - d30 - d15: center[0] - d30,
               center[1] - d30: center[1] + d30]
    q2 = array[center[0] - d30: center[0] + d30,
               center[1] + d30: center[1] + d30 + d15]
    q3 = array[center[0] + d30: center[0] + d30 + d15,
               center[1] - d30: center[1] + d30]
    q4 = array[center[0] - d30: center[0] + d30,
               center[1] - d30 - d15: center[1] - d30]

    background_mask = circle_indices(array.shape, r5, center[:2])
    bc = array[background_mask].mean()

    q = [q1, q2, q3, q4]
    FWHM = lambda b, bc: (b.max()+bc)/2.0
    p = [qq > FWHM(qq, bc) for qq in q]

    results.arrays['table'] = np.zeros((4, 7), dtype='S32')
    lenghts = np.array([np.count_nonzero(pp.sum(axis=i % 2)) for i, pp in enumerate(p)], dtype=np.float)
    lenghts *= spacing[1] * np.tan(np.deg2rad(23))
    lenght = lenghts.mean()
    results.add_data('slice_thickness_measured', lenght, np.float)
    results.arrays['table'][0, 1:] = ['Region {}'.format(i) for i in range(1,5)] + ['Mean', 'Nominal']
    results.arrays['table'][1:, 0] = ['Measured [mm]', 'Deviation [mm]', 'Deviation [%]']
    for i in range(4):
        results.arrays['table'][1, i+1] = str(round(lenghts[i], 3))
        results.arrays['table'][2, i+1] = str(round((spacing[0]-lenghts[i]), 3))
        results.arrays['table'][3, i+1] = str(round(100*(spacing[0]-lenghts[i])/spacing[0], 1))
    results.arrays['table'][1, 5] = str(round(lenght, 3))
    results.arrays['table'][1, 6] = str(round(spacing[0], 3))
    results.arrays['table'][2, 5] = str(round((spacing[0]-lenght), 3))
    results.arrays['table'][3, 5] = str(round(100*(spacing[0]-lenght)/spacing[0], 1))
    results.success = True
#    import pdb;pdb.set_trace()
    return results

def uniformity_analysis(ct_array, ct_pos, ct_uids, metadata):
    spacing = metadata['spacing']
    offset = metadata['phantom_offset']
    center, rods, rotation_angle, rod_spacing = cp404localizer(metadata)
    center_array_index = np.argmin(np.abs(ct_pos[:, 2] + offset - (- 160.)))
    center = np.rint(center).astype(np.int)

    ct_array_indices = image_world_transform(metadata['image_orientation'],
                                             metadata['image_position'],
                                             spacing,
                                             ct_pos,
                                             inverse=True)

    roi_radius = int(round(150. * .1 / 2. / spacing[1]))
    roi_distance = int(round(150. / 2. / spacing[1]) - roi_radius / 2.)

    analysis = Analysis('uniformity', metadata)
    analysis.arrays['circle_rois'] = np.zeros(5 * ct_array.shape[0], dtype=[('x', np.float),('y', np.float),('z', np.float),('radius', np.float),('label', 'S32')])


    measurements = np.zeros((ct_array.shape[0], 5))
    for i, dxdy in enumerate([(0, 0, 'Center'), (1, 0, '1'), (0, 1, '2'),(-1, 0, '3'),(0, -1, '4')]):
        pos = center[0] + dxdy[0] * roi_distance, center[1] + dxdy[1] * roi_distance
        index = circle_indices(ct_array.shape[1:], roi_radius, pos)
        for j in range(measurements.shape[0]):
#            import pdb;pdb.set_trace()
            analysis.arrays['circle_rois'][i * measurements.shape[0] + j] = (pos[0], pos[1], ct_array_indices[j, 2], roi_radius, dxdy[2])
            measurements[j, i] = ct_array[j, :, :][index].mean()

    analysis.arrays['plot_info'] = np.array(['Position [mm]', 'Deviation from center [HU]', 'Uniformity'])
    plot = np.zeros(measurements.shape[0], dtype=[('x', np.float),('y', np.float),('label', 'S32'),('xstd', np.float),('ystd', np.float)])
    plot['x'] = ct_pos[:, 2]
    measurements_diff = np.zeros((ct_array.shape[0], 4))
    for i in range(4):
        measurements_diff[:, i] = measurements[:, 0] - measurements[:, i+1]
    plot['y'] = measurements_diff.mean(axis=1)
    plot['ystd'] = measurements_diff.std(axis=1)
    analysis.arrays['plots'].append(plot)

    analysis.arrays['referenced_images'] = np.array(ct_uids, dtype='S64')

    analysis.arrays['table'] = np.zeros((3, 7), dtype='S32')
    analysis.arrays['table'][0, :] = ['', 'Region 1', 'Region 2', 'Region 3', 'Region 4', 'Mean', 'Center']
    analysis.arrays['table'][:, 0] = ['', 'CT number [HU]', 'Deviation center [HU]']
    for i in range(4):
        analysis.arrays['table'][1, i+1] = str(round(measurements[center_array_index, i+1], 2))
        analysis.arrays['table'][2, i+1] = str(round(measurements_diff[center_array_index, i], 2))
    analysis.arrays['table'][1, 5] = str(round(measurements[center_array_index, :4].mean(), 2))
    analysis.arrays['table'][1, 6] = str(round(measurements[center_array_index, 0], 2))
    analysis.arrays['table'][2, 5] = str(round(measurements_diff[center_array_index, :4].mean(), 2))
    import pdb;pdb.set_trace()
    return analysis
#def test_world_transform(ct_array, ct_pos, metadata):
#    rods_position = metadata['cp404_rods_position'].reshape((4, 3))
#    image_orientation = metadata['image_orientation']
#    image_position = metadata['image_position']
#    spacing = metadata['spacing']
#
##    image_position = ct_pos[0, :].ravel()
#    print(metadata['phantom_offset'], image_position)
#    print(rods_position)
#    print(image_world_transform(image_orientation, image_position, spacing,
#                                rods_position, inverse=True))
#
##    import pdb; pdb.set_trace()
#    corners = image_world_transform(image_orientation, image_position, spacing,
#                                    rods_position, inverse=True)
#
##    import pdb; pdb.set_trace()

ANALYSIS_REGIONS = [(linarity_analysis, 0, -10, 10),
                    (slice_thickness_analysis, 0, -10, 10),
                    (uniformity_analysis, -160, -10, 10)]

def analysis_dispatcher(metadata, positions):
    """
    Find available analysis for a series based on image position.
    Returns analysis function and min max of image indices
    """
    offset = metadata['phantom_offset']
    pos = positions[:, 2] + offset
    indices = np.arange(positions.shape[0], dtype=np.int)
    for analysis, center, minval, maxval in ANALYSIS_REGIONS:
        req_ind = np.logical_and(pos > (center+minval ), pos < (center+maxval))
#        import pdb;pdb.set_trace()
        if np.any(req_ind):
            min_ind = indices[req_ind].min()
            max_ind = indices[req_ind].max()
            yield (analysis, min_ind, max_ind)


def analyse_all(database):
    for series in database.get_series():
        metadata = database.get_series_metadata(series)
        pos = database.get_series_array(metadata['study_uid'], series, 'image_positions')

        if metadata['is_localized']:
            for analysis, min_ind, max_ind in analysis_dispatcher(metadata, pos):
                ctarray, ct_uids, ct_pos = database.get_series_array(metadata['study_uid'],
                                                             series,
                                                             'ctarray',
                                                             'image_uids',
                                                             'image_positions',
                                                             start=min_ind,
                                                             stop=max_ind)
                res = analysis(ctarray, ct_pos, ct_uids, metadata)
                database.add_analysis(res)
















