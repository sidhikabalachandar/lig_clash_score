from schrodinger.structutils.transform import get_centroid
import math
import sys
sys.path.insert(1, '../util')
from schrod_replacement_util import *


X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def create_pose(c, grid_loc_x, grid_loc_y, grid_loc_z, rot_x, rot_y, rot_z):
    translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)
    conformer_center = list(get_centroid(c))
    coords = c.getXYZ(copy=True)

    displacement_vector = get_coords_array_from_list(conformer_center)
    to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
    from_origin_matrix = get_translation_matrix(displacement_vector)
    rot_matrix_x = get_rotation_matrix(X_AXIS, math.radians(rot_x))
    rot_matrix_y = get_rotation_matrix(Y_AXIS, math.radians(rot_y))
    rot_matrix_z = get_rotation_matrix(Z_AXIS, math.radians(rot_z))
    new_coords = rotate_structure(coords, from_origin_matrix, to_origin_matrix, rot_matrix_x,
                                  rot_matrix_y, rot_matrix_z)

    return new_coords


def parse_name(name):
    conformer_index, grid_loc, rot = name.split('_')
    conformer_index = int(conformer_index)
    grid_loc_x, grid_loc_y, grid_loc_z = grid_loc.split(',')
    grid_loc_x = int(grid_loc_x)
    grid_loc_y = int(grid_loc_y)
    grid_loc_z = int(grid_loc_z)
    rot_x, rot_y, rot_z = rot.split(',')
    rot_x = int(rot_x)
    rot_y = int(rot_y)
    rot_z = int(rot_z)

    return conformer_index, grid_loc_x, grid_loc_y, grid_loc_z, rot_x, rot_y, rot_z