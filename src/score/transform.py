"""
Manipulate atom coordinates with matrix operations.

Provides array methods to translate and rotate atoms in a
structure.Structure.  The coin of the realm is a four by four numpy array.
The first 3x3 is a rotation matrix, the last 3x1 is translation matrix,
and the 4th row is a spectator row.

All angles are in radians.

    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0.  1.]] # spectator row

The elements in the last row are spectators and don't contribute to
atomic coordinate changes.

Copyright Schrodinger, LLC. All rights reserved.

"""

import math
import numpy

from schrodinger.infra import mm

X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def get_vector_magnitude(a):
    """
    Returns magnitute of a specified vector (numpy array)
    """
    return numpy.linalg.norm(a)


def get_normalized_vector(vector):
    """
    Returns normalized version of the specified vector (numpy array)
    """
    length = get_vector_magnitude(vector)
    if length == 0.0:
        # Fix for Ev:118884
        return vector
    else:
        return vector / length


def get_angle_between_vectors(a, b):
    """
    Return angle between 2 vectors
    """
    # cos(angle)  = dot(a, b) / ( mag1 * mag2)

    mag_mult = get_vector_magnitude(a) * get_vector_magnitude(b)
    if mag_mult == 0.0:
        raise ValueError("Can not calculate angle when either vector is null")

    cosangle = numpy.dot(a, b) / mag_mult
    cosangle = max(min(cosangle, 1), -1)  # keep in [-1, 1] bound
    return numpy.arccos(cosangle)


def translate_structure(st, x=0.0, y=0.0, z=0.0, atom_index_list=None):
    """
    Translates the atom coordinates along Cartesian x, y, and z axes.

    st (structure.Structure)

    x (float)
        Distance, in angstroms, along positive x to translate.

    y (float)
        Distance, in angstroms, along positive y to translate.

    z (float)
        Distance, in angstroms, along positive z to translate.

    atom_index_list (list)
        Integer indexes for the atoms to transform.  If the list is not
        specified then all atoms in the structure are transformed. If the
        list is empty, none of the atoms are transformed.

    """

    trans_matrix = get_translation_matrix([x, y, z])
    transform_structure(st, trans_matrix, atom_index_list)


def rotate_structure(st, x_angle=0, y_angle=0, z_angle=0, rot_center=None):
    """
    Rotates the structure about x axis, then y axis, then z axis.

    st (structure.Structure)

    x_angle (float)
        Angle, in radians, about x to right-hand rotate.

    y_angle (float)
        Angle, in radians, about y to right-hand rotate.

    z_angle (float)
        Angle, in radians, about z to right-hand rotate.

    rot_center (list)
        Cartesian coordinates (x, y, z) for the center of rotation.
        By default, rotation happens about the origin (0, 0, 0)

    """

    # This action is achieved in four steps
    # 1)  Find the vector that moves the rot_center to the origin
    # 2)  Move the structure along that vector
    # 3)  Apply rotations
    # 4)  Move the structure back

    # Adjust the center of rotation if needed.
    displacement_vector = None
    if rot_center:
        # FIXME skip if already at origin
        displacement_vector = get_coords_array_from_list(rot_center)
        to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
        transform_structure(st, to_origin_matrix)

    # Apply rotation
    rot_matrix_x = get_rotation_matrix(X_AXIS, x_angle)
    rot_matrix_y = get_rotation_matrix(Y_AXIS, y_angle)
    rot_matrix_z = get_rotation_matrix(Z_AXIS, z_angle)

    transform_structure(st, rot_matrix_x)
    transform_structure(st, rot_matrix_y)
    transform_structure(st, rot_matrix_z)

    # FOR SOME REASON NEITHER OF THESE WILL NOT WORK:
    #combined_rot_matrix = rot_matrix_x * rot_matrix_y * rot_matrix_z
    #combined_rot_matrix = numpy.dot( numpy.dot(rot_matrix_x, rot_matrix_y), rot_matrix_z)
    #transform_structure(st, combined_rot_matrix)

    # Move the structure back, if needed
    if rot_center:
        # FIMXE skip if already ot origin
        from_origin_matrix = get_translation_matrix(displacement_vector)
        transform_structure(st, from_origin_matrix)

    return


def transform_atom_coordinates(coords, matrix):
    """
    Transforms the specified atom coordinates using a 4x4 transformation matrix.

    :param coords: Coordinate array (x, y, z)
    :type coords: numpy.array

    :param matrix: 4x4 numpy array representation of transformation matrix.
    :type matrix: numpy.array

    :return: Transformed coordinates
    :rtype: numpy.array
    """

    coords4 = numpy.array(coords)
    # Add 4-th element:
    coords4.resize(4)
    coords4[3] = 1.0

    new_coords = numpy.dot(matrix, coords4)

    return new_coords[0:3]


def transform_structure(st, matrix, atom_index_list=None):
    """
    Transforms atom coordinates of the structure using a 4x4
    transformation matrix.  An optional list of atom numbers defining
    a subset of atoms for transformation may be specified.

    st (structure.Structure)

    matrix (numpy.array)
        4x4 numpy array representation of transformation matrix.

    atom_index_list (list)
        Integer indexes for the atoms to transform.  If the list is not
        specified then all atoms in the structure are transformed. If the
        list is empty, none of the atoms are transformed.

    """

    # Modifying this array will directly alter the actual coordinates:
    atom_xyz_array = st.getXYZ(copy=False)

    if atom_index_list is None:
        atom_index_list = list(range(1, st.atom_total + 1))

    # TODO Consider using mmct_ct_transform()
    for at_index in atom_index_list:
        coords = atom_xyz_array[at_index - 1]
        new_coords = transform_atom_coordinates(coords, matrix)
        numpy.put(coords, range(3), new_coords)

        # Ev:118675 If the atom has alternate position, transform it as well:
        # FIXME Is there a way we can speed this up by doing more on the C side??
        if mm.mmct_atom_has_alt_position(st.handle, at_index):
            xyz = mm.mmct_atom_get_alt_xyz(st.handle, at_index)
            xyz = transform_atom_coordinates(xyz, matrix)
            mm.mmct_atom_set_alt_xyz(st.handle, at_index, xyz)
    return


def get_centroid(st, atom_list=None):
    """
    Returns the structure's centroid as a 4-element numpy array::

        [x y z 0.0]

    NOTE: Periodic boundary conditions (PBC) are NOT honored.

    :type st: structure.Structure

    :type atom_list: list(int)
    :param atom_list: A list of 1-based atom indices. If provided, the centroid
                      of the atoms in this list will be calculated instead of
                      the centroid of all atoms.
    """

    center_coords = numpy.zeros((4), 'd')  # four floats

    # Returns a numpy array of (x,y,z) arrays:
    if (atom_list):
        atom_list = numpy.array(atom_list)
        atom_list -= 1
        atom_xyz_array = st.getXYZ(copy=False)[atom_list]
    else:
        atom_xyz_array = st.getXYZ(copy=False)

    # Will return averages of X, Y, and Z coordinates (as 3-item array):\
    averages = numpy.average(atom_xyz_array, 0)  # axis of 0 means top-level

    center_coords[0] = averages[0]
    center_coords[1] = averages[1]
    center_coords[2] = averages[2]

    return center_coords


def translate_center_to_origin(st, origin=None):
    """
    Translates the structure's center to the origin. The difference between this
    function and `translate_centroid_to_origin` is that the
    centroid is the average of all atoms, whereas the center is the middle of
    the atoms. The centroid can be very far from the center for structures with
    a high percent of the atoms located in one region of space and a few atoms
    very far away.

    :type st: `structure.Structure`
    :param st: Structure that will modified

    :type origin: list(3 floats)
    :param orgin: Coordinates of the new origin
    """

    if not origin:
        origin = [0., 0., 0.]

    if len(origin) != 3:
        raise RuntimeError("'origin' must be an array of three floats")

    centroid = numpy.array(origin)

    coords = st.getXYZ()
    mins = coords.min(0)
    # ptp gives the span of each X, Y or Z coordinate
    spans = coords.ptp(0)

    center = numpy.array([m + 0.5 * s for m, s in zip(mins, spans)])
    movement = centroid - center

    st.setXYZ(coords + movement)
    return


def translate_centroid_to_origin(st, atom_list=None):
    """
    Translates the structure's centroid to the origin.

    :type st: structure.Structure

    :type atom_list: list(int)
    :param atom_list: A list of 1-based atom indices. If provided, the centroid
                      of the atoms in this list will be translated to the
                      origin.
    """

    new_origin = get_centroid(st, atom_list)
    to_origin_matrix = get_translation_matrix(-1 * new_origin)
    transform_structure(st, to_origin_matrix, atom_list)
    return


# For backward-compatability:
translate_to_origin = translate_centroid_to_origin


def get_translation_matrix(trans):
    """
    Returns a 4x4 numpy array representing a translation matrix from
    a 3-element list.

    trans (list)
        3-element list (x,y,z).
    """

    trans_matrix = numpy.identity(4, 'd')  # four floats
    trans_matrix[0][3] = float(trans[0])
    trans_matrix[1][3] = float(trans[1])
    trans_matrix[2][3] = float(trans[2])
    return trans_matrix


def get_rotation_matrix(axis, angle):
    """
    Returns a 4x4 numpy array representing a right-handed rotation
    matrix about the specified axis running through the origin by some angle

    axis (vector)
        Normalized (unit) vector for the axis around which to rotate.
        Can be one of predefined axis: X_AXIS, Y_AXIS, Z_AXIS, or arbitrary
        axis.

    angle (float)
        Angle, in radians, about which to rotate the structure about
        the axis.
    """

    # From: http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    #
    # Rotation matrix =
    #
    # t*x*x + c       t*x*y - z*s     t*x*z + y*s
    # t*x*y + z*s     t*y*y + c       t*y*z - x*s
    # t*x*z - y*s     t*y*z + x*s     t*z*z + c
    #
    # where,
    #
    # * c = cos(angle)
    # * s = sin(angle)
    # * t = 1 - c
    # * x = normalised x portion of the axis vector
    # * y = normalised y portion of the axis vector
    # * z = normalised z portion of the axis vector

    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x = axis[0]
    y = axis[1]
    z = axis[2]

    rot_matrix = numpy.identity(4, 'd')  # four floats
    rot_matrix[0] = [t * x * x + c, t * x * y - z * s, t * x * z + y * s, 0.0]
    rot_matrix[1] = [t * x * y + z * s, t * y * y + c, t * y * z - x * s, 0.0]
    rot_matrix[2] = [t * x * z - y * s, t * y * z + x * s, t * z * z + c, 0.0]

    return rot_matrix


def get_rotation_matrix_from_eulers(phi, theta, psi):
    """
    Returns a 4x4 numpy array representing a rotation matrix generated
    from a list of Euler angles.  The angles of rotation (phi, theta
    and psi) are applied in order, and are defined as:

    phi
        Angle to rotate by about Z axis [0 to 2pi in radians]

    theta
        Angle to rotate by about X' axis [0 to 2pi in radians]

    psi
        Angle to rotate by about Z' axis [0 to 2pi in radians]

    """

    # For further reference and formulae of the rotation matrix elements, see:
    # http://mathworld.wolfram.com/EulerAngles.html (eqs 6-14)
    phi = float(phi)
    theta = float(theta)
    psi = float(psi)

    cospsi = math.cos(psi)
    cosphi = math.cos(phi)
    costheta = math.cos(theta)
    sinphi = math.sin(phi)
    sinpsi = math.sin(psi)
    sintheta = math.sin(theta)

    rot_matrix = numpy.identity(4, 'd')  # four floats
    rot_matrix[0][0] = cospsi * cosphi - costheta * sinphi * sinpsi
    rot_matrix[0][1] = cospsi * sinphi + costheta * cosphi * sinpsi
    rot_matrix[0][2] = sinpsi * sintheta
    rot_matrix[1][0] = -1.0 * sinpsi * cosphi - costheta * sinphi * cospsi
    rot_matrix[1][1] = costheta * cosphi * cospsi - sinpsi * sinphi
    rot_matrix[1][2] = cospsi * sintheta
    rot_matrix[2][0] = sintheta * sinphi
    rot_matrix[2][1] = -1.0 * sintheta * cosphi
    rot_matrix[2][2] = costheta

    return rot_matrix


def get_coords_array_from_list(coords_list):
    """
    Returns coordinates as a 4-element numpy array: (x,y,z,0.0).

    coords_list (list or array)
        3 elements: x, y, z.

    """

    coords = numpy.zeros((4), 'd')  # four floats
    coords[0] = coords_list[0]
    coords[1] = coords_list[1]
    coords[2] = coords_list[2]
    return coords


def get_alignment_matrix(a_vector, b_vector):
    """
    Returns a Numpy 4x4 rotation matrix that will align a_vector
    onto b_vector.

    a_vector (array)
    numpy array of vector coordinates (x, y, z)

    b_vector (array)
    numpy array of vector coordinates (x, y, z)

    """

    # Calculate the normal to of a_vector x b_vector, this is the
    # axis of rotation to align a_vector:
    normal_vector = numpy.cross(a_vector, b_vector)

    # Convert to a unit vector (normalize):
    normal_vector = get_normalized_vector(normal_vector)

    # Angle to rotate by:
    angle_rad = get_angle_between_vectors(a_vector, b_vector)

    # Return the rotation matrix:
    matrix = get_rotation_matrix(normal_vector, angle_rad)

    return matrix


def get_reflection_matrix(reflect_axis, axis_origin=None):
    """
    Returns a 4x4 Numpy matrix which will reflect all points through a
    mirror plane (defined by a unit vector normal to that plane and a
    point in the plane).

    reflect_axis (array, len 3)
    Normalized (unit) vector defining the mirror plane

    axis_origin (array, len 3)
    point which lies in the mirror plane, if None, origin will be used
    """

    if axis_origin is None:
        axis_origin = numpy.zeros(3)

    d = numpy.dot(reflect_axis, axis_origin)
    rx = reflect_axis[0]
    ry = reflect_axis[1]
    rz = reflect_axis[2]

    reflect_mat = numpy.array(
        [[1.0 - 2.0 * rx * rx, -2.0 * rx * ry, -2.0 * rx * rz, 2.0 * rx * d],
         [-2.0 * rx * ry, 1.0 - 2.0 * ry * ry, -2.0 * ry * rz, 2.0 * ry * d],
         [-2.0 * rx * rz, -2.0 * ry * rz, 1.0 - 2.0 * rz * rz,
          2.0 * rz * d], [0.0, 0.0, 0.0, 1.0]])
    return reflect_mat
