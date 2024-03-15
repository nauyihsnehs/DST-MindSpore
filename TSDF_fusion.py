import open3d as o3d
import numpy as np
import pdb
from scipy.spatial.transform import Rotation as R

class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)
def quaternion_to_rotation_matrix(q):  # x, y ,z ,w
    rot_matrix = np.array(
        [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])],
         [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0])],
         [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]],
        dtype=np.float32)
    return rot_matrix

def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(float, metastr.split()))
            #pdb.set_trace()
            q=[metadata[4],metadata[5],metadata[6],metadata[7]]
            #r1 = R.from_quat(q)
            #r1.as_matrix()
            mat3x3=quaternion_to_rotation_matrix(q)
            xyz=[metadata[1],metadata[2],metadata[3]]

            mat4x4 = np.zeros(shape=(4, 4))
            mat4x4[0:3,0:3]=mat3x3[:,:]
            mat4x4[0,3]=xyz[0]
            mat4x4[1,3]=xyz[1]
            mat4x4[2,3]=xyz[2]
            mat4x4[3,3]=1
            #pdb.set_trace()
            # for i in range(7):
            #     matstr = f.readline()
            #     mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            #traj.append(CameraPose(metadata, mat))
            traj.append(CameraPose(xyz,mat4x4))
            metastr = f.readline()
    return traj
def read_trajectory_sim(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(float, metastr.split()))
            metastr1 = f.readline()
            metadata1 = list(map(float, metastr1.split()))
            metastr2 = f.readline()
            metadata2 = list(map(float, metastr2.split()))
            metastr3 = f.readline()
            #pdb.set_trace()
            # q=[metadata[4],metadata[5],metadata[6],metadata[7]]
            # #r1 = R.from_quat(q)
            # #r1.as_matrix()
            # mat3x3=quaternion_to_rotation_matrix(q)
            xyz=[0,0,0]
            mat3x4=np.array([metadata,metadata1,metadata2])
            mat4x4 = np.zeros(shape=(4, 4))
            mat4x4[0:3,0:4]=mat3x4[:,:]
            #pdb.set_trace()
            mat4x4[3,3]=1
            traj.append(CameraPose(xyz,mat4x4))
            metastr = f.readline()
    return traj
#redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
# read camera pose
camera_path="/mnt/data3/wzq/kt1-TUM-rgbdcompatiblepngswithnoise/livingRoom1n.gt.freiburg"
camera_path_sim="/mnt/data3/wzq/kt0-TUM-rgbdcompatiblepngswithnoise/livingRoom0n.gt.sim"
#camera_path="/mnt/data3/wzq/kt1-TUM-rgbdcompatiblepngswithnoise/livingRoom1n.gt.freiburg"
camera_poses = read_trajectory(camera_path)
#pdb.set_trace()

#TSDF volume integration
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.2,
    sdf_trunc=0.6,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

root_path="/mnt/data3/wzq/kt1-TUM-rgbdcompatiblepngswithnoise/" #1509
#root_path="/mnt/data3/wzq/kt1-TUM-rgbdcompatiblepngswithnoise/"
for i in range(len(camera_poses)):
    print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image(
        root_path+"rgb/"+str(i+1)+".png")
    depth = o3d.io.read_image(
        root_path+"depth/"+str(i+1)+".png")
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
       color, depth, depth_trunc=400, convert_rgb_to_intensity=False)
    volume.integrate(
        rgbd,
        # o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        o3d.camera.PinholeCameraIntrinsic(640,480,np.array([[481.20 ,   0. , 319.5],[  0. ,-480.00 , 239.5],[  0. ,   0. ,   1. ]])),
        np.linalg.inv(camera_poses[i].pose))
        # camera_poses[i].pose)
    # pdb.set_trace()
#mesh extract
pdb.set_trace()

volume.save(root_path+'voxel.npz')
volume1= o3d.geometry.VoxelBlockGrid.load(root_path+'voxel.npz')
mesh = volume.extract_triangle_mesh()
filename=root_path+'1.obj'

#pdb.set_trace()
mesh.compute_vertex_normals()
#o3d.iowrite_triangle_mesh(filename, mesh, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)
o3d.io.write_triangle_mesh(filename, mesh)
# o3d.visualization.draw_geometries([mesh],
#                                   front=[0.5297, -0.1873, -0.8272],
#                                   lookat=[2.0712, 2.0312, 1.7251],
#                                   up=[-0.0558, -0.9809, 0.1864],
#                                   zoom=0.47)
