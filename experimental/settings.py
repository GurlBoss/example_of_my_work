
from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point, Vector3
class FrameAdv():
    ''' Advanced variables derived from frame object
    '''
    def __init__(self):
        self.l = HandAdv()
        self.r = HandAdv()

class HandAdv():
    ''' Advanced variables of hand derived from hand object
    '''
    def __init__(self):
        self.visible = False # If this hand (Left/Right) is visible
        self.conf = 0.0 # <0.0-1.0> Confidence of hand recognition
        self.OC = [0.0] * 5 # Probability of finger opened <0.0-1.0>, for each finger
        self.TCH12, self.TCH23, self.TCH34, self.TCH45 = [0.0] * 4 # Touch probability of e.g.TCH12 (first finger and second finger touch <0.0,1.0>)
        self.TCH13, self.TCH14, self.TCH15 = [0.0] * 3
        self.vel = [0.0] * 3 # Velocity of palm [x,y,z]
        self.pPose = PoseStamped() # Pose of palm (ROS)
        self.pRaw = [0.0] * 6 # palm pose: x, y, z, roll, pitch, yaw
        self.pNormDir = [0.0] * 6 # palm normal vector and direction vector
        # Extra data
        self.rot = Quaternion()
        self.rotRaw = [0.0] * 3
        self.rotRawEuler = [0.0] * 3
        # Experimental
        self.time_last_stop = 0.0

        self.grab = 0.0
        self.pinch = 0.0

        ## Data processed for learning
        # direction vectors
        self.wrist_hand_angles_diff = []
        self.fingers_angles_diff = []
        self.pos_diff_comb = []
        #self.pRaw
        self.index_position = []
