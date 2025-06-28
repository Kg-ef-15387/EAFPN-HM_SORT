import copy
import numpy as np
from collections import deque
from boxmot.motion.cmc import get_cmc_method
from boxmot.motion.kalman_filters.xysr_kf import KalmanFilterXYSR
from boxmot.utils.association import associate, hm_associate, linear_assignment
from boxmot.utils.iou import get_asso_func
from boxmot.utils.iou import run_asso_func
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils import PerClassDecorator
from boxmot.utils.ops import xyxy2xysr


def k_previous_obs(observations, cur_age, k):

    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def img_extract(box_xyxy, img):
    box_xywh = [box_xyxy[0], box_xyxy[1], box_xyxy[2]-box_xyxy[0], box_xyxy[3]-box_xyxy[1]]
    img_crop = copy.deepcopy(img[round(box_xywh[1]):round(box_xywh[1] + box_xywh[3]), round(box_xywh[0]):round(box_xywh[0] + box_xywh[2])])
    return img_crop


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self,
                 bbox,
                 cls,
                 det_ind,
                 img,
                 delta_t=3,
                 max_obs=50,
                 Q_xy_scaling = 0.01,
                 Q_s_scaling = 0.0001):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        self.det_ind = det_ind

        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling

        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=max_obs)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0

        self.kf.Q[4:6, 4:6] *= self.Q_xy_scaling
        self.kf.Q[-1, -1] *= self.Q_s_scaling

        self.kf.x[:4] = xyxy2xysr(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.max_obs = max_obs
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = bbox[-1]
        self.cls = cls
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)
        self.velocity = None
        self.delta_t = delta_t
        self.img_crop = img_extract(bbox, img)
        # import matplotlib.pyplot as plt
        # plt.subplot(121), plt.title('original_img')
        # plt.imshow(img[:, :, ::-1])
        # plt.subplot(122), plt.title('crop')
        # plt.imshow(self.img_crop[:, :, ::-1])
        # plt.show()


    def update(self,
               bbox,
               cls,
               det_ind,
               img
               ):
        """
        Updates the state vector with observed bbox.
        """
        self.det_ind = det_ind
        if bbox is not None:
            self.conf = bbox[-1]
            self.cls = cls
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)

            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(xyxy2xysr(bbox))
            match_img_crop = img_extract(bbox, img)
            # import matplotlib.pyplot as plt
            # plt.subplot(131), plt.title('original_img')
            # plt.imshow(img[:, :, ::-1])
            # plt.subplot(132), plt.title('original_crop')
            # plt.imshow(self.img_crop[:, :, ::-1])
            # plt.subplot(133), plt.title('current_crop')
            # plt.imshow(match_img_crop[:, :, ::-1])
            # plt.show()
            self.img_crop = match_img_crop
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    # CMC
    def apply_affine_correction(self, affine):
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        # For OCR
        # OCR + CMC
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        # Apply to each box in the range of velocity computation
        # OCM + CMC
        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

        # Also need to change kf state, but might be frozen
        self.kf.apply_affine_correction(m, t)


class HMSort(BaseTracker):
    def __init__(
        self,
        per_class=False,
        det_thresh=0.2,
        max_age=30,
        min_hits=3,
        asso_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_cascade=True,
        use_ssim=False,
        ssim_ratio=0.1,
        use_vlad=False,
        vlad_ratio=0.1,
        use_byte=False,
        Q_xy_scaling=0.01,
        Q_s_scaling=0.0001
    ):
        super().__init__(max_age=max_age)
        self.per_class = per_class
        self.max_age = max_age
        self.min_hits = min_hits
        self.asso_threshold = asso_threshold
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = get_asso_func(asso_func)
        self.inertia = inertia
        self.use_cascade=use_cascade
        self.use_ssim = use_ssim,
        self.ssim_ratio = ssim_ratio,
        self.use_vlad = use_vlad,
        self.vlad_ratio = vlad_ratio,
        self.use_byte = use_byte
        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling
        self.cmc = get_cmc_method('sof')()
        self.cmc_off = False
        KalmanBoxTracker.count = 0


    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections
        (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        self.frame_count += 1

        print('------------------------:{}'.format(self.frame_count))

        h, w = img.shape[0:2]

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4]

        inds_low = confs > 0.1
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(
            inds_low, inds_high
        )  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = confs > self.det_thresh
        dets = dets[remain_inds]

        dets_img_crop_list = [img_extract(i, img) for i in dets]
        # for img in dets_img_crop_list:
        #     import cv2
        #     cv2.namedWindow("img_crop", cv2.WINDOW_NORMAL)
        #     cv2.imshow("img_crop", img)
        #     cv2.waitKey(0)

        trks_img_crop_list = [trk.img_crop for trk in self.active_tracks]

        trks_ind_list = [i for i in range(len(self.active_tracks))]
        dets_ind_list = [i for i in range(len(dets))]

        if not self.cmc_off:
            transform = self.cmc.apply(img, dets[:, :4])
            for trk in self.active_tracks:
                trk.apply_affine_correction(transform)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.active_tracks), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array((0, 0))
                for trk in self.active_tracks
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])
        k_observations = np.array(
            [
                k_previous_obs(trk.observations, trk.age, self.delta_t)
                for trk in self.active_tracks
            ]
        )

        # First Match
        if not self.use_cascade:
            matched, unmatched_dets, unmatched_trks = associate(
                dets[:, 0:5], trks, self.asso_func, self.asso_threshold, velocities, k_observations, self.inertia, w, h
            )

        # ----------------------------------Cascade Match----------------------------------
        else:
            matched = []
            unmatched_dets = dets_ind_list.copy()
            unmatched_trks = trks_ind_list.copy()

            cascade_depth = 30

            for level in range(cascade_depth):
                if len(unmatched_dets) == 0:  # No detections left
                    break
                track_indices_l = [
                    k for k in trks_ind_list
                    if self.active_tracks[k].time_since_update == 1 + level
                ]
                if len(track_indices_l) == 0:  # Nothing to match at this level
                    continue
                current_dets = np.array([dets[i] for i in unmatched_dets])
                current_dets_img = [dets_img_crop_list[i] for i in unmatched_dets]
                current_trks = np.array([trks[i] for i in track_indices_l])
                current_velocities = np.array([velocities[i] for i in track_indices_l])
                current_k_observations = np.array([k_observations[i] for i in track_indices_l])
                current_trks_img = [trks_img_crop_list[i] for i in track_indices_l]

                current_matched, current_unmatched_dets, current_unmatched_trks = hm_associate(
                    current_dets[:, 0:5], current_trks, self.asso_func, self.asso_threshold, current_velocities,
                    current_k_observations, self.inertia, w, h, self.use_ssim, self.ssim_ratio, self.use_vlad,
                    self.vlad_ratio, detection_img=current_dets_img, track_img=current_trks_img
                )

                for match in current_matched:
                    det_ind = unmatched_dets[match[0]]
                    match[0] = det_ind
                    trk_ind = track_indices_l[match[1]]
                    match[1] = trk_ind
                    matched.append(match)

                for match in matched:

                    det_ind = match[0]
                    trk_ind = match[1]

                    if det_ind in unmatched_dets:
                        unmatched_dets.remove(det_ind)
                    if trk_ind in unmatched_trks:
                        unmatched_trks.remove(trk_ind)

            matched = np.array(matched)
            unmatched_dets = np.array(unmatched_dets)
            unmatched_trks = np.array(unmatched_trks)
        # ----------------------------------Cascade Match----------------------------------

        # if matched.shape[0] != 0 and unmatched_dets.shape[0] != 0 and unmatched_trks.shape[0] != 0:
        #     print('---------------------')

        for m in matched:
            self.active_tracks[m[1]].update(dets[m[0], :5], dets[m[0], 5], dets[m[0], 6], img)

        # Second Match
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(
                dets_second, u_trks
            )
            iou_left = np.array(iou_left)
            if iou_left.max() > self.asso_threshold:
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.active_tracks[trk_ind].update(
                        dets_second[det_ind, :5], dets_second[det_ind, 5], dets_second[det_ind, 6], img
                    )
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = run_asso_func(self.asso_func, left_dets, left_trks, w, h)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.asso_threshold:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.active_tracks[trk_ind].update(dets[det_ind, :5], dets[det_ind, 5], dets[det_ind, 6], img)
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(
                    unmatched_dets, np.array(to_remove_det_indices)
                )
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )

        for m in unmatched_trks:
            self.active_tracks[m].update(None, None, None, img)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :5],
                                   dets[i, 5],
                                   dets[i, 6],
                                   img=img,
                                   delta_t=self.delta_t,
                                   Q_xy_scaling=self.Q_xy_scaling,
                                   Q_s_scaling=self.Q_s_scaling,
                                   max_obs=self.max_obs)
            self.active_tracks.append(trk)

        i = len(self.active_tracks)

        # if i > 0:
        #     crop = self.active_tracks[0].img_crop
        #     import cv2
        #     cv2.namedWindow("track", cv2.WINDOW_NORMAL)
        #     cv2.imshow("track", crop)
        #     cv2.waitKey(2)

        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive
                ret.append(
                    np.concatenate((d, [trk.id + 1], [trk.conf], [trk.cls], [trk.det_ind])).reshape(
                        1, -1
                    )
                )
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.array([])