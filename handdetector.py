import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_threshold=0.5,
                 tracking_threshold=0.5, point_color=(0,0,255), connections_color=(255,255,255), ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_threshold = detection_threshold
        self.tracking_threshold = tracking_threshold
        self.point_color = point_color
        self.connections_color = connections_color

        self.hands_mp = mp.solutions.hands
        self.hands = self.hands_mp.Hands(
            self.mode,
            self.max_hands,
            1,
            self.detection_threshold,
            self.tracking_threshold,
        )

        self.drawing_mp = mp.solutions.drawing_utils
        self.drawing_points_config = self.drawing_mp.DrawingSpec(color=self.point_color)
        self.drawing_connections_config = self.drawing_mp.DrawingSpec(color=self.connections_color)

    def find_hands(self, image, drawing=True):
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_RGB)
        if self.results.multi_hand_landmarks:
            for points in self.results.multi_hand_landmarks:
                if drawing:
                    self.drawing_mp.draw_landmarks(image, points, self.hands_mp.HAND_CONNECTIONS, self.drawing_points_config, self.drawing_connections_config)
        return image

    def find_points(self, image, drawing=True, color=(255, 0, 0), radius=7):
        height, width, _ = image.shape
        empty_hand = [(0, 0, 0)] * 21

        hands = {"Left": empty_hand, "Right": empty_hand}

        if self.results.multi_hand_landmarks and self.results.multi_handedness:
            for hand_landmarks, handedness in zip(
                    self.results.multi_hand_landmarks, self.results.multi_handedness
            ):
                label = handedness.classification[0].label

                hand = []
                for point in hand_landmarks.landmark:
                    center_x = int(point.x * width)
                    center_y = int(point.y * height)
                    center_z = int(point.z * width)
                    hand.append((center_x, center_y, center_z))

                hands[label] = hand

        return hands["Left"] + hands["Right"]

def main():
    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while cap.isOpened():
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        image = detector.find_hands(image)
        point_list = detector.find_points(image)
        print(point_list)
        cv2.imshow('Hand Tracking', image)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()